{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Check (checkModule) where

import Control.Applicative
import Control.Monad.Except (foldM, lift, unless, when)
import Control.Monad.Trans.State
--import Debug.Trace (trace)
import Data.Foldable (traverse_)
import Data.Maybe (fromJust, isJust, isNothing)
import Data.Traversable (for)
import Data.Tuple (swap)
import Data.Void

import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Text.Lazy as T

import Env
import PPrint
import Syntax
import Util

-- In a given callable scope in Magnolia, the following variable-related rules
-- apply:
--   (1) variables tagged as MUnk exist, but can not be used
--   (2) shadowing of variables is forbidden
-- TODO: hold names in scope to avoid inconsistencies (like order of decls
--       mattering
type VarScope = Env (TypedVar PhCheck)
type ModuleScope = Env [MDecl PhCheck]

-- TODO: this is a hacky declaration for predicate operators, which should
-- be available in any module scope. We are still thinking about what is the
-- right way to do this. For now, we hardcode these functions here.

hackyPrelude :: [TCDecl]
hackyPrelude = map (CallableDecl . Ann [LocalDecl Nothing]) (unOps <> binOps)
  where
    lhsVar = Ann Pred $ Var MObs (VarName "#pred1#") Pred
    rhsVar = Ann Pred $ Var MObs (VarName "#pred2#") Pred
    mkFn args nameStr =
      Callable Function (FuncName nameStr) args Pred Nothing ExternalBody
    unOps = [mkFn [lhsVar] "!_"]
    binOps = map (\s -> mkFn [lhsVar, rhsVar] ("_" <> s <> "_"))
      ["&&", "||", "!=", "=>", "<=>"]


-- TODO: enforce type in scope?
initScope :: [TypedVar p] -> VarScope
initScope = M.fromList . map mkScopeVar
  where
    mkScopeVar :: TypedVar p -> (Name, TypedVar PhCheck)
    mkScopeVar (Ann _ (Var mode name typ)) =
      (name, Ann { _ann = typ, _elem = Var mode name typ })

mkScopeVarsImmutable :: VarScope -> VarScope
mkScopeVarsImmutable = M.map mkImmutable
  where
    mkImmutable (Ann src (Var mode name ty)) =
      let newMode = case mode of MOut -> MUnk ; MUnk -> MUnk ; _ -> MObs
      in Ann src (Var newMode name ty)

-- TODO: can variables exist without a type in the scope?
-- TODO: replace things with relevant sets
-- TODO: handle circular dependencies
checkModule
  :: Env [TCTopLevelDecl]
  -> MModule PhParse
  -> MgMonad (Env [TCTopLevelDecl])
checkModule tlDecls (Ann src (RefModule typ name ref)) = do
  -- TODO: cast module: do we need to check out the deps?
  ~(Ann _ (MModule _ _ decls deps)) <-
    lookupTopLevelRef src (M.map getModules tlDecls) ref >>= castModule typ
  let renamedModule = MModule typ name decls deps :: MModule' PhCheck
      renamedModuleDecl = MModuleDecl $ Ann (LocalDecl src) renamedModule
  return $ M.insertWith (<>) name [renamedModuleDecl] tlDecls

checkModule tlDecls (Ann modSrc (MModule moduleType moduleName decls deps)) = do
  when (maybe False (any isLocalDecl . getModules)
              (M.lookup moduleName tlDecls)) $
    throwLocatedE MiscErr modSrc $ "duplicate local module name " <>
      pshow moduleName <> " in package."
  let callables = getCallableDecls decls
  -- Step 1: expand uses
  (depScope, checkedDeps) <- do
      -- TODO: check here that we are only importing valid types of modules.
      -- For instance, a program can not be imported into a concept, unless
      -- downcasted explicitly to a signature/concept.
      (depScopeList, checkedDeps) <- unzip <$>
        mapM (buildModuleFromDependency tlDecls) deps
      depScope <- foldM mergeModules M.empty depScopeList
      return (depScope, checkedDeps)
  -- TODO: hacky step, make prelude.
  -- Step 2: register predicate functions if needed
  hackyScope <- foldM insertAndMergeDecl depScope hackyPrelude
  -- Step 3: register types
  -- TODO: remove name as parameter to "insertAndMergeDecl"
  baseScope <- foldMAccumErrors registerType hackyScope types
  -- Step 4: check and register protos
  protoScope <- do
    -- we first register all the protos without checking the guards
    protoScopeWithoutGuards <-
      foldMAccumErrors (registerProto False) baseScope callables
    -- now that we have access to all the functions in scope, we can check the
    -- guard
    foldMAccumErrors (registerProto True) protoScopeWithoutGuards callables
  -- Step 5: check bodies if allowed
  finalScope <- foldMAccumErrors checkBody protoScope callables
  -- Step 6: check that everything is resolved if program
  -- TODO: return typechecked elements here.
  -- TODO: accumulate here too
  when (moduleType == Program) $
    traverse_
      (traverse_ (recover (checkImplemented finalScope)) . getCallableDecls)
      finalScope
  -- TODO: check that name is unique?
  -- TODO: make module
  let resultModuleDecl = MModuleDecl $
        Ann { _ann = LocalDecl modSrc
            , _elem = MModule moduleType moduleName finalScope checkedDeps
            } -- :: TCModule
  return $ M.insertWith (<>) moduleName [resultModuleDecl] tlDecls
  where
    types :: [TypeDecl PhCheck]
    types = map (\t -> Ann [LocalDecl (_ann t)] (Type (nodeName t)))
      (getTypeDecls decls)

    checkBody
      :: ModuleScope
      -> CallableDecl PhParse
      -> MgMonad ModuleScope
    checkBody env decl@(Ann src (Callable ctype fname args retType _ cbody))
      -- TODO: move consistency check here to another function, for clarity.
      | moduleType /= Concept, Axiom <- ctype =
          throwLocatedE DeclContextErr src
            "axioms can only be declared in concepts"
      | EmptyBody <- cbody, Axiom <- ctype =
          throwLocatedE InvalidDeclErr src "axiom without a body"
      | moduleType /= External, ExternalBody <- cbody =
          throwLocatedE CompilerErr src $ pshow ctype <>
            " can not be declared external in " <> pshow moduleType
      | moduleType `notElem` [Implementation, Program], MagnoliaBody _ <- cbody,
        ctype `elem` [Function, Predicate, Procedure] =
          throwLocatedE InvalidDeclErr src $ pshow ctype <>
            " can not have a body in " <> pshow moduleType
      | EmptyBody <- cbody = checkProto True env decl >> return env
      -- In this case, the body of the function is not empty. It is either
      -- external (in which case, we only check the proto with the guard once
      -- again), or internal (in which case we need to type check it).
      | otherwise = do
          (tcBody, bodyRetType) <- case cbody of
            MagnoliaBody expr -> do
              (finalScope, tcExpr) <-
                annotateScopedExprStmt env (initScope args) (Just retType) expr
              mapM_ (checkArgIsUpdated finalScope) args
              return (MagnoliaBody tcExpr, _ann tcExpr)
            -- TODO: check whether final scope satisfies mode contract
            ExternalBody -> return (ExternalBody, retType)
            EmptyBody -> throwLocatedE CompilerErr src $
              "pattern matching fail in callable body check for " <> pshow fname
          Ann typAnn (Callable _ _ tcArgs _ tcGuard _) <-
            checkProto True env decl
          let tcAnnDecl = Ann typAnn $
                Callable ctype fname tcArgs retType tcGuard tcBody
          if bodyRetType == retType
          then insertAndMergeDecl env (CallableDecl tcAnnDecl)
          else throwLocatedE TypeErr src $
            "expected return type to be " <> pshow retType <> " in " <>
            pshow ctype <> " but return value has type " <>
            pshow bodyRetType

    checkArgIsUpdated :: VarScope -> TypedVar PhParse -> MgMonad ()
    checkArgIsUpdated scope (Ann src arg) = case _varMode arg of
      MOut -> case M.lookup (nodeName arg) scope of
        Nothing -> throwLocatedE CompilerErr src $ "argument " <>
          pshow (nodeName arg) <> " is out of scope after consistency checks "
        Just v -> unless (_varMode (_elem v) == MUpd) $
          throwLocatedE ModeMismatchErr src $ "argument " <>
          pshow (nodeName arg) <> " is 'out' but is not populated by the " <>
          "procedure body"
      _ -> return ()


    checkImplemented :: ModuleScope -> CallableDecl PhCheck -> MgMonad ()
    checkImplemented env callable@(Ann declO
      (Callable _ callableName _ _ _ cbody)) = case cbody of
        EmptyBody -> do
          let ~(Just matches) = getCallableDecls <$> M.lookup callableName env
              anonDefinedMatches =
                map (mkAnonProto <$$>) $ filter callableIsImplemented matches
              src = srcCtx $ head declO -- TODO: is head the best one to get?
          when ((mkAnonProto <$$> callable) `notElem` anonDefinedMatches) $
            throwLocatedE InvalidDeclErr src $ pshow callable <>
              " was left unimplemented in program"
        _ -> return ()

-- TODO: finish module casting.
castModule
  :: MModuleType
  -> MModule PhCheck
  -> MgMonad (MModule PhCheck)
castModule dstTyp origModule@(Ann declO (MModule srcTyp _ _ _)) = do
  let moduleWithSwappedType = return $ swapTyp <$$> origModule
      moduleAsSig = return $ mkSignature <$$> origModule
  case (srcTyp, dstTyp) of
    (Signature, Concept) -> moduleWithSwappedType
    (Signature, Implementation) -> moduleWithSwappedType
    (Signature, Program) -> noCast
    (Concept, Signature) -> moduleAsSig -- TODO: strip axioms
    (Concept, Implementation) -> undefined -- TODO: strip axioms?
    (Concept, Program) -> noCast
    (Implementation, Signature) -> moduleAsSig
    (Implementation, Concept) -> moduleAsSig
    (Implementation, Program) -> undefined -- TODO: check if valid program?
    (Program, Signature) -> moduleAsSig
    (Program, Concept) -> undefined -- TODO: moduleAsSig?
    (Program, Implementation) -> moduleWithSwappedType
    _ -> return origModule
  where
    noCast = throwLocatedE MiscErr (srcCtx declO) $
      pshow srcTyp <> " can not be casted to " <> pshow dstTyp
    swapTyp :: MModule' PhCheck -> MModule' PhCheck
    swapTyp (MModule _ name decls deps) = MModule dstTyp name decls deps
    swapTyp (RefModule _ _ v) = absurd v

    mkSignature :: MModule' PhCheck -> MModule' PhCheck
    mkSignature (MModule _ name decls deps) =
      -- TODO: should we do something with deps?
      MModule Signature name (M.map mkSigDecls decls) deps
    mkSignature (RefModule _ _ v) = absurd v

    -- TODO: make the lists non-empty in AST
    mkSigDecls :: [TCDecl] -> [TCDecl]
    mkSigDecls decls = foldl handleSigDecl [] decls

    handleSigDecl :: [TCDecl] -> TCDecl -> [TCDecl]
    handleSigDecl acc decl@(TypeDecl _) = decl : acc
    handleSigDecl acc (CallableDecl d) = case d of
      -- When casting to signature, axioms are stripped away.
      Ann _ MAxiom {} -> acc
      _ -> CallableDecl (prototypize <$$> d) : acc

    prototypize :: CallableDecl' p -> CallableDecl' p
    prototypize (Callable ctype name args retType guard _) =
      Callable ctype name args retType guard EmptyBody

castModule _ (Ann _ (RefModule _ _ v)) = absurd v
-- TODO: sanity check modes

annotateScopedExprStmt ::
  ModuleScope ->
  VarScope ->
  Maybe MType ->
  MExpr PhParse ->
  MgMonad (VarScope, MExpr PhCheck)
annotateScopedExprStmt modul inputScope inputMaybeExprType e =
  go inputScope inputMaybeExprType e
  where
  -- mExprTy is a parameter to disambiguate function calls overloaded
  -- solely on return types. It will *not* be used if the type can be inferred
  -- without it, and there is thus no guarantee that the resulting annotated
  -- expression will carry the specified type annotation. If this is important,
  -- it must be checked outside the call.
  go :: VarScope -> Maybe MType -> MExpr PhParse
     -> MgMonad (VarScope, MExpr PhCheck)
  go scope mExprTy (Ann src expr) = case expr of
    -- TODO: annotate type and mode on variables
    -- TODO: deal with mode
    MVar (Ann _ (Var _ name typ)) -> case M.lookup name scope of
        Nothing -> throwLocatedE UnboundVarErr src (pshow name)
        Just (Ann varType v) -> let typAnn = fromJust typ in
          if isNothing typ || typAnn == varType
          then let tcVar = MVar (Ann varType (Var (_varMode v) name (Just varType)))
               in return (scope, Ann varType tcVar)
          else throwLocatedE TypeErr src $ "got conflicting type " <>
            "annotation for var " <> pshow name <> ": " <> pshow name <>
            " has type " <> pshow varType <> " but type annotation is " <>
            pshow typAnn
    -- TODO: deal with casting
    MCall name args maybeCallCast -> do
        -- Type inference is fairly basic. We assume that for variables, either
        -- the type has been specified/previously inferred, or the variable is
        -- unset and also untyped. The only way to restrict an unknown variable
        -- type is to infer it during a function call. For that reason, we start
        -- out by checking all the sub-function calls in order to iteratively
        -- add constraints to our variables.
        -- In the current implementation, the constraints are resolved at every
        -- function call and not in the outermost call; this will cause
        -- inference to fail for function calls such as f(a:T, g(a)) when 'g'
        -- has multiple possible corresponding bindings, and 'a' is unset
        -- before the call. This will be improved upon later on.
        --
        -- We typecheck arguments expressions that are *not* variable
        -- references with an immutable scope, to ensure they do not affect
        -- the outer environment. This is because the evaluation order of
        -- arguments is not necessarily defined, and side effects may otherwise
        -- affect the result of the computation.
        -- Argument expressions that are variable references are however
        -- typechecked using the normal scope; this is because they then need
        -- to carry their corresponding mode, as opposed to other expressions
        -- which are anyway always "obs". "annotateScopedExpr" takes care of
        -- this distinction.
        tcArgs <- traverse (annotateScopedExpr scope Nothing) args
        let argTypes = map _ann tcArgs
            -- TODO: deal with modes here
            -- TODO: change with passing cast as parameter & whatever that implies
            candidates = filter (prototypeMatchesTypeConstraints argTypes Nothing) $
              getCallableDecls (M.findWithDefault [] name modul)
            tcExpr = MCall name tcArgs maybeCallCast
        -- TODO: stop using list to reach return val, maybe use Data.Sequence?
        -- TODO: do something with procedures and axioms here?
        case candidates of
          []  -> throwLocatedE UnboundFunctionErr src $ pshow name <>
            " with arguments of type (" <>
            T.intercalate ", " (map pshow argTypes) <> ")"
          -- In this case, we have 1+ matches. We can have more than one match
          -- for several reasons:
          -- (1) the arguments are fully specified but the function can be
          --     overloaded solely on its return type. In this case, we request
          --     explicit typing from the user
          -- (2) we registered several prototypes that refer to the same
          --     function in different ways. This can happen right now because
          --     of how the environment is implemented.
          --     TODO: use sets, and hash arguments ignoring their type to fix
          --     (2).
          matches -> do
            tcAnnExpr <-
              let possibleTypes = S.fromList $ map getReturnType matches in
              case maybeCallCast <|> mExprTy of
                Nothing ->
                  if S.size possibleTypes == 1
                  then return $ Ann (getReturnType (L.head matches)) tcExpr
                  else throwLocatedE TypeErr src $ "could not deduce return " <>
                    "type of call to " <> pshow name <> ". Possible " <>
                    "candidates have return types " <>
                    pshow (S.toList possibleTypes) <> ". Consider adding a " <>
                    "type annotation"
                Just cast ->
                  if S.size possibleTypes == 1
                  then do
                    let typAnn = getReturnType (L.head matches)
                    when (isJust maybeCallCast && typAnn /= cast) $
                      throwLocatedE TypeErr src $ "no matching candidate " <>
                        "for call to " <> pshow name <> " with type " <>
                        "annotation '" <> pshow cast <> "'"
                    return $ Ann typAnn tcExpr
                  else do
                    unless (cast `S.member` possibleTypes) $
                      throwLocatedE TypeErr src $ "could not deduce return " <>
                        "type of call to " <> pshow name <> ". Possible " <>
                        "candidates have return types " <>
                        pshow (S.toList possibleTypes) <> ". Consider " <>
                        "adding a type annotation"
                    return $ Ann cast tcExpr
            -- Here, using "head matches" is good enough, even though it can
            -- (and will) return the "wrong" candidate, in cases when several
            -- callables are overloaded solely on their return type. However:
            --
            -- (1) callables can only have a non-Unit return type if they are
            --     `function`s (i.e. all their arguments have mode `obs`)
            -- (2) there can be no ambiguity regarding whether we are calling a
            --     `function` or a `procedure`; we always know which type of
            --     callable we are invoking.
            --
            -- The compiler uses different namespaces for functions and
            -- procedures, ensuring therefore that all candidates are either
            -- functions (all their args are `obs`), or procedures (there is a
            -- single possible return type, therefore no overloading on return
            -- type is possible).
            --
            -- This means that checking the modes and names with a random
            -- candidate works, but checking other properties may fail.
            --
            -- TODO(bchetioui, 2021/06/10): note that this does not take into
            -- account the current bug allowing overloading procedures on
            -- modes. Once this is appropriately forbidden, however, this
            -- should always work. Remove this comment once fixed.
            checkArgModes src (L.head matches) tcArgs
            -- TODO: update modes here
            let vArgs = foldr
                  (\a l -> case _elem a of MVar v -> v:l ; _ -> l) [] tcArgs
                updateMode (Ann vty v) = case _varMode v of
                  MOut -> Ann vty (v { _varMode = MUpd }) ; _ -> Ann vty v
                endScope = foldr (M.adjust updateMode . nodeName) scope vArgs
            return (endScope, tcAnnExpr)
            -- TODO: ignore modes when overloading functions
    MBlockExpr blockType exprStmts -> case blockType of
      MEffectfulBlock -> do
        when (any isValueExpr exprStmts) $
          throwLocatedE CompilerErr src
            "effectful block contains value expressions"
        (scope', tcExprStmts) <- mapAccumM (`go` Nothing) scope exprStmts
        let endScope = M.fromList $ map (\k -> (k, scope' M.! k)) (M.keys scope)
        return (endScope, Ann Unit (MBlockExpr blockType tcExprStmts))
      MValueBlock -> do
        unless (any isValueExpr exprStmts) $
          throwLocatedE CompilerErr src
            "value block does not contain any value expression"
        let immutableScope = mkScopeVarsImmutable scope
        (_, tcExprStmts) <- mapAccumM (`go` Nothing) immutableScope exprStmts
        let retTypes = S.fromList $ map _ann (NE.filter isValueExpr tcExprStmts)
            ~(Just retType) = S.lookupMin retTypes
        unless (S.size retTypes == 1) $
          throwLocatedE TypeErr src $
            "value block has conflicting return types (found " <>
            T.intercalate ", " (map pshow (S.toList retTypes)) <> ")"
        return (scope, Ann retType (MBlockExpr blockType tcExprStmts))
    MValue expr' -> do
      tcExpr <- annotateScopedExpr scope mExprTy expr'
      checkExprMode src tcExpr MObs
      return (scope, Ann (_ann tcExpr) (MValue tcExpr))
    MLet (Ann _ (Var mode name mTy)) mExpr -> do
      unless (isNothing $ M.lookup name scope) $
        throwLocatedE MiscErr src $ "variable already defined in scope: " <>
          pshow name
      case mExpr of
        Nothing -> do
          when (isNothing mTy) $
            throwLocatedE NotImplementedErr src $ "can not yet declare a " <>
              "variable without specifying its type or an assignment " <>
              "expression"
          let (Just ty) = mTy
          checkTypeExists modul (WithSrc src ty)
          unless (mode == MOut) $
            throwLocatedE ModeMismatchErr src $ "variable " <> pshow name <>
              " is declared with mode " <> pshow mode <> " but is not " <>
              "initialized"
          -- Variable is unset, therefore has to be MOut.
          let newVar = Var MOut name ty
          return ( M.insert name (Ann ty newVar) scope
                 , Ann Unit (MLet (Ann ty (Var MOut name mTy)) Nothing)
                 )
        Just rhsExpr -> do
          tcRhsExpr <- annotateScopedExpr scope (mTy <|> mExprTy) rhsExpr
          checkExprMode src tcRhsExpr MObs
          let tcExpr =
                MLet (Ann (_ann tcRhsExpr) (Var mode name mTy)) (Just tcRhsExpr)
              rhsType = _ann tcRhsExpr
          case mTy of
            Nothing -> do
              let newVar = Var mode name rhsType
              return (M.insert name (Ann rhsType newVar) scope, Ann Unit tcExpr)
            Just ty -> do
              unless (ty == rhsType) $
                throwLocatedE TypeErr src $ "variable " <> pshow name <>
                  " has type annotation " <> pshow ty <> " but type " <>
                  "of assignment expression is " <> pshow rhsType
              when (mode == MOut) $
                throwLocatedE CompilerErr src $ "mode of variable " <>
                  "declaration can not be " <> pshow mode <> " if an " <>
                  "assignment expression is provided"
              let newVar = Var mode name ty
              return (M.insert name (Ann ty newVar) scope, Ann Unit tcExpr)
    MIf cond trueExpr falseExpr -> do
      tcCond <- annotateScopedExpr scope (Just Pred) cond
      unless (_ann tcCond == Pred) $
        throwLocatedE TypeErr src $ "expected condition to have type " <>
          pshow Pred <> " but got " <> pshow (_ann tcCond)
      checkExprMode src tcCond MObs
      (trueScope, tcTrueExpr) <- go scope mExprTy trueExpr
      checkExprMode src tcTrueExpr MObs
      (falseScope, tcFalseExpr) <- go scope (Just (_ann tcTrueExpr)) falseExpr
      checkExprMode src tcFalseExpr MObs
      unless (_ann tcTrueExpr == _ann tcFalseExpr) $
        throwLocatedE TypeErr src $ "expected branches of conditional to " <>
          "have the same type but got " <> pshow (_ann tcTrueExpr) <>
          " and " <> pshow (_ann tcFalseExpr)
      unless (  _ann tcTrueExpr == Unit
             && (scope /= falseScope || scope /= trueScope)) $
        throwLocatedE CompilerErr src $
          "if statement performed stateful computations in one of its " <>
          "and also returned a value"
      let tcExpr = MIf tcCond tcTrueExpr tcFalseExpr
      -- TODO: add sets of possible types to variables for type inference.
      -- TODO: can we set a variable to be "linearly assigned to" (only one
      --       time)?
      -- Modes are either upgraded in both branches, or we should throw an
      -- error. Each variable v in the parent scope has to satisfy one of the
      -- following properties:
      --   (1) v's mode is initially out, but updated to upd in the condition
      --   (2) v's mode is initially out, but updated to upd in both branches
      --   (3) v's mode is not updated throughout the computation (always true
      --       if the initial mode is not out).
      -- TODO: deal with short circuiting when emitting C++
      -- TODO: we can actually use the "least permissive variable mode"
      --       constraint to allow var to only be set in one branch; do we want
      --       to allow that?
      let scopeVars = M.toList $ M.mapWithKey (\k -> const $
            let f = fromJust . M.lookup k in (f trueScope, f falseScope)) scope
      resultScope <- M.fromList <$> for scopeVars (
          \(n, (Ann ty1 v1, Ann ty2 v2)) -> do
              -- (1) || (2) || (3) <=> _varMode v1 == _varMode v2
              unless (_varMode v1 == _varMode v2) $
                throwLocatedE CompilerErr src $ "modes should be updated in " <>
                  "the same way in if node"
              unless (ty1 == ty2) $
                throwLocatedE CompilerErr src $ "types should not change " <>
                  "in call to if function"
              return (n, Ann ty1 v1))
      return (resultScope, Ann (_ann tcTrueExpr) tcExpr)
    MAssert cond -> do
      tcCond <- annotateScopedExpr scope (Just Pred) cond
      when (_ann tcCond /= Pred) $
        throwLocatedE TypeErr src $ "expected expression to have type " <>
          pshow Pred <> " in predicate but got " <> pshow (_ann tcCond)
      checkExprMode src tcCond MObs
      return (scope, Ann Unit (MAssert tcCond))
    MSkip -> return (scope, Ann Unit MSkip)
    -- TODO: use for annotating AST

  getReturnType (Ann _ (Callable _ _ _ returnType _ _)) = returnType

  checkArgModes :: SrcCtx -> CallableDecl p -> [MExpr p] -> MgMonad ()
  checkArgModes src (Ann _ (Callable _ name callableArgs _ _ _)) callArgs = do
    let callableArgModes = map (_varMode . _elem) callableArgs
        callArgModes = map exprMode callArgs
        modeMismatches = filter (\(_, cand, call) -> not $ call `fitsMode` cand)
                                (zip3 [1::Int ..] callableArgModes callArgModes)
    unless (null modeMismatches) $ throwLocatedE ModeMismatchErr src $
      "incompatible modes in call to " <> pshow name <> ": " <>
      T.intercalate "; " (map (\(argNo, expected, mismatch) ->
        "expected " <> pshow expected <> " but got " <> pshow mismatch <>
        " for argument #" <> pshow argNo) modeMismatches)

  exprMode :: MExpr p -> MVarMode
  exprMode (Ann _ expr) = case expr of MVar v -> _varMode (_elem v) ; _ -> MObs

  checkExprMode :: SrcCtx -> MExpr p -> MVarMode -> MgMonad ()
  checkExprMode src expr mode =
    unless (exprMode expr `fitsMode` mode) $
        throwLocatedE ModeMismatchErr src $ "expected expression to have " <>
          "mode obs but it has mode " <> pshow (exprMode expr)

  fitsMode :: MVarMode -> MVarMode -> Bool
  fitsMode instanceMode targetedMode  = case instanceMode of
    MUnk -> False
    MUpd -> True
    _ -> instanceMode == targetedMode

  annotateScopedExpr :: VarScope -> Maybe MType -> MExpr PhParse
                     -> MgMonad (MExpr PhCheck)
  annotateScopedExpr sc mTy' e' =
    let inScope = case _elem e' of MVar _ -> sc ; _ -> mkScopeVarsImmutable sc
    in snd <$> annotateScopedExprStmt modul inScope mTy' e'


prototypeMatchesTypeConstraints
  :: [MType] -> Maybe MType -> TCCallableDecl -> Bool
prototypeMatchesTypeConstraints argTypeConstraints mreturnTypeConstraint
  (Ann _ (Callable _ _ declArgs declReturnType _ _))
  | length declArgs /= length argTypeConstraints = False
  | otherwise =
      let ~(Just returnTypeConstraint) = mreturnTypeConstraint
          fitsArgTypeConstraints =
            and $ zipWith (\x y -> x == _ann y) argTypeConstraints declArgs
          fitsReturnTypeConstraints = isNothing mreturnTypeConstraint ||
            returnTypeConstraint == declReturnType
      in fitsArgTypeConstraints && fitsReturnTypeConstraints

insertAndMergeDecl
  :: ModuleScope -> TCDecl -> MgMonad ModuleScope
insertAndMergeDecl env decl = do
  newDeclList <- mkNewDeclList
  return $ M.insert name newDeclList env
  where
    name = nodeName decl
    mkNewDeclList = case decl of
      TypeDecl tdecl ->
        (:[]) . TypeDecl <$> mergeTypes tdecl (head <$> M.lookup name env)
      CallableDecl cdecl ->
        map CallableDecl <$> mergePrototypes cdecl (M.lookup name env)

    -- TODO: is this motivation for storing types and callables in different
    -- scopes?
    mergeTypes :: TCTypeDecl -> Maybe TCDecl -> MgMonad TCTypeDecl
    mergeTypes annT1@(Ann declO1 t1) mannT2 = case mannT2 of
      Nothing                        -> return annT1
      Just (TypeDecl (Ann declO2 _)) -> return $ Ann (mergeAnns declO1 declO2) t1
      Just (CallableDecl cdecl) ->
        throwLocatedE CompilerErr (srcCtx (head declO1)) $ "a callable " <>
        "was registered with name " <> pshow (nodeName cdecl) <> "but " <>
        pshow (nodeName cdecl) <> " belongs to the type namespace"

    mergePrototypes
      :: TCCallableDecl -> Maybe [TCDecl] -> MgMonad [TCCallableDecl]
    mergePrototypes anncallableDecl@(Ann declO callableDecl) mdecls
      | Nothing <- mdecls = return [anncallableDecl]
      | Just decls <- mdecls = do
        let protos = getCallableDecls decls
            (Callable ctype _ args returnType mguard cbody) = callableDecl
            argTypes = map (_varType . _elem) args
            (matches, nonMatches) = L.partition
              (prototypeMatchesTypeConstraints argTypes (Just returnType)) protos
            (matchesWithBody, matchesWithoutBody) =
              L.partition callableIsImplemented matches
        -- If the callable is already in the environment, we do not insert it
        -- a second time.
        if anncallableDecl `elem` matches then return $ matches <> nonMatches
        else if not (null matches) then do
          when (cbody /= EmptyBody && not (null matchesWithBody)) $
            throwLocatedE InvalidDeclErr (srcCtx . head $ declO) $ -- TODO: how to add src info better here?
              "attempting to import two different implementations for " <>
              pshow ctype <> " " <> pshow name
          when (length matchesWithBody > 1 || length matchesWithoutBody > 1) $
            throwLocatedE CompilerErr (srcCtx . head $ declO) $
              "context contains several definitions of the same prototype " <>
              "in AST for " <> pshow ctype <> " " <> pshow name
          when (length matchesWithBody == 1 && length matchesWithoutBody == 1 &&
                  extractGuard (head matchesWithBody) /=
                  extractGuard (head matchesWithoutBody)) $
            throwLocatedE CompilerErr (srcCtx . head $ declO) $
              "existing prototype and implementation have inconsistent " <>
              "guards for " <> pshow ctype <> " " <> pshow name

          -- From here on out, we know that we have at least an existing match,
          -- and that the guard is the same for all the matches. Therefore, we
          -- generate a unique new guard based on the new prototype, and insert
          -- it into all the matches we have.
          newGuard <- mergeGuards mguard (extractGuard (head matches))
          let newMatches = map (flip replaceGuard newGuard <$$>) matches
          -- We know that we have exactly one of two possible configurations
          -- here:
          -- (1) anncallableDecl does not have a body, and we already have
          --     an equivalent prototype in scope
          -- (2) anncallableDecl has a body, and we already have the relevant
          --     prototype in scope
          -- TODO: clean out irrelevant protos where possible, and
          --       remove reliance on the existence of a prototype without body
          --       in scope (not sure if absolutely needed here, but either way)
          if callableIsImplemented anncallableDecl
          then return $ anncallableDecl : newMatches <> nonMatches
          else return $ newMatches <> nonMatches
        else return $ anncallableDecl : nonMatches

    mergeGuards :: CGuard p -> CGuard p -> MgMonad (CGuard p)
    mergeGuards mguard1 mguard2 = case (mguard1, mguard2) of
      (Nothing, _)  -> return mguard2
      (_ , Nothing) -> return mguard1
      (Just guard1, Just guard2) ->
        -- We consider two guards as equivalent only if they are
        -- syntactically equal.
        if guard1 == guard2 then return $ Just guard1
        else throwLocatedE NotImplementedErr Nothing -- TODO: make guard1 && guard2 + add right srcctx
          "merging of two callables with different guards"

    mergeAnns :: t ~ [DeclOrigin] => t -> t -> t
    mergeAnns declOs1 declOs2 = S.toList $ S.fromList (declOs1 <> declOs2)

    extractGuard ~(Ann _ (Callable _ _ _ _ mguard _)) = mguard

mergeModules :: ModuleScope -> ModuleScope -> MgMonad ModuleScope
mergeModules mod1 mod2 =
  foldM insertAndMergeDeclList mod1 (map snd $ M.toList mod2)
  where
    insertAndMergeDeclList initEnv declList =
      foldM insertAndMergeDecl initEnv declList

checkTypeExists ::
  ModuleScope ->
  WithSrc Name ->
  MgMonad ()
checkTypeExists modul (WithSrc src name)
  | Unit <- name = return ()
  | Pred <- name = return ()
  | otherwise = case M.lookup name modul of
      Nothing      -> throwLocatedE UnboundTypeErr src (pshow name)
      Just matches -> if not $ null (getTypeDecls matches)
                      then return ()
                      else throwLocatedE UnboundTypeErr src (pshow name)

registerType
  :: ModuleScope
  -> TypeDecl PhCheck
  -> MgMonad ModuleScope
registerType modul annType =
  foldM insertAndMergeDecl modul (mkTypeUtils annType)

mkTypeUtils
  :: TypeDecl PhCheck
  -> [MDecl PhCheck]
mkTypeUtils annType = [ TypeDecl annType
                      , CallableDecl (Ann (_ann annType) equalityFnDecl)
                      , CallableDecl (Ann (_ann annType) assignProcDecl)
                      ]
  where
    mkVar mode nameStr =
      Ann (nodeName annType) $ Var mode (VarName nameStr) (nodeName annType)
    equalityFnDecl =
      Callable Function (FuncName "_==_") (map (mkVar MObs) ["e1", "e2"]) Pred
               Nothing ExternalBody
    assignProcDecl =
      Callable Procedure (ProcName "_=_")
              [mkVar MOut "var", mkVar MObs "expr"] Unit Nothing ExternalBody

-- TODO: ensure functions have a return type defined, though should be handled
-- by parsing.

registerProto ::
  Bool -> -- whether to register/check guards or not
  ModuleScope ->
  CallableDecl PhParse ->
  MgMonad ModuleScope
registerProto checkGuards modul annCallable =
  do -- TODO: ensure bodies are registered later on
    checkedProto <- checkProto checkGuards modul annCallable
    insertAndMergeDecl modul (CallableDecl checkedProto)

-- TODO: check for procedures, predicates, axioms (this is only for func)?
-- TODO: check guard
checkProto
  :: Bool
  -> ModuleScope
  -> CallableDecl PhParse
  -> MgMonad (CallableDecl PhCheck)
checkProto checkGuards env
    (Ann src (Callable ctype name args retType mguard _)) = do
  tcArgs <- checkArgs args
  checkTypeExists env (WithSrc src retType)
  tcGuard <- case mguard of
    Nothing -> return Nothing
    Just guard -> if checkGuards
      then Just . snd <$>
        annotateScopedExprStmt env (initScope args) (Just Pred) guard
      else return Nothing
  return Ann { _ann = [LocalDecl src]
             , _elem = Callable ctype name tcArgs retType tcGuard EmptyBody
             }
  where checkArgs :: [TypedVar PhParse] -> MgMonad [TypedVar PhCheck]
        checkArgs vars = do
          -- TODO: make sure there is no need to check
          --when (ctype /= Function) $ error "TODO: proc/axiom/pred"
          let varSet = S.fromList [_varName v | (Ann _ v) <- vars]
          if S.size varSet /= L.length vars
          then throwLocatedE MiscErr src $
            "duplicate argument names in declaration of " <> pshow name
          else mapM checkArgType vars

        checkArgType :: TypedVar PhParse -> MgMonad (TypedVar PhCheck)
        checkArgType (Ann argSrc (Var mode varName typ)) = do
          checkTypeExists env (WithSrc argSrc typ)
          return $ Ann typ (Var mode varName typ)

buildModuleFromDependency
  :: Env [TCTopLevelDecl]
  -> MModuleDep PhParse
  -> MgMonad (Env [TCDecl], TCModuleDep)
buildModuleFromDependency env (Ann src (MModuleDep ref renamings castToSig)) =
  do  match <- lookupTopLevelRef src (M.map getModules env) ref
      -- TODO: strip non-signature elements here.
      decls <- if castToSig
        then do
          decls' <- moduleDecls <$> castModule Signature match
          -- When casting to signature, external definitions are discarded.
          -- Because some external functions are registered initially for each
          -- type (e.g. _==_), we make sure to register them again.
          let typeDecls = foldl (\acc d -> acc <> getTypeDecls d) [] decls'
          foldM registerType decls' typeDecls
        else return $ moduleDecls match
      -- TODO: gather here env of known renamings
      checkedRenamings <-
        mapM (expandRenamingBlock (M.map getNamedRenamings env)) renamings
      renamedDecls <- foldM applyRenamingBlock decls checkedRenamings
      -- TODO: do this well
      let checkedDep = Ann src (MModuleDep ref checkedRenamings castToSig)
      -- TODO: hold checkedModuleDeps in M.Map FQN Deps (change key type)
      return (renamedDecls, checkedDep)

-- TODO: cleanup duplicates, make a set of MDecl instead of a list?
-- TODO: finish renamings
-- TODO: add annotations to renamings, work with something better than just
--       name.
-- TODO: improve error diagnostics
applyRenamingBlock
  :: ModuleScope
  -> MRenamingBlock PhCheck
  -> MgMonad ModuleScope
applyRenamingBlock modul renamingBlock@(Ann src (MRenamingBlock renamings)) = do
  -- TODO: pass renaming decls to expand them
  let inlineRenamings = mkInlineRenamings renamingBlock
      renamingMap = M.fromList inlineRenamings
      (sources, targets) = ( L.map fst inlineRenamings
                           , L.map snd inlineRenamings
                           )
      -- TODO: will have to modify when renamings can be more atomic and
      -- namespace can be specified.
      filterSources sourceNames namespace =
        filter (\(Name _ nameStr) ->
                isNothing $ M.lookup (Name namespace nameStr) modul) sourceNames
      unknownSources =
        foldl filterSources sources [NSFunction, NSProcedure, NSType]
      -- All renamings in a renaming block happen at the same time. If we have
      -- names that are both the source and target of different renamings, we
      -- need to ensure the renaming for which the name is the source happens
      -- before the one for which it is the target. A renaming block r like
      -- [T1 -> T2, T2 -> T1, T3 -> T4] has no valid ordering to ensure that,
      -- however; to avoid running into trouble in such a case, we instead
      -- transform r into two subsequent renaming blocks r' and r'', such that
      -- r'  = [T1 -> freeName1, T2 -> freeName2], and
      -- r'' = [freeName1 -> T2, freeName2 -> T1, T3 -> T4]. From a theoretical
      -- point of view, r'' . r' = r, and it makes handling unambiguous in
      -- practice.
      -- TODO: another view is that max 1 renaming can be applied for each
      -- name. Could optimize later if need be.
      occurOnBothSides = L.filter (`elem` targets) sources
      unambiguousRenamings = filter (\(source, _) ->
                                     source `notElem` occurOnBothSides)
                                    inlineRenamings
      -- TODO: cleanup
      r' = zip occurOnBothSides
                    [GenName ("gen#" ++ show i) | i <- [1..] :: [Int]]
      r'' = unambiguousRenamings <>
              [(freeName, fromJust $ M.lookup source renamingMap) |
               (source, freeName) <- r']
  (modul', renamings') <- (
      if M.size renamingMap /= L.length renamings
      then throwLocatedE MiscErr src "duplicate source in renaming block"
      else if not (null unknownSources)
      then throwLocatedE MiscErr src $ "renaming block has unknown " <>
        "sources: " <> pshow unknownSources
      else if not (null occurOnBothSides)
      then let annR' = map (Ann (LocalDecl Nothing) . InlineRenaming) r' in
           applyRenamingBlock modul (MRenamingBlock annR' <$$ renamingBlock)
           >>= \modul' -> return (modul', r'')
      else return (modul, inlineRenamings))
  return $ M.fromListWith (<>) $ L.map (\(k, decls) ->
          (tryAllRenamings replaceName k renamings',
           L.map (flip (tryAllRenamings applyRenaming) renamings') decls)) $
           M.toList modul'
  where tryAllRenamings renamingFun target = foldl renamingFun target

-- TODO: specialize replacements based on namespaces?
replaceName :: Name -> InlineRenaming -> Name
replaceName origName@(Name ns nameStr) (Name _ sourceStr, Name _ targetStr) =
  if sourceStr == nameStr then Name ns targetStr else origName

-- Applies a renaming to a declaration. Renamings only affect names defined at
-- declaration level; this means that they do not affect local variables.
applyRenaming :: MDecl PhCheck -> InlineRenaming -> MDecl PhCheck
applyRenaming decl renaming = case decl of
  TypeDecl typeDecl -> TypeDecl $ applyRenamingInTypeDecl <$$> typeDecl
  CallableDecl callableDecl ->
    CallableDecl $ applyRenamingInCallableDecl <$$> callableDecl
  where
    replaceName' = flip replaceName renaming
    applyRenamingInTypeDecl (Type typ) = Type $ replaceName' typ
    applyRenamingInCallableDecl (Callable ctyp name vars retType mguard cbody) =
      Callable ctyp (replaceName' name)
        (map applyRenamingInTypedVar vars) (replaceName' retType)
        (applyRenamingInExpr <$> mguard)
        (case cbody of
          MagnoliaBody body -> MagnoliaBody (applyRenamingInExpr body)
          _ -> cbody)
    applyRenamingInTypedVar :: TypedVar PhCheck -> TypedVar PhCheck
    applyRenamingInTypedVar (Ann typAnn (Var mode name typ)) =
      Ann (replaceName' typAnn) $ Var mode name (replaceName' typ)
    applyRenamingInMaybeTypedVar
      :: MaybeTypedVar PhCheck -> MaybeTypedVar PhCheck
    applyRenamingInMaybeTypedVar (Ann typAnn (Var mode name typ)) =
      Ann (replaceName' typAnn) $ Var mode name (replaceName' <$> typ)
    applyRenamingInExpr :: MExpr PhCheck -> MExpr PhCheck
    applyRenamingInExpr (Ann typAnn expr) =
      Ann (replaceName' typAnn) (applyRenamingInExpr' expr)
    applyRenamingInExpr' expr = case expr of
      MVar var -> MVar $ applyRenamingInMaybeTypedVar var
      MCall name vars typ -> MCall (replaceName' name)
        (map applyRenamingInExpr vars) (replaceName' <$> typ)
      MBlockExpr blockTy stmts ->
        MBlockExpr blockTy $ NE.map applyRenamingInExpr stmts
      MValue expr' -> MValue $ applyRenamingInExpr expr'
      MLet v rhsExpr -> MLet (applyRenamingInMaybeTypedVar v)
        (applyRenamingInExpr <$> rhsExpr)
      MIf cond trueExpr falseExpr -> MIf (applyRenamingInExpr cond)
        (applyRenamingInExpr trueExpr) (applyRenamingInExpr falseExpr)
      MAssert cond -> MAssert (applyRenamingInExpr cond)
      MSkip -> MSkip


-- === utils ===

mapAccumM :: (Traversable t, Monad m)
          => (a -> b -> m (a, c)) -> a -> t b -> m (a, t c)
mapAccumM f a tb = swap <$> mapM go tb `runStateT` a
  where go b = do s <- get
                  (s', r) <- lift $ f s b
                  put s'
                  return r