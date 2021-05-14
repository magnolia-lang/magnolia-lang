{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Check (checkModule) where

import Control.Applicative
import Control.Monad.Except
    ( when, foldM, unless, ExceptT, Except, MonadTrans(lift) )
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

type VarScope = Env (TypedVar PhCheck)
type ModuleScope = Env [MDecl PhCheck] --ModuleEnv PhCheck

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

-- TODO: can variables exist without a type in the scope?
-- TODO: replace things with relevant sets
-- TODO: handle circular dependencies
checkModule
  :: Env [TCTopLevelDecl]
  -> MModule PhParse
  -> Except Err (Env [TCTopLevelDecl])
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
  baseScope <- foldM registerType hackyScope types
  -- Step 4: check and register protos
  protoScope <- do
    -- we first register all the protos without checking the guards
    protoScopeWithoutGuards <- foldM (registerProto False) baseScope callables
    -- now that we have access to all the functions in scope, we can check the
    -- guard
    foldM (registerProto True) protoScopeWithoutGuards callables
  -- Step 5: check bodies if allowed
  finalScope <- foldM checkBody protoScope callables
  -- Step 6: check that everything is resolved if program
  -- TODO: return typechecked elements here.
  when (moduleType == Program) $
      traverse_ (traverse_ (checkImplemented finalScope) . getCallableDecls)
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
      -> Except Err ModuleScope
    checkBody env decl@(Ann src (Callable ctype fname args retType _ cbody))
      | moduleType /= Concept, Axiom <- ctype =
          throwLocatedE DeclContextErr src
            "axioms can only be declared in concepts"
      | EmptyBody <- cbody, Axiom <- ctype =
          throwLocatedE InvalidDeclErr src "axiom without a body"
      | moduleType /= External, ExternalBody <- cbody =
          throwLocatedE CompilerErr src $ pshow ctype <>
            " can not be declared external in " <> pshow moduleType
      | moduleType `notElem` [Implementation, Program], MagnoliaBody _ <- cbody,
        ctype `elem` [Function, Procedure] =
          throwLocatedE InvalidDeclErr src $ pshow ctype <>
            " can not have a body in " <> pshow moduleType
      | EmptyBody <- cbody = checkProto True env decl >> return env
      -- In this case, the body of the function is not empty. It is either
      -- external (in which case, we only check the proto with the guard once
      -- again), or internal (in which case we need to type check it).
      | otherwise = do
          (bodyTC, bodyRetType) <- case cbody of
            MagnoliaBody expr -> do
              tcExpr <-
                annotateScopedExpr env (initScope args) (Just retType) expr
              return (MagnoliaBody tcExpr, _ann tcExpr)
            ExternalBody -> return (ExternalBody, retType)
            EmptyBody -> throwLocatedE CompilerErr src $
              "pattern matching fail in callable body check for " <> pshow fname
          Ann typAnn (Callable _ _ argsTC _ guardTC _) <-
            checkProto True env decl
          let annDeclTC = Ann typAnn $
                Callable ctype fname argsTC retType guardTC bodyTC
          if bodyRetType == retType
          then insertAndMergeDecl env (CallableDecl annDeclTC)
          else throwLocatedE TypeErr src $
            "expected return type to be " <> pshow retType <> " in " <>
            pshow ctype <> "but return value has type " <>
            pshow bodyRetType


    checkImplemented :: ModuleScope -> CallableDecl PhCheck -> Except Err ()
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
  :: Monad t
  => MModuleType
  -> MModule PhCheck
  -> ExceptT Err t (MModule PhCheck)
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

annotateScopedExpr ::
  ModuleScope ->
  VarScope ->
  Maybe MType ->
  MExpr PhParse ->
  Except Err (MExpr PhCheck)
annotateScopedExpr inputModule inputScope inputMaybeExprType e = do
  snd <$> go inputModule inputScope inputMaybeExprType e
  where
  -- maybeExprType is a parameter to disambiguate function calls overloaded
  -- solely on return types. It will *not* be used if the type can be inferred
  -- without it, and there is thus no guarantee that the resulting annotated
  -- expression will carry the specified type annotation. If this is important,
  -- it must be checked outside the call.
  go :: ModuleScope -> VarScope -> Maybe MType -> MExpr PhParse
     -> Except Err (VarScope, MExpr PhCheck)
  go modul scope maybeExprType (Ann src expr) = case expr of
    -- TODO: annotate type and mode on variables
    -- TODO: deal with mode
    MVar (Ann _ (Var mode name typ)) -> case M.lookup name scope of
        Nothing -> throwLocatedE UnboundVarErr src (pshow name)
        Just (Ann varType _) -> let typAnn = fromJust typ in
          if isNothing typ || typAnn == varType
          then let tcVar = MVar (Ann varType (Var mode name (Just varType)))
               in return (scope, Ann { _ann = varType, _elem = tcVar })
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
        -- First, check that all the arguments contain only computations that
        -- are without effects on the environment. This is because evaluation
        -- order might otherwise affect the result of the computation.
        when (any isStateful args) $
          let argNo = fst . head $ filter (isStateful . snd) $ zip [1..] args
                :: Int in
          throwLocatedE MiscErr src $ "expected stateless computations " <>
            "as call arguments but argument #" <> pshow argNo <> " was " <>
            "stateful in call to " <> pshow name
        argsTC <- traverse (annotateScopedExpr modul scope Nothing) args
        let argTypes = map _ann argsTC
            -- TODO: deal with modes here
            -- TODO: change with passing cast as parameter & whatever that implies
            candidates = filter (prototypeMatchesTypeConstraints argTypes Nothing) $
              getCallableDecls (M.findWithDefault [] name modul)
            exprTC = MCall name argsTC maybeCallCast
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
          matches ->
            let possibleTypes = S.fromList $ map getReturnType matches in
            case maybeCallCast <|> maybeExprType of
              Nothing ->
                if S.size possibleTypes == 1
                then let typAnn = getReturnType (L.head matches) in
                     return (scope, Ann { _ann = typAnn, _elem = exprTC })
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
                  return ( scope
                        , Ann { _ann = typAnn, _elem = exprTC }
                        )
                else do
                  unless (cast `S.member` possibleTypes) $
                    throwLocatedE TypeErr src $ "could not deduce return " <>
                      "type of call to " <> pshow name <> ". Possible " <>
                      "candidates have return types " <>
                      pshow (S.toList possibleTypes) <> ". Consider " <>
                      "adding a type annotation"
                  return ( scope
                        , Ann { _ann = cast, _elem = exprTC }
                        )
              -- TODO: ignore modes when overloading functions
    MBlockExpr exprStmts -> do
      (intermediateScope, initExprStmtsTC) <-
          mapAccumM (flip (go modul) Nothing) scope (NE.init exprStmts)
      -- The last exprStmt must be treated differently because it's potentially
      -- annotated.
      (scope', lastExprStmtTC) <-
          go modul intermediateScope maybeExprType (NE.last exprStmts)
      -- Return the old scope with potential new annotations; this makes sure
      -- that variables declared in the current scope are eliminated.
      -- Note that we do not yet have nested scopes; this means that it is
      -- never possible to shadow names from an outer scope.
      let endScope = M.fromList $ map (\k -> (k, scope' M.! k)) (M.keys scope)
          newBlock = MBlockExpr $ NE.fromList (initExprStmtsTC <>
                                               [lastExprStmtTC])
      return (endScope, Ann { _ann = _ann lastExprStmtTC, _elem = newBlock })
    MLet mode name maybeType maybeExpr -> do
      unless (isNothing $ M.lookup name scope) $
        throwLocatedE MiscErr src $ "variable already defined in scope: " <>
          pshow name
      case maybeExpr of
        Nothing -> do
          when (isNothing maybeType) $
            throwLocatedE NotImplementedErr src $ "can not yet declare a " <>
              "variable without specifying its type or an assignment " <>
              "expression"
          let (Just typ) = maybeType
          checkTypeExists modul (WithSrc src typ)
          unless (mode == MOut) $
            throwLocatedE ModeMismatchErr src $ "variable " <> pshow name <>
              " is declared with mode " <> pshow mode <> " but is not " <>
              "initialized"
          -- Variable is unset, therefore has to be MOut.
          let newVar = Var MOut name typ
          return ( M.insert name Ann { _ann = typ, _elem = newVar } scope
                 , Ann Unit (MLet mode name maybeType Nothing)
                 )
        Just rhsExpr -> do
          (scope', rhsExprTC) <- go modul scope
              (maybeType <|> maybeExprType) rhsExpr
          let exprTC = MLet mode name maybeType (Just rhsExprTC)
              rhsType = _ann rhsExprTC
          case maybeType of
            Nothing -> do
              let newVar = Var mode name rhsType
              return ( M.insert name (Ann rhsType newVar) scope'
                     , Ann { _ann = Unit, _elem = exprTC }
                     )
            Just typ -> do
              unless (typ == rhsType) $
                throwLocatedE TypeErr src $ "variable " <> pshow name <>
                  " has type annotation " <> pshow typ <> " but type " <>
                  "of assignment expression is " <> pshow rhsType
              when (mode == MOut) $
                throwLocatedE CompilerErr src $ "mode of variable " <>
                  "declaration can not be " <> pshow mode <> " if an " <>
                  "assignment expression is provided"
              let newVar = Var mode name typ
              return ( M.insert name Ann { _ann = typ, _elem = newVar } scope'
                     , Ann { _ann = Unit, _elem = exprTC }
                     )
    MIf cond bTrue bFalse -> do
      let statefulBranches = map snd $ filter (isStateful . fst)
            [(cond, "condition"), (bTrue, "true branch"),
             (bFalse, "false branch")] :: [T.Text]
      unless (null statefulBranches) $
        throwLocatedE MiscErr src $ "expected if-then-else nodes to be " <>
          "stateless computations but " <> pshow (head statefulBranches) <>
          "is stateful"
      (condScope, condExprTC) <- go modul scope (Just Pred) cond
      when (_ann condExprTC /= Pred) $
        throwLocatedE TypeErr src $ "expected condition to have type " <>
          pshow Pred <> " but got " <> pshow (_ann condExprTC)
      (trueScope, trueExprTC) <- go modul condScope maybeExprType bTrue
      (falseScope, falseExprTC) <-
        go modul condScope (Just (_ann trueExprTC)) bFalse
      when (_ann trueExprTC /= _ann falseExprTC) $
        throwLocatedE TypeErr src $ "expected branches of conditional to " <>
          "have the same type but got " <> pshow (_ann trueExprTC) <>
          " and " <> pshow (_ann falseExprTC)
      let exprTC = MIf condExprTC trueExprTC falseExprTC
      -- TODO: join scopes. Description below:
      -- TODO: add sets of possible types to variables for type inference.
      -- TODO: can we set a variable to be "linearly assigned to" (only one
      --       time)?
      -- Modes are either upgraded in both branches, or we should throw an
      -- error. Each variable v in the parent scope has to satisfy one of the
      -- following properties: (TODO: (1), (2) and (4) are invalid now).
      --   (1) v's type is unknown in the condition and the 2 branches
      --   (2) v's type is unknown in the condition, and set to the same value
      --       in the 2 branches (TODO: an intersection exists?)
      --   (3) v's type is known in the initial scope
      --   (4) v's type is set in the condition.
      -- Additionally, they also must satisfy one of the following properties:
      --   (5) v's mode is initially out, but updated to upd in the condition
      --   (6) v's mode is initially out, but updated to upd in both branches
      --   (7) v's mode is not updated throughout the computation (always true
      --       if the initial mode is not out).
      -- Note: all variables from the initial scope should exist in all scopes.
      -- TODO: deal with short circuiting when emitting C++
      -- TODO: we can actually use the "least permissive variable mode"
      --       constraint to allow var to only be set in one branch; do we want
      --       to allow that?
      -- TODO: this will become irrelevant logic once if is implemented as a
      --       function call. Modes should actually *not* change, as arguments
      --       should be stateless computations.
      let scopeVars = M.toList $ M.map (\(Ann _ (Var _ name _)) ->
               let findVar = fromJust . M.lookup name in
               (findVar trueScope, findVar falseScope)) scope
      resultScope <- M.fromList <$> for scopeVars (
          \(n, (Ann typ1 v1, Ann typ2 v2)) -> do
              -- (5) || (6) || (7) <=> _varMode v1 == _varMode v2
              when (_varMode v1 /= _varMode v2) $
                throwLocatedE CompilerErr src $ "modes should not change " <>
                  "in call to if function"
              when (typ1 /= typ2) $
                throwLocatedE CompilerErr src $ "types should not change " <>
                  "in call to if function" -- TODO: someday they might
              return (n, Ann { _ann = typ1, _elem = v1 }))
      return (resultScope, Ann { _ann = _ann trueExprTC, _elem = exprTC })
    MAssert cond -> do
      (scope', newCond) <- go modul scope (Just Pred) cond
      when (_ann newCond /= Pred) $
        throwLocatedE TypeErr src $ "expected expression to have type " <>
          pshow Pred <> " in predicate but got " <> pshow (_ann newCond)
      return (scope', Ann { _ann = Unit, _elem = MAssert newCond })
    MSkip -> return (scope, Ann { _ann = Unit, _elem = MSkip })
    -- TODO: use for annotating AST

  getReturnType (Ann _ (Callable _ _ _ returnType _ _)) = returnType

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
  :: ModuleScope -> TCDecl -> Except Err ModuleScope
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
    mergeTypes :: TCTypeDecl -> Maybe TCDecl -> Except Err TCTypeDecl
    mergeTypes annT1@(Ann declO1 t1) mannT2 = case mannT2 of
      Nothing                        -> return annT1
      Just (TypeDecl (Ann declO2 _)) -> return $ Ann (mergeAnns declO1 declO2) t1
      Just (CallableDecl cdecl) ->
        throwLocatedE CompilerErr (srcCtx (head declO1)) $ "a callable " <>
        "was registered with name " <> pshow (nodeName cdecl) <> "but " <>
        pshow (nodeName cdecl) <> " belongs to the type namespace"

    mergePrototypes
      :: TCCallableDecl -> Maybe [TCDecl] -> Except Err [TCCallableDecl]
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

    mergeGuards :: CGuard p -> CGuard p -> Except Err (CGuard p)
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

mergeModules :: ModuleScope -> ModuleScope -> Except Err ModuleScope
mergeModules mod1 mod2 =
  foldM insertAndMergeDeclList mod1 (map snd $ M.toList mod2)
  where
    insertAndMergeDeclList initEnv declList =
      foldM insertAndMergeDecl initEnv declList

checkTypeExists ::
  ModuleScope ->
  WithSrc Name ->
  Except Err ()
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
  -> Except Err ModuleScope
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
  Except Err ModuleScope
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
  -> Except Err (CallableDecl PhCheck)
checkProto checkGuards env
    (Ann src (Callable ctype name args retType mguard _)) = do
  tcArgs <- checkArgs args
  checkTypeExists env (WithSrc src retType)
  tcGuard <- case mguard of
    Nothing -> return Nothing
    Just guard -> if checkGuards
      then Just <$> annotateScopedExpr env (initScope args) (Just Pred) guard
      else return Nothing
  return Ann { _ann = [LocalDecl src]
             , _elem = Callable ctype name tcArgs retType tcGuard EmptyBody
             }
  where checkArgs :: [TypedVar PhParse] -> Except Err [TypedVar PhCheck]
        checkArgs vars = do
          -- TODO: make sure there is no need to check
          --when (ctype /= Function) $ error "TODO: proc/axiom/pred"
          let varSet = S.fromList [_varName v | (Ann _ v) <- vars]
          if S.size varSet /= L.length vars
          then throwLocatedE MiscErr src $
            "duplicate argument names in declaration of " <> pshow name
          else mapM checkArgType vars

        checkArgType :: TypedVar PhParse -> Except Err (TypedVar PhCheck)
        checkArgType (Ann argSrc (Var mode varName typ)) = do
          checkTypeExists env (WithSrc argSrc typ)
          return $ Ann typ (Var mode varName typ)

buildModuleFromDependency
  :: Env [TCTopLevelDecl]
  -> MModuleDep PhParse
  -> Except Err (Env [TCDecl], TCModuleDep)
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
  -> Except Err ModuleScope
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
    applyRenamingInMaybeTypedVar :: MaybeTypedVar PhCheck -> MaybeTypedVar PhCheck
    applyRenamingInMaybeTypedVar (Ann typAnn (Var mode name typ)) =
      Ann (replaceName' typAnn) $ Var mode name (replaceName' <$> typ)
    applyRenamingInExpr :: MExpr PhCheck -> MExpr PhCheck
    applyRenamingInExpr (Ann typAnn expr) =
      Ann (replaceName' typAnn) (applyRenamingInExpr' expr)
    applyRenamingInExpr' expr = case expr of
      MVar var -> MVar $ applyRenamingInMaybeTypedVar var
      MCall name vars typ -> MCall (replaceName' name)
        (map applyRenamingInExpr vars) (replaceName' <$> typ)
      MBlockExpr stmts -> MBlockExpr $ NE.map applyRenamingInExpr stmts
      MLet mode name typ rhsExpr -> MLet mode name (replaceName' <$> typ)
        (applyRenamingInExpr <$> rhsExpr)
      MIf cond bTrue bFalse -> MIf (applyRenamingInExpr cond)
        (applyRenamingInExpr bTrue) (applyRenamingInExpr bFalse)
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

isStateful :: MExpr p -> Bool
isStateful (Ann _ expr) = case expr of
  -- Operations that can affect the scope are:
  -- - variable declarations/variable assignments
  -- - calls to procedures
  MVar _ -> False
  MCall (FuncName _) args _ -> any isStateful args
  MCall (ProcName _) _ _ -> True
  MCall {} -> error $ "call to something that is neither a type " <>
                      "of function nor a procedure."
  MBlockExpr stmts -> or $ NE.map isStateful stmts
  MLet {} -> True
  MIf cond bTrue bFalse -> any isStateful [cond, bTrue, bFalse]
  MAssert cond -> isStateful cond
  MSkip -> False
