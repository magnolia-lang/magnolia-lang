{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Check (checkModule) where

import Control.Applicative
import Control.Monad.Except
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

type VarScope = Env (UVar PhCheck)
type ModuleScope = Env [UDecl PhCheck] --ModuleEnv PhCheck

initScope :: [UVar p] -> VarScope
initScope = M.fromList . map mkScopeVar
  where
    mkScopeVar :: UVar p -> (Name, UVar PhCheck)
    mkScopeVar ~(Ann _ (Var mode name (Just typ))) =
      (name, Ann { _ann = typ, _elem = Var mode name (Just typ) })

-- TODO: can variables exist without a type in the scope?
-- TODO: replace things with relevant sets
-- TODO: handle circular dependencies
checkModule :: Env [TCTopLevelDecl]
  -> UModule PhParse
  -> Except Err (Env [TCTopLevelDecl])
checkModule tlDecls (Ann src (RefModule typ name refName)) =
  case M.lookup refName (M.map getModules tlDecls) of
    Nothing -> throwLocatedE UnboundModuleErr src $ pshow refName
    Just refs -> case refs of
      -- TODO: NE.NonEmpty instead of []? Checking [] is annoying.
      []  -> throwLocatedE CompilerErr src
        "module name exists but is not bound in module expansion"
      _   -> do
         -- TODO: add disambiguation attempts
        when (length refs /= 1) $
          throwLocatedE AmbiguousNamedRenamingRefErr src $
            pshow refName <> "'. Candidates are: " <> pshow refs
        -- cast module: do we need to check out the deps?
        ~(Ann _ (UModule _ _ decls deps)) <- castModule typ (head refs)
        let renamedModule = UModule typ name decls deps :: UModule' PhCheck
            renamedModuleDecl =
              UModuleDecl $ Ann (LocalDecl src) renamedModule
        return $ M.insertWith (<>) name [renamedModuleDecl] tlDecls

checkModule tlDecls (Ann modSrc (UModule moduleType moduleName decls deps)) = do
  let modules = M.map getModules tlDecls
      namedRenamings = M.map getNamedRenamings tlDecls
  -- Step 1: expand uses
  (depScope, checkedDeps) <- foldl joinTuple (M.empty, M.empty) <$>
      mapM (buildModuleFromDependency namedRenamings modules) deps
  -- Step 2: register types
  let baseScope = M.unionWith (<>) depScope $ M.fromList types
  -- Step 3: check and register protos
  protoScope <- foldM registerProto baseScope callables
  -- Step 4: check bodies if allowed
  (finalScope, callablesTC) <- mapAccumM checkBody protoScope callables
  -- Step 5: check that everything is resolved if program
  -- TODO: return typechecked elements here.
  when (moduleType == Program) $
      traverse_ (checkImplemented finalScope) callablesTC
  -- TODO: check that name is unique?
  -- TODO: make module
  let resultModuleDecl = UModuleDecl $
        Ann { _ann = LocalDecl modSrc
            , _elem = UModule moduleType moduleName finalScope checkedDeps
            } -- :: TCModule
  return $ M.insertWith (<>) moduleName [resultModuleDecl] tlDecls
  where
    types :: [(Name, [UDecl PhCheck])]
    types = [ (typeName, [Ann { _ann = [LocalDecl srcAnn]
                              , _elem = UType typeName
                              }])
            | Ann srcAnn (UType typeName) <- decls]
    callables = [d | d@(Ann _ UCallable {}) <- decls]

    joinTuple (a, b) (a', b') = (M.unionWith (<>) a a', M.unionWith (<>) b b')

    -- TODO: deal with guards (unify, check)
    -- TODO: improve error messages
    checkBody
      :: ModuleScope
      -> UDecl PhParse
      -> Except Err (ModuleScope, UDecl PhCheck)
    checkBody env decl@(~(Ann src (UCallable ctype fname args retType
                                             mguard body)))
      | moduleType /= Concept, Axiom <- ctype =
          throwLocatedE DeclContextErr src
            "axioms can only be declared in concepts"
      | Nothing <- body, Axiom <- ctype =
          throwLocatedE InvalidDeclErr src "axiom without a body"
      | moduleType `notElem` [Implementation, Program], Just _ <- body,
        ctype `elem` [Function, Procedure] =
          throwLocatedE InvalidDeclErr src $ pshow ctype <>
            " can not have a body in " <> pshow moduleType
      -- TODO: handle case for programs
      | Nothing <- body = (,) env <$> checkProto env decl
      | Just expr <- body = do
          -- TODO: fix, propagate required type forward?
          bodyTC <- annotateScopedExpr env (initScope args) (Just retType) expr
          ~(Ann typAnn (UCallable _ declName argsTC _ guardTC _)) <-
              checkProto env decl
          let annDeclTC = Ann typAnn $
                UCallable ctype declName argsTC retType guardTC
                          (Just bodyTC)
          if _ann bodyTC == retType
          then return (M.insertWith (<>) fname [annDeclTC] env, annDeclTC)
          else throwLocatedE TypeErr src $
            "expected return type to be " <> pshow retType <> " in " <>
            pshow ctype <> "but return value has type " <>
            pshow (_ann bodyTC)

    checkImplemented :: ModuleScope -> UDecl PhCheck -> Except Err ()
    checkImplemented env callable@(~(Ann declO (UCallable _ callableName
                                                          _ _ _ body)))
      | Just _ <- body = return ()
      | Nothing <- body = do
          let ~(Just matches) = M.lookup callableName env
              anonDefinedMatches =
                map (mkAnonProto <$$>) $ filter isDefined matches
              src = srcCtx $ head declO -- TODO: is head the best one to get?
          when ((mkAnonProto <$$> callable) `notElem` anonDefinedMatches) $
            throwLocatedE InvalidDeclErr src $ pshow callable <>
              " was left unimplemented in program"

    mkAnonProto ~(UCallable ctype callableName args retType mguard mbody) =
      UCallable ctype callableName (map (mkAnonVar <$$>) args) retType
                mguard mbody
    mkAnonVar (Var mode _ typ) = Var mode (GenName "#anon#") typ
    isDefined ~(Ann _ (UCallable _ _ _ _ _ mbody)) = isJust mbody

-- TODO: finish module casting.
castModule
  :: Monad t
  => UModuleType
  -> UModule PhCheck
  -> ExceptT Err t (UModule PhCheck)
castModule dstTyp origModule@(Ann declO (UModule srcTyp _ _ _)) = do
  let moduleWithSwappedType = return $ swapTyp <$$> origModule
  case (srcTyp, dstTyp) of
    (Signature, Concept) -> moduleWithSwappedType
    (Signature, Implementation) -> moduleWithSwappedType
    (Signature, Program) -> noCast
    (Concept, Signature) -> undefined -- TODO: strip axioms
    (Concept, Implementation) -> undefined -- TODO: strip axioms?
    (Concept, Program) -> noCast
    (Implementation, Signature) -> undefined -- TODO: mkSignature
    (Implementation, Concept) -> undefined -- TODO: mkSignature?
    (Implementation, Program) -> undefined -- TODO: check if valid program?
    (Program, Signature) -> undefined -- TODO: mkSignature
    (Program, Concept) -> undefined -- TODO: mkSignature?
    (Program, Implementation) -> moduleWithSwappedType
    _ -> return origModule
  where
    noCast = throwLocatedE MiscErr (srcCtx declO) $
      pshow srcTyp <> " can not be casted to " <> pshow dstTyp
    swapTyp :: UModule' PhCheck -> UModule' PhCheck
    swapTyp (UModule _ name decls deps) = UModule dstTyp name decls deps
    swapTyp (RefModule _ _ v) = absurd v

castModule _ (Ann _ (RefModule _ _ v)) = absurd v
-- TODO: sanity check modes

annotateScopedExpr ::
  ModuleScope ->
  VarScope ->
  Maybe UType ->
  UExpr PhParse ->
  Except Err (UExpr PhCheck)
annotateScopedExpr inputModule inputScope inputMaybeExprType e = do
  snd <$> go inputModule inputScope inputMaybeExprType e
  where
  -- maybeExprType is a parameter to disambiguate function calls overloaded
  -- solely on return types. It will *not* be used if the type can be inferred
  -- without it, and there is thus no guarantee that the resulting annotated
  -- expression will carry the specified type annotation. If this is important,
  -- it must be checked outside the call.
  go :: ModuleScope -> VarScope -> Maybe UType -> UExpr PhParse
     -> Except Err (VarScope, UExpr PhCheck)
  go modul scope maybeExprType (Ann src expr) = case expr of
    -- TODO: annotate type and mode on variables
    -- TODO: deal with mode
    UVar (Ann _ (Var _ name typ)) -> case M.lookup name scope of
        Nothing -> throwLocatedE UnboundVarErr src (pshow name)
        Just annScopeVar@(Ann varType _) -> let typAnn = fromJust typ in
          if isNothing typ || typAnn == varType
          then return (scope, Ann { _ann = varType, _elem = UVar annScopeVar })
          else throwLocatedE MiscErr src $ "got conflicting type " <>
            "annotation for var " <> pshow name <> ": " <> pshow name <>
            " has type " <> pshow varType <> " but type annotation is " <>
            pshow typAnn
    -- TODO: deal with casting
    UCall name args maybeCallCast -> do
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
            candidates = filter (isCompatibleFunctionDecl argTypes) $
              M.findWithDefault [] name modul
            exprTC = UCall name argsTC maybeCallCast
        -- TODO: stop using list to reach return val, maybe use Data.Sequence?
        -- TODO: do something with procedures and axioms here?
        case candidates of
          []  -> throwLocatedE UnboundFunctionErr src $ pshow name
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
    UBlockExpr exprStmts -> do
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
          newBlock = UBlockExpr $ NE.fromList (initExprStmtsTC <>
                                               [lastExprStmtTC])
      return (endScope, Ann { _ann = _ann lastExprStmtTC, _elem = newBlock })
    ULet mode name maybeType maybeExpr -> do
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
          unless (mode == UOut) $
            throwLocatedE ModeMismatchErr src $ "variable " <> pshow name <>
              " is declared with mode " <> pshow mode <> " but is not " <>
              "initialized"
          -- Variable is unset, therefore has to be UOut.
          let newVar = Var UOut name (Just typ)
          return ( M.insert name Ann { _ann = typ, _elem = newVar } scope
                 , Ann Unit (ULet mode name maybeType Nothing)
                 )
        Just rhsExpr -> do
          (scope', rhsExprTC) <- go modul scope
              (maybeType <|> maybeExprType) rhsExpr
          let exprTC = ULet mode name maybeType (Just rhsExprTC)
              rhsType = _ann rhsExprTC
          case maybeType of
            Nothing -> do
              let newVar = Var mode name (Just rhsType)
              return ( M.insert name (Ann rhsType newVar) scope'
                     , Ann { _ann = Unit, _elem = exprTC }
                     )
            Just typ -> do
              unless (typ == rhsType) $
                throwLocatedE TypeErr src $ "variable " <> pshow name <>
                  " has type annotation " <> pshow typ <> " but type " <>
                  "of assignment expression is " <> pshow rhsType
              when (mode == UOut) $
                throwLocatedE CompilerErr src $ "mode of variable " <>
                  "declaration can not be " <> pshow mode <> " if an " <>
                  "assignment expression is provided"
              let newVar = Var mode name (Just typ)
              return ( M.insert name Ann { _ann = typ, _elem = newVar } scope'
                     , Ann { _ann = Unit, _elem = exprTC }
                     )
    UIf cond bTrue bFalse -> do
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
      let exprTC = UIf condExprTC trueExprTC falseExprTC
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
    UAssert cond -> do
      (scope', newCond) <- go modul scope (Just Pred) cond
      when (_ann newCond /= Pred) $
        throwLocatedE TypeErr src $ "expected expression to have type " <>
          pshow Pred <> " in predicate but got " <> pshow (_ann newCond)
      return (scope', Ann { _ann = Unit, _elem = UAssert newCond })
    USkip -> return (scope, Ann { _ann = Unit, _elem = USkip })
    -- TODO: use for annotating AST

  getReturnType ~(Ann _ (UCallable _ _ _ returnType _ _)) = returnType

  isCompatibleFunctionDecl :: [UType] -> UDecl PhCheck -> Bool
  isCompatibleFunctionDecl typeConstraints (Ann _ (UCallable _ _ args _ _ _))
    | length args /= length typeConstraints = False
    | otherwise = and $ zipWith (\x y -> x == _ann y) typeConstraints args
  isCompatibleFunctionDecl _ _ = False

checkTypeExists ::
  ModuleScope ->
  WithSrc Name ->
  Except Err ()
checkTypeExists modul (WithSrc src name)
  | Unit <- name = return ()
  | Pred <- name = return ()
  | otherwise = case M.lookup name modul of
      Nothing      -> throwLocatedE UnboundTypeErr src (pshow name)
      Just matches -> if Ann undefined (UType name) `elem` matches
                      then return ()
                      else throwLocatedE UnboundTypeErr src (pshow name)

-- TODO: ensure functions have a return type defined, though should be handled
-- by parsing.

registerProto ::
  ModuleScope ->
  UDecl PhParse ->
  Except Err ModuleScope
registerProto modul annDecl
  | Ann _ (UCallable _ name _ _ _ _) <- annDecl = do
      -- TODO: ensure bodies are registered later on
      checkedProto <- checkProto modul annDecl
      return $ M.insertWith (<>) name [checkedProto] modul
  | otherwise = return modul

-- TODO: check for procedures, predicates, axioms (this is only for func)?
-- TODO: check guard
checkProto :: ModuleScope -> UDecl PhParse -> Except Err (UDecl PhCheck)
checkProto modul ~(Ann src (UCallable ctype name args retType mguard _)) = do
  checkedArgs <- checkArgs args
  checkTypeExists modul (WithSrc src retType)
  return Ann { _ann = [LocalDecl src]
             , _elem = UCallable ctype name checkedArgs retType Nothing Nothing -- TODO: reinsert guard
             }
  where checkArgs :: [UVar PhParse] -> Except Err [UVar PhCheck]
        checkArgs vars = do
          -- TODO: make sure there is no need to check
          --when (ctype /= Function) $ error "TODO: proc/axiom/pred"
          let varSet = S.fromList [_varName v | (Ann _ v) <- vars]
          if S.size varSet /= L.length vars
          then throwLocatedE MiscErr src $
            "duplicate argument names in declaration of " <> pshow name
          else if not $ null [v | v@(Ann _ (Var _ _ Nothing)) <- vars]
          then throwLocatedE CompilerErr src
            "argument missing accompanying type binding in function prototype"
          else mapM checkArgType vars

        checkArgType :: UVar PhParse -> Except Err (UVar PhCheck)
        checkArgType var
          | Ann argSrc (Var mode varName (Just typ)) <- var = do
              checkTypeExists modul (WithSrc argSrc typ)
              return $ Ann typ (Var mode varName (Just typ))
          | otherwise = error "unreachable (checkArgs)"

buildModuleFromDependency
  :: Env [UNamedRenaming PhCheck]
  -> Env [TCModule]
  -> UModuleDep PhParse
  -> Except Err (Env [TCDecl], Env [TCModuleDep])
buildModuleFromDependency namedRenamings pkg
                          (Ann src (UModuleDep name renamings)) =
  case M.lookup name pkg of
    Nothing -> throwLocatedE UnboundModuleErr src $ pshow name
    -- TODO: make NonEmpty elements in Env?
    Just [] -> throwLocatedE CompilerErr src "module has been removed from env"
    Just [m] -> do
      -- TODO: gather here env of known renamings
      checkedRenamings <- mapM (expandRenamingBlock namedRenamings) renamings
      renamedDecls <- foldM applyRenamingBlock (moduleDecls m) checkedRenamings
      -- TODO: do this well
      let checkedDep = Ann src (UModuleDep name checkedRenamings)
      return (renamedDecls, M.singleton name [checkedDep])
    Just _ -> throwLocatedE NotImplementedErr src
      "do not yet deal with ambiguous module names"
      -- TODO: implement fully qualified use names"

-- TODO: cleanup duplicates, make a set of UDecl instead of a list?
-- TODO: finish renamings
-- TODO: add annotations to renamings, work with something better than just
--       name.
-- TODO: improve error diagnostics
applyRenamingBlock
  :: ModuleScope
  -> URenamingBlock PhCheck
  -> Except Err ModuleScope
applyRenamingBlock modul renamingBlock@(Ann src (URenamingBlock renamings)) = do
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
           applyRenamingBlock modul (URenamingBlock annR' <$$ renamingBlock)
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
applyRenaming :: UDecl PhCheck -> InlineRenaming -> UDecl PhCheck
applyRenaming annDecl renaming = applyRenamingInDecl <$$> annDecl
  where replaceName' = flip replaceName renaming
        applyRenamingInDecl :: UDecl' PhCheck -> UDecl' PhCheck
        applyRenamingInDecl decl
          | UType name <- decl = UType $ replaceName' name
          | UCallable ctyp name vars retType mguard mbody <- decl =
                UCallable ctyp (replaceName' name)
                  (map applyRenamingInVar vars) (replaceName' retType)
                  (applyRenamingInExpr <$> mguard)
                  (applyRenamingInExpr <$> mbody)
        applyRenamingInVar :: UVar PhCheck -> UVar PhCheck
        applyRenamingInVar (Ann typAnn (Var mode name typ)) =
          Ann (replaceName' typAnn) $ Var mode name (replaceName' <$> typ)
        applyRenamingInExpr :: UExpr PhCheck -> UExpr PhCheck
        applyRenamingInExpr (Ann typAnn expr) =
          Ann (replaceName' typAnn) (applyRenamingInExpr' expr)
        applyRenamingInExpr' expr
          | UVar var <- expr = UVar $ applyRenamingInVar var
          | UCall name vars typ <- expr =
                UCall (replaceName' name) (map applyRenamingInExpr vars)
                  (replaceName' <$> typ)
          | UBlockExpr stmts <- expr =
                UBlockExpr $ NE.map applyRenamingInExpr stmts
          | ULet mode name typ assignmentExpr <- expr =
                ULet mode name (replaceName' <$> typ)
                     (applyRenamingInExpr <$> assignmentExpr)
          | UIf cond bTrue bFalse <- expr =
                UIf (applyRenamingInExpr cond) (applyRenamingInExpr bTrue)
                    (applyRenamingInExpr bFalse)
          | UAssert cond <- expr = UAssert (applyRenamingInExpr cond)
          | USkip <- expr = USkip


-- === utils ===

mapAccumM :: (Traversable t, Monad m)
          => (a -> b -> m (a, c)) -> a -> t b -> m (a, t c)
mapAccumM f a tb = swap <$> mapM go tb `runStateT` a
  where go b = do s <- get
                  (s', r) <- lift $ f s b
                  put s'
                  return r

isStateful :: UExpr p -> Bool
isStateful (Ann _ expr) = case expr of
  -- Operations that can affect the scope are:
  -- - variable declarations/variable assignments
  -- - calls to procedures
  UVar _ -> False
  UCall (FuncName _) args _ -> any isStateful args
  UCall (ProcName _) _ _ -> True
  UCall {} -> error $ "call to something that is neither a type " <>
                      "of function nor a procedure."
  UBlockExpr stmts -> or $ NE.map isStateful stmts
  ULet {} -> True
  UIf cond bTrue bFalse -> any isStateful [cond, bTrue, bFalse]
  UAssert cond -> isStateful cond
  USkip -> False
