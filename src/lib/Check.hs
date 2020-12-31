{-# LANGUAGE OverloadedStrings #-}

module Check (checkPackage, checkModule) where

import Control.Monad (foldM)
import Control.Monad.Except
--import Debug.Trace (trace)
import Data.Foldable (traverse_)
import Data.Maybe (fromJust, isJust, isNothing)
import Data.Monoid ((<>))
import Data.Traversable (for)

import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Text.Lazy as T

import Env
import PPrint
import Syntax

-- TODO: WIP Err
type Err = WithSrc T.Text

type Scope = Env UVar

initScope :: [UVar] -> Scope
initScope = M.fromList . map (\v@(WithSrc _ (Var _ name _)) -> (name, v))

checkPackage :: GlobalEnv -> UPackage -> Except Err GlobalEnv
checkPackage _ UPackage {} = undefined

-- TODO: can variables exist without a type in the scope?
-- TODO: replace things with relevant sets
-- TODO: handle circular dependencies
checkModule :: Package -> UModule -> Except Err Package
--checkModule pkg modul | trace (pshow modul) False = undefined
checkModule pkg (WithSrc _ (UModule moduleType name decls deps)) = do
  -- Step 1: expand uses
  uses <- foldl joinEnvs M.empty <$> mapM (buildModuleFromDep pkg) deps
  -- Step 2: register types
  let baseEnv = M.unionWith (<>) uses $ M.fromList types
  -- Step 3: check and register protos
  env <- foldM registerProto baseEnv callables
  -- Step 4: check bodies if allowed
  traverse_ (checkBody env) callables
  -- Step 5: check that everything is resolved if program
  when (moduleType == Program) $ traverse_ (checkImplemented env) callables
  -- TODO: check that name is unique?
  return $ M.insert name env pkg
  where
    types = [(typeName, [decl]) | decl@(WithSrc _ (UType typeName)) <- decls]
    callables = [d | d@(WithSrc _ UCallable {}) <- decls]
    joinEnvs = M.unionWith (<>)
    checkBody env callable@(~(WithSrc _ (UCallable callableType _ args body)))
      | moduleType /= Concept, Axiom <- callableType =
          throwError $ "Axioms can only be declared in concepts." <$ callable
      | Nothing <- body, Axiom <- callableType =
          throwError $ "Axiom without a body." <$ callable
      | moduleType `notElem` [Implementation, Program], Just _ <- body,
        callableType `elem` [Function, Procedure] =
          throwError $ (pshow callableType <> " can not have a body in " <>
                        pshow moduleType) <$ callable
      -- TODO: handle case for programs
      | Nothing <- body = return ()
      | Just expr <- body = do
          let returnType = getReturnType callable
          possibleTypes <- inferScopedExprType env (initScope args) expr
          case callableType of
            Function -> if returnType `elem` possibleTypes then return ()
                        else throwError $ ("Expected return type " <>
                          pshow returnType <> " but body has one of the " <>
                          "following types: " <> pshow possibleTypes <> ".") <$
                          callable
            _        -> if length possibleTypes == 1 &&
                          head possibleTypes == returnType
                        then return ()
                        else throwError $ ("Expected " <> pshow returnType <>
                          " return type in " <> pshow callableType <>
                          "but return value has one of the following " <>
                          "types: " <> pshow possibleTypes <> ".") <$ callable

    checkImplemented :: Module -> UDecl -> Except Err ()
    checkImplemented env callable@(~(WithSrc _ (UCallable _ callableName
                                                          _ body)))
      | Just _ <- body = return ()
      | Nothing <- body = do
          let ~(Just matches) = M.lookup callableName env
              anonDefinedMatches =
                map (mkAnonProto <$>) $ filter isDefined matches
          when ((mkAnonProto <$> callable) `notElem` anonDefinedMatches) $
               throwError $ "Callable is unimplemented in program." <$ callable

    mkAnonProto ~(UCallable callableType callableName args body) =
      UCallable callableType callableName (map (mkAnonVar <$>) args) body
    mkAnonVar (Var mode _ typ) = Var mode (GenName "#anon#") typ
    isDefined ~(WithSrc _ (UCallable _ _ _ body)) = isJust body

-- TODO: sanity check modes

inferScopedExprType :: Module -> Scope -> UExpr -> Except Err [UType]
inferScopedExprType inputModule inputScope e = do
  snd <$> inferScopedExprType' inputModule inputScope e
  where
  inferScopedExprType' :: Module -> Scope -> UExpr
                         -> Except Err (Scope, [UType])
  inferScopedExprType' modul scope exprWSrc@(WithSrc src expr) = case expr of
    -- TODO: annotate type and mode here; return modes?
    UVar (WithSrc _ (Var _ name typ)) -> case M.lookup name scope of
        Nothing -> throwError $ "No such variable in current scope" <$ exprWSrc
        Just (WithSrc _ (Var _ _ typ')) -> let typAnn = fromJust typ in
          case typ' of
            Nothing     -> if isNothing typ then return (scope, [])
                           else return (scope, [typAnn])
            Just typSet -> if isNothing typ || typSet == typAnn
                           then return (scope, [typSet])
                           else
                             throwError $ "Conflicting types for var" <$
                                          exprWSrc
    UCall name args _   -> do
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
        -- First, expand the scope by running through the arguments; at every
        -- sub function calls, variables may get annotated.
        -- TODO: opportunities for optimization here if need be.
        scope' <- foldM (((fst <$>) <$>) . inferScopedExprType' modul) scope
                        args
        -- Second, actually infer the types of the arguments as much as
        -- possible.
        -- TODO: is traverse a mapAccumL?
        argTypes <- traverse (inferScopedExprType modul scope') args
        let candidates = filter (isCompatibleFunctionDecl argTypes) $
              M.findWithDefault [] name modul
        -- TODO: stop using list to reach return val, maybe use Data.Sequence?
        -- TODO: do something with procedures and axioms here?
        case candidates of
          []  -> throwError $ ("No corresponding function with name '" <>
                               pshow name <> "' in scope") <$ exprWSrc
          [fun] -> setScopeAndReturnType scope' args fun
          -- In this case, we have many matches. This may happen for several
          -- reasons:
          -- (1) the arguments are not explicitly typed and resolution is
          --     ambiguous
          -- (2) the arguments are fully specified but the function can be
          --     overloaded solely on its return type. In this case, we request
          --     explicit typing from the user
          -- (3) we registered several prototypes that refer to the same
          --     function in different ways. This can happen right now because
          --     of how the environment is implemented.
          --     TODO: use sets, and hash arguments ignoring their type to fix
          --     (3).
          matches ->
            let protos = S.fromList $ map getFunctionSignature matches in
            if S.size protos == 1
            then setScopeAndReturnType scope' args (L.head matches)
            else throwError $
                "Ambiguous function calls, several possible matches" <$ exprWSrc
    UBlockExpr exprStmts -> do
      -- The type of a block expression is the type of its last expression
      -- statement. Therefore, we ignore the return types of all the other
      -- statements and only go through them to build the required scope.
      scope' <- foldM (((fst <$>) <$>) . inferScopedExprType' modul) scope
                      (NE.init exprStmts)
      (scope'', types) <- inferScopedExprType' modul scope' (NE.last exprStmts)
      -- Return the old scope with potential new annotations; this makes sure
      -- that variables declared in the current scope are eliminated.
      -- Note that we do not yet have nested scopes; this means that it is
      -- never possible to shadow names from an outer scope.
      let endScope = M.fromList $ map (\k -> (k, scope'' M.! k)) (M.keys scope)
      return (endScope, types)
    ULet mode name maybeType maybeExpr -> do
      unless (isNothing $ M.lookup name scope) $
           throwError $ ("A variable with name " <> pshow name <> "already " <>
                         "exists in the current scope.") <$ exprWSrc
      case maybeExpr of
        Nothing -> do
          when (isNothing maybeType) $
               throwError $ ("Attempting to declare variable with no type " <>
                             "annotation or assignment expression") <$ exprWSrc
          let (Just typ) = maybeType
          checkTypeExists (typ <$ exprWSrc) modul
          unless (mode == UOut) $
                 throwError $ ("Variable can not be set to " <> pshow mode <>
                               " without being initialized.") <$ exprWSrc
          -- Variable is unset, therefore has to be UOut.
          return (M.insert name (WithSrc src (Var UOut name (Just typ))) scope,
                  [Unit])
        Just expr' -> do
          (scope', types) <- inferScopedExprType' modul scope expr'
          case maybeType of
            Nothing -> do
              when (L.length types /= 1) $
                   throwError $ ("Could not infer variable type from " <>
                                 "assignment expression.") <$ exprWSrc
              return
                (M.insert name (WithSrc src (Var mode name (Just (head types))))
                          scope',
                 [Unit])
            Just typ -> do
              unless (typ `elem` types) $
                     throwError $ "Attempted to cast variable to invalid type"
                                  <$ exprWSrc
              when (mode == UOut) $
                    throwError $ "Invalid mode, shouldn't happen" <$ exprWSrc
              return
                (M.insert name (WithSrc src (Var mode name (Just typ))) scope',
                 [Unit])
    UIf cond bTrue bFalse -> do
      (condScope, condTypes) <- inferScopedExprType' modul scope cond
      when (condTypes /= [Pred]) $
           throwError $ ("Expected predicate but got one of the following " <>
                         "types: " <> pshow condTypes <> ".") <$ exprWSrc
      (trueScope, bTrueTypes) <- inferScopedExprType' modul condScope bTrue
      (falseScope, bFalseTypes) <- inferScopedExprType' modul condScope bFalse

      -- Unifying lists. Probably they should be sets?
      -- TODO: make sets
      let commonTypes = filter (`elem` bFalseTypes) bTrueTypes
      when (null commonTypes) $
           throwError $ ("Could not unify if branches; candidates for True " <>
                         "branch are " <> pshow bTrueTypes <> " and " <>
                         "candidates for False branch are " <>
                         pshow bFalseTypes <> ".") <$ exprWSrc
      -- TODO: join scopes. Description below:
      -- TODO: add sets of possible types to variables for type inference.
      -- TODO: can we set a variable to be "linearly assigned to" (only one
      --       time)?
      -- Modes are either upgraded in both branches, or we should throw an
      -- error. Each variable v in the parent scope has to satisfy one of the
      -- following properties:
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
      let scopeVars = M.toList $ M.map (\(WithSrc _ (Var _ name _)) ->
               let f = fromJust . M.lookup name in
               (f trueScope, f falseScope)) scope
      resultScope <- M.fromList <$> for scopeVars (
          \(n, (WithSrc vSrc v1, WithSrc _ v2)) -> do
              -- (5) || (6) || (7) <=> _varMode v1 == _varMode v2
              when (_varMode v1 /= _varMode v2) $
                  throwError $ ("Mode of variable " <> pshow n <> "is " <>
                                "inconsistent across branches.") <$ exprWSrc
              finalType <- case (_varType v1, _varType v2) of
                -- (2) || (3) || (4)
                (Just x, Just y) -> do
                    when (x /= y) $
                         throwError $ ("Variable " <> pshow n <> " has " <>
                                       "inconsistent types across branches. " <>
                                       "Candidate is " <> pshow x <> " in " <>
                                       "the True branch and " <> pshow y <>
                                       " in the False branch.") <$ exprWSrc
                    return $ Just x
                -- TODO: possible (2) when type inference is better
                (Just x, Nothing)  ->
                    throwError $ ("Could not infer type of " <> pshow n <>
                                  " in the False branch of if statement but " <>
                                  " type is set to " <> pshow x <> " in " <>
                                  "the True branch.") <$ exprWSrc
                -- TODO: possible (2) when type inference is better
                (Nothing, Just y)  ->
                    throwError $ ("Could not infer type of " <> pshow n <>
                                  " in the True branch of if statement but " <>
                                  " type is set to " <> pshow y <> " in " <>
                                  "the False branch.") <$ exprWSrc
                -- (1)
                (Nothing, Nothing) -> return Nothing

              return (n, WithSrc vSrc $ Var (_varMode v1) n finalType))
      return (resultScope, commonTypes)
    UAssert cond -> do
      (scope', types) <- inferScopedExprType' modul scope cond
      when (types /= [Pred]) $
           throwError $ ("Expected Predicate call in assertion but " <>
                         "expression has one of the following possible " <>
                         "types " <> pshow types <> ".") <$ exprWSrc
      return (scope', [Unit])
    USkip -> return (scope, [Unit])
    -- TODO: use for annotating AST
    UTypedExpr {} -> undefined

  getFunctionSignature (WithSrc _ (UFunc _ args _)) = map getArgType args
  getFunctionSignature _ = error ("Expected function parameter in " <>
                                  "getFunctionSignature")

  setScopeAndReturnType scope' args fun@(~(WithSrc _ (UFunc _ argVars _))) =
    let returnType = getReturnType fun in
    return (foldl updateScope scope' (zip args argVars), [returnType])

  isCompatibleFunctionDecl :: [[UType]] -> UDecl -> Bool
  isCompatibleFunctionDecl typeConstraints (WithSrc _ (UFunc _ args _))
    -- The last argument stored in a function declaration corresponds to the
    -- return type.
    | length args /= length typeConstraints + 1 = False
    | otherwise = and $ zipWith fitsTypeConstraints typeConstraints args
  isCompatibleFunctionDecl _ _ = False

  fitsTypeConstraints :: [UType] -> UVar -> Bool
  fitsTypeConstraints typeConstraints ~(WithSrc _ (Var _ _ (Just typ)))
    -- If an argument has an unspecified argType, it is a wildcard in the
    -- current call,
    | [] <- typeConstraints = True
    | typ `elem` typeConstraints = True
    | otherwise = False

  updateScope :: Scope -> (UExpr, UVar) -> Scope
  updateScope scope (WithSrc _ expr, ~(WithSrc _ (Var _ _ typ@(Just _))))
    | UVar (WithSrc _ (Var _ name _)) <- expr = case M.lookup name scope of
          Just v@(WithSrc _ (Var mode _ Nothing)) ->
              M.insert name (Var mode name typ <$ v) scope
          Just (WithSrc _ (Var _ _ (Just _))) -> scope
          _ -> error "This should not happen (updateScope)"
    | otherwise = scope

checkTypeExists :: WithSrc Name -> Module -> Except Err ()
checkTypeExists (WithSrc src name) modul = case M.lookup name modul of
  Nothing      -> throwError (WithSrc src err)
  Just matches -> if NoCtx (UType name) `elem` matches then return ()
                  else throwError (WithSrc src err)
  where
    err = "Type " <> pshow name <> " does not exist in current scope."

-- TODO: ensure functions have a return type defined, though should be handled
-- by parsing.
registerProto :: Module -> UDecl -> Except Err Module
registerProto modul declWSrc@(WithSrc _ decl)
  | UCallable _ name args _ <- decl = do
      -- TODO: check for procedures, predicates, axioms (this is only for func)?
      checkArgs args
      >> return (M.insertWith (<>) name [declWSrc] modul)
  | otherwise = return modul
  where checkArgs :: [UVar] -> Except Err ()
        checkArgs vars = do
          -- TODO: make sure there is no need to check
          --when (callableType /= Function) $ error "TODO: proc/axiom/pred"
          let varSet = S.fromList [name | (WithSrc _ (Var _ name _)) <- vars]

          if S.size varSet /= L.length vars
          then throwError $ "Duplicate argument names in function prototype." <$
                            declWSrc
          else if not $ null [v | v@(WithSrc _ (Var _ _ Nothing)) <- vars]
          then throwError $ ("Argument missing accompanying type binding in " <>
                             "function prototype") <$ declWSrc
          else traverse_ checkArgType vars

        checkArgType :: UVar -> Except Err ()
        checkArgType var
          | (WithSrc _ (Var _ _ (Just typ))) <- var =
              checkTypeExists (typ <$ var) modul
          | otherwise = error "unreachable (checkArgs)"

buildModuleFromDep :: Package -> UModuleDep -> Except Err Module
buildModuleFromDep pkg dep@(WithSrc _ (UModuleDep name renamings)) =
  case M.lookup name pkg of
    Nothing ->
      throwError $ ("No module named " <> pshow name <> "in scope.") <$ dep
    Just modul -> foldM applyRenamingBlock modul renamings

-- TODO: cleanup duplicates, make a set of UDecl instead of a list?
-- TODO: finish renamings
-- TODO: add annotations to renamings, work with something better than just
--       name.
-- TODO: improve error diagnostics
applyRenamingBlock :: Module -> RenamingBlock -> Except Err Module
applyRenamingBlock modul renamingBlock@(WithSrc _ renamings) = do
  let renamingMap = M.fromList renamings
      (sources, targets) = (L.map fst renamings, L.map snd renamings)
      -- TODO: will have to modify when renamings can be more atomic and
      -- namespace can be specified.
      filterSources srcs ns =
        filter (\(Name _ src) -> isNothing $ M.lookup (Name ns src) modul) srcs
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
                                    renamings
      -- TODO: cleanup
      r' = zip occurOnBothSides
                    [GenName ("gen#" ++ show i) | i <- [1..] :: [Int]]
      r'' = unambiguousRenamings <>
              [(freeName, fromJust $ M.lookup source renamingMap) |
               (source, freeName) <- r']
  (modul', renamings') <- (
      if M.size renamingMap /= L.length renamings
      then throwError $ "Duplicate key in renaming block." <$ renamingBlock
      else if not (null unknownSources)
      then throwError $ "Renaming block has unknown sources." <$ renamingBlock
      else if not (null occurOnBothSides)
      then applyRenamingBlock modul (r' <$ renamingBlock)
           >>= \modul' -> return (modul', r'')
      else return (modul, renamings))
  return $ M.fromListWith (<>) $ L.map (\(k, decls) ->
          (tryAllRenamings replaceName k renamings',
           L.map (flip (tryAllRenamings applyRenaming) renamings') decls)) $
           M.toList modul'
  where tryAllRenamings renamingFun target = foldl renamingFun target

-- TODO: specialize replacements based on namespaces?
replaceName :: Name -> Renaming -> Name
replaceName origName@(Name ns nameStr) (Name _ sourceStr, Name _ targetStr) =
  if sourceStr == nameStr then Name ns targetStr else origName

-- Applies a renaming to a declaration. Renamings only affect names defined at
-- declaration level; this means that they do not affect local variables.
applyRenaming :: UDecl -> Renaming -> UDecl
applyRenaming _decl renaming = applyRenamingInDecl <$> _decl
  where replaceName' = flip replaceName renaming
        applyRenamingInDecl decl
          | UType name <- decl = UType $ replaceName' name
          | UCallable ctyp name vars expr <- decl =
                UCallable ctyp (replaceName' name)
                  (map (applyRenamingInVar <$>) vars)
                  (applyRenamingInExpr <$> expr)
        applyRenamingInVar (Var mode name typ) =
          Var mode name (replaceName' <$> typ)
        applyRenamingInExpr = (applyRenamingInExpr' <$>)
        applyRenamingInExpr' expr
          | UVar var <- expr = UVar $ applyRenamingInVar <$> var
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
          | UTypedExpr expr' <- expr = UTypedExpr (applyRenamingInExpr expr')


-- === utils ===

getArgType :: UVar -> UType
getArgType ~(WithSrc _ (Var _ _ (Just typ))) = typ

getReturnType :: UDecl -> UType
getReturnType ~(WithSrc _ (UCallable callableType _ args _)) =
  case callableType of
    Axiom     -> Unit
    Function  -> getArgType $ last args
    Predicate -> Pred
    Procedure -> Unit
