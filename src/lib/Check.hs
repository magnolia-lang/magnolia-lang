{-# LANGUAGE OverloadedStrings #-}

module Check (checkPackage, checkModule) where

import Control.Applicative
import Control.Monad (foldM)
import Control.Monad.Except
import Control.Monad.Trans.State
--import Debug.Trace (trace)
import Data.Foldable (traverse_)
import Data.Maybe (fromJust, isJust, isNothing)
import Data.Monoid ((<>))
import Data.Traversable (for)
import Data.Tuple (swap)

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

type VarScope = Env (UVar PhCheck)
type ModuleScope = ModuleEnv PhCheck
type PackageScope = PackageEnv PhCheck
type GlobalScope = GlobalEnv PhCheck

checkError :: SrcCtx -> T.Text -> Except Err a
checkError src err = throwError $ WithSrc src err

initScope :: [UVar p] -> VarScope
initScope = M.fromList . map mkScopeVar
  where
    mkScopeVar :: UVar p -> (Name, UVar PhCheck)
    mkScopeVar ~(Ann _ (Var mode name (Just typ))) =
      (name, Ann { _ann = typ, _elem = Var mode name (Just typ) })

checkPackage :: GlobalScope -> UPackage PhParse -> Except Err GlobalScope
checkPackage _ _ = undefined

-- TODO: can variables exist without a type in the scope?
-- TODO: replace things with relevant sets
-- TODO: handle circular dependencies
checkModule :: PackageScope -> UModule PhParse -> Except Err PackageScope
--checkModule pkg modul | trace (pshow modul) False = undefined
checkModule pkg (Ann _ (UModule moduleType moduleName decls deps)) = do
  -- Step 1: expand uses
  uses <- foldl joinEnvs M.empty <$> mapM (buildModuleFromDep pkg) deps
  -- Step 2: register types
  let baseScope = M.unionWith (<>) uses $ M.fromList types
  -- Step 3: check and register protos
  protoScope <- foldM registerProto baseScope callables
  -- Step 4: check bodies if allowed
  (finalScope, callablesTC) <- mapAccumM checkBody protoScope callables
  -- Step 5: check that everything is resolved if program
  -- TODO: return typechecked elements here.
  when (moduleType == Program) $
      traverse_ (checkImplemented finalScope) callablesTC
  -- TODO: check that name is unique?
  return $ M.insert moduleName finalScope pkg
  where
    types :: [(Name, [UDecl PhCheck])]
    types = [ (typeName, [Ann { _ann = ann, _elem = UType typeName }])
            | Ann ann (UType typeName) <- decls]
    callables = [d | d@(Ann _ UCallable {}) <- decls]
    joinEnvs = M.unionWith (<>)
    checkBody :: ModuleScope
      -> UDecl PhParse
      -> Except Err (ModuleScope, UDecl PhCheck)
    checkBody env decl@(~(Ann src (UCallable callableType fname args retType
                                             body)))
      | moduleType /= Concept, Axiom <- callableType =
          checkError src "Axioms can only be declared in concepts."
      | Nothing <- body, Axiom <- callableType =
          checkError src "Axiom without a body."
      | moduleType `notElem` [Implementation, Program], Just _ <- body,
        callableType `elem` [Function, Procedure] =
          checkError src $ pshow callableType <> " can not have a body in " <>
                           pshow moduleType
      -- TODO: handle case for programs
      | Nothing <- body = (,) env <$> checkProto env decl
      | Just expr <- body = do
          -- TODO: fix, propagate required type forward?
          bodyTC <- annotateScopedExpr env (initScope args) (Just retType) expr
          ~(Ann typAnn (UCallable _ declName argsTC _ _)) <- checkProto env decl
          let annDeclTC = Ann typAnn $
                UCallable callableType declName argsTC retType (Just bodyTC)
          if _ann bodyTC == retType
          then return (M.insertWith (<>) fname [annDeclTC] env, annDeclTC)
          else checkError src ("Expected " <> pshow retType <> " return " <>
              "type in " <> pshow callableType <> "but return value has " <>
              "type: " <> pshow (_ann bodyTC) <> ".")

    checkImplemented :: ModuleScope -> UDecl PhCheck -> Except Err ()
    checkImplemented env callable@(~(Ann src (UCallable _ callableName
                                                        _ _ body)))
      | Just _ <- body = return ()
      | Nothing <- body = do
          let ~(Just matches) = M.lookup callableName env
              anonDefinedMatches =
                map (mkAnonProto <$$>) $ filter isDefined matches
          when ((mkAnonProto <$$> callable) `notElem` anonDefinedMatches) $
               checkError src "Callable is unimplemented in program."

    mkAnonProto ~(UCallable callableType callableName args retType body) =
      UCallable callableType callableName (map (mkAnonVar <$$>) args) retType
                body
    mkAnonVar (Var mode _ typ) = Var mode (GenName "#anon#") typ
    isDefined ~(Ann _ (UCallable _ _ _ _ body)) = isJust body

-- TODO: sanity check modes

annotateScopedExpr ::
  ModuleScope ->
  VarScope ->
  Maybe UType ->
  UExpr PhParse ->
  Except Err (UExpr PhCheck)
annotateScopedExpr inputModule inputScope inputMaybeExprType e = do
  snd <$> annotateScopedExpr' inputModule inputScope inputMaybeExprType e
  where
  -- maybeExprType is a parameter to disambiguate function calls overloaded
  -- solely on return types. It will *not* be used if the type can be inferred
  -- without it, and there is thus no guarantee that the resulting annotated
  -- expression will carry the specified type annotation. If this is important,
  -- it must be checked outside the call.
  annotateScopedExpr' :: ModuleScope -> VarScope -> Maybe UType -> UExpr PhParse
                         -> Except Err (VarScope, UExpr PhCheck)
  annotateScopedExpr' modul scope maybeExprType (Ann src expr) = case expr of
    -- TODO: annotate type and mode here; return modes?
    UVar (Ann _ (Var mode name typ)) -> case M.lookup name scope of
        Nothing -> checkError src "No such variable in current scope"
        Just annScopeVar@(Ann varType _) -> let typAnn = fromJust typ in
          if isNothing typ || typAnn == varType
          then return (scope, Ann { _ann = varType, _elem = UVar annScopeVar })
          else checkError src "Conflicting types for var" -- TODO, expand
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
        when (or $ map isStateful args) $
          checkError src $ "Call expression can not have stateful " <>
                           "computations as arguments."
        argsTC <- traverse (annotateScopedExpr modul scope Nothing) args
        let argTypes = map _ann argsTC
            candidates = filter (isCompatibleFunctionDecl argTypes) $
              M.findWithDefault [] name modul
            exprTC = UCall name argsTC maybeCallCast
        -- TODO: stop using list to reach return val, maybe use Data.Sequence?
        -- TODO: do something with procedures and axioms here?
        case candidates of
          []  -> checkError src ("No corresponding function with name '" <>
                                 pshow name <> "' in scope")
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
              Nothing -> if S.size possibleTypes == 1
                         then let typAnn = getReturnType (L.head matches) in
                              return ( scope
                                     , Ann { _ann = typAnn, _elem = exprTC }
                                     )
                         else checkError src $ "Can not deduce return type " <>
                           "from function call; consider adding a type" <>
                           "annotation."
              Just cast -> if S.size possibleTypes == 1
                           then do
                             let typAnn = getReturnType (L.head matches)
                             when (not (isNothing maybeCallCast) &&
                                   typAnn /= cast) $
                               checkError src $ "No candidate matching " <>
                                 "type annotation."
                             return ( scope
                                    , Ann { _ann = typAnn, _elem = exprTC }
                                    )
                           else do
                             unless (cast `S.member` possibleTypes) $
                               checkError src $ "No candidate matching " <>
                                 "type annotation."
                             return ( scope
                                    , Ann { _ann = cast, _elem = exprTC }
                                    )
              -- TODO: ignore modes when overloading functions
    UBlockExpr exprStmts -> do
      (intermediateScope, initExprStmtsTC) <-
          mapAccumM (flip (annotateScopedExpr' modul) Nothing) scope
                    (NE.init exprStmts)
      -- The last exprStmt must be treated differently because it's potentially
      -- annotated.
      (scope', lastExprStmtTC) <-
          annotateScopedExpr' modul intermediateScope maybeExprType
                             (NE.last exprStmts)
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
           checkError src ("A variable with name " <> pshow name <>
                           "already exists in the current scope.")
      case maybeExpr of
        Nothing -> do
          when (isNothing maybeType) $
               checkError src ("Attempting to declare variable with no " <>
                               "type annotation or assignment expression")
          let (Just typ) = maybeType
          checkTypeExists modul (WithSrc src typ)
          unless (mode == UOut) $
                 checkError src ("Variable can not be set to " <> pshow mode <>
                                 " without being initialized.")
          -- Variable is unset, therefore has to be UOut.
          let newVar = Var UOut name (Just typ)
          return ( M.insert name Ann { _ann = typ, _elem = newVar } scope
                 , Ann Unit (ULet mode name maybeType Nothing)
                 )
        Just rhsExpr -> do
          (scope', rhsExprTC) <- annotateScopedExpr' modul scope
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
                  checkError src "Attempted to cast variable to invalid type"
              when (mode == UOut) $
                  checkError src "Invalid mode, shouldn't happen"
              let newVar = Var mode name (Just typ)
              return ( M.insert name Ann { _ann = typ, _elem = newVar } scope'
                     , Ann { _ann = Unit, _elem = exprTC }
                     )
    UIf cond bTrue bFalse -> do
      (condScope, condExprTC) <-
          annotateScopedExpr' modul scope (Just Pred) cond
      when (_ann condExprTC /= Pred) $
           checkError src ("Expected predicate but got the following " <>
                           "type: " <> pshow (_ann condExprTC) <> ".")
      (trueScope, trueExprTC) <-
          annotateScopedExpr' modul condScope maybeExprType bTrue
      (falseScope, falseExprTC) <-
          annotateScopedExpr' modul condScope (Just (_ann trueExprTC)) bFalse
      when (_ann trueExprTC /= _ann falseExprTC) $
          checkError src ("Could not unify if branches; True branch has " <>
                          "type " <> pshow (_ann trueExprTC) <> " and False " <>
                          "branch has type " <> pshow (_ann falseExprTC) <> ".")
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
      let scopeVars = M.toList $ M.map (\(Ann _ (Var _ name _)) ->
               let findVar = fromJust . M.lookup name in
               (findVar trueScope, findVar falseScope)) scope
      resultScope <- M.fromList <$> for scopeVars (
          \(n, (Ann typ1 v1, Ann typ2 v2)) -> do
              -- (5) || (6) || (7) <=> _varMode v1 == _varMode v2
              when (_varMode v1 /= _varMode v2) $
                  checkError src ("Mode of variable " <> pshow n <> "is " <>
                                  "inconsistent across branches.")
              when (typ1 /= typ2) $ error "Shouldn't happen" -- TODO: remove
              return (n, Ann { _ann = typ1
                             , _elem = v1
                             }))
      return (resultScope, Ann { _ann = _ann trueExprTC, _elem = exprTC })
    UAssert cond -> do
      (scope', newCond) <-
          annotateScopedExpr' modul scope (Just Pred) cond
      when (_ann newCond /= Pred) $
           checkError src ("Expected Predicate call in assertion but " <>
                           "expression has one of the following possible " <>
                           "types " <> pshow (_ann newCond) <> ".")
      return (scope', Ann { _ann = Unit, _elem = UAssert newCond })
    USkip -> return (scope, Ann { _ann = Unit, _elem = USkip })
    -- TODO: use for annotating AST

  getReturnType ~(Ann _ (UCallable _ _ _ returnType _)) = returnType

  updateScope :: VarScope -> (UExpr p, UVar PhCheck) -> VarScope
  updateScope scope (Ann _ expr, annVar@(Ann _ ~(Var _ _ typ@(Just _))))
    | UVar (Ann _ (Var _ name _)) <- expr = case M.lookup name scope of
          Just (Ann _ (Var mode _ Nothing)) ->
              M.insert name (Var mode name typ <$$ annVar) scope
          Just (Ann _ (Var _ _ (Just _))) -> scope
          _ -> error "This should not happen (updateScope)"
    | otherwise = scope

  isCompatibleFunctionDecl :: [UType] -> UDecl PhCheck -> Bool
  isCompatibleFunctionDecl typeConstraints (Ann _ (UCallable _ _ args _ _))
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
      Nothing      -> checkError src err
      Just matches -> if Ann Nothing (UType name) `elem` matches then return ()
                      else checkError src err
    where err = "Type " <> pshow name <> " does not exist in current scope."

-- TODO: ensure functions have a return type defined, though should be handled
-- by parsing.

registerProto ::
  ModuleScope ->
  UDecl PhParse ->
  Except Err ModuleScope
registerProto modul annDecl
  | Ann _ (UCallable _ name _ _ _) <- annDecl = do
      -- TODO: ensure bodies are registered later on
      checkedProto <- checkProto modul annDecl
      return $ M.insertWith (<>) name [checkedProto] modul
  | otherwise = return modul

-- TODO: check for procedures, predicates, axioms (this is only for func)?
checkProto :: ModuleScope -> UDecl PhParse -> Except Err (UDecl PhCheck)
checkProto modul ~(Ann ann (UCallable callableType name args retType _)) = do
  checkedArgs <- checkArgs args
  checkTypeExists modul (WithSrc ann retType)
  return Ann { _ann = ann
             , _elem = UCallable callableType name checkedArgs retType Nothing
             }
  where checkArgs :: [UVar PhParse] -> Except Err [UVar PhCheck]
        checkArgs vars = do
          -- TODO: make sure there is no need to check
          --when (callableType /= Function) $ error "TODO: proc/axiom/pred"
          let varSet = S.fromList [_varName v | (Ann _ v) <- vars]
          if S.size varSet /= L.length vars
          then checkError ann "Duplicate argument names in function prototype."
          else if not $ null [v | v@(Ann _ (Var _ _ Nothing)) <- vars]
          then checkError ann ("Argument missing accompanying type " <>
                               "binding in function prototype.")
          else mapM checkArgType vars

        checkArgType :: UVar PhParse -> Except Err (UVar PhCheck)
        checkArgType var
          | Ann src (Var mode varName (Just typ)) <- var = do
              checkTypeExists modul (WithSrc src typ)
              return $ Ann typ (Var mode varName (Just typ))
          | otherwise = error "unreachable (checkArgs)"

buildModuleFromDep ::
  PackageScope ->
  UModuleDep PhParse ->
  Except Err ModuleScope
buildModuleFromDep pkg (Ann ann (UModuleDep name renamings)) =
  case M.lookup name pkg of
    Nothing ->
      checkError ann ("No module named " <> pshow name <> "in scope.")
    Just modul -> foldM applyRenamingBlock modul renamings

-- TODO: cleanup duplicates, make a set of UDecl instead of a list?
-- TODO: finish renamings
-- TODO: add annotations to renamings, work with something better than just
--       name.
-- TODO: improve error diagnostics
applyRenamingBlock :: ModuleScope -> URenamingBlock PhParse -> Except Err ModuleScope
applyRenamingBlock modul renamingBlock@(Ann src (URenamingBlock renamings)) = do
  let renamingMap = M.fromList renamings
      (sources, targets) = (L.map fst renamings, L.map snd renamings)
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
                                    renamings
      -- TODO: cleanup
      r' = zip occurOnBothSides
                    [GenName ("gen#" ++ show i) | i <- [1..] :: [Int]]
      r'' = unambiguousRenamings <>
              [(freeName, fromJust $ M.lookup source renamingMap) |
               (source, freeName) <- r']
  (modul', renamings') <- (
      if M.size renamingMap /= L.length renamings
      then checkError src "Duplicate key in renaming block."
      else if not (null unknownSources)
      then checkError src "Renaming block has unknown sources."
      else if not (null occurOnBothSides)
      then applyRenamingBlock modul (URenamingBlock r' <$$ renamingBlock)
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
applyRenaming :: UDecl PhCheck -> Renaming -> UDecl PhCheck
applyRenaming annDecl renaming = applyRenamingInDecl <$$> annDecl
  where replaceName' = flip replaceName renaming
        --applyRenamingInDecl :: UDecl p -> UDecl p
        applyRenamingInDecl decl
          | UType name <- decl = UType $ replaceName' name
          | UCallable ctyp name vars retType expr <- decl =
                UCallable ctyp (replaceName' name)
                  (map (applyRenamingInVar <$$>) vars) (replaceName' retType)
                  (applyRenamingInExpr <$> expr)
        applyRenamingInVar (Var mode name typ) =
          Var mode name (replaceName' <$> typ)
        applyRenamingInExpr = (applyRenamingInExpr' <$$>)
        applyRenamingInExpr' :: UExpr' p -> UExpr' p
        applyRenamingInExpr' expr
          | UVar var <- expr = UVar $ applyRenamingInVar <$$> var
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

getArgType :: UVar a -> UType
getArgType ~(Ann _ (Var _ _ (Just typ))) = typ

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
  UCall (FuncName _) args _ -> or $ map isStateful args
  UCall (ProcName _) _ _ -> True
  UBlockExpr stmts -> or $ NE.map isStateful stmts
  ULet {} -> True
  UIf cond bTrue bFalse -> or $ map isStateful [cond, bTrue, bFalse]
  UAssert cond -> isStateful cond
  USkip -> False
