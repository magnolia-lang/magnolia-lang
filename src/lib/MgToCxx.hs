{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module MgToCxx (
    mgPackageToCxxSelfContainedProgramPackage
  )
  where

import Control.Monad (foldM, join, unless)
--import Control.Monad.IO.Class (liftIO)
import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isJust)
import qualified Data.Set as S
import Data.Void (absurd)

import Cxx.Syntax
import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util

-- | Not really a map, but a wrapper over 'foldMAccumErrors' that allows writing
-- code like
--
-- >>> mapM f t
--
-- on lists that accumulates errors within the 'MgMonad' and returns a list of
-- items.
mapMAccumErrors :: (a -> MgMonad b) -> [a] -> MgMonad [b]
mapMAccumErrors f = foldMAccumErrors (\acc a -> (acc <>) . (:[]) <$> f a) []

-- TODOs left:
-- - handle properly external required types
-- - better documentation
-- - produce good design documentation
-- - write examples to test more
-- - perhaps find a better way to deal with bound names and fresh name
--   generation, e.g. by carrying a HashSet of names within our context
-- - deal with guards
-- - test extensively new name generation; what happens when a function
--   overloaded on return type is mutified, and a procedure with that
--   prototype and the same name exists? We need to make sure the function
--   is properly renamed.

-- | Transforms a Magnolia typechecked package into a C++ self-contained
-- package. Self-contained here means that a C++ module (a struct or class) is
-- generated for each program defined in the Magnolia package. No code is
-- generated for other types of modules, and the same function exposed in two
-- different programs is code generated twice – once for each program.
mgPackageToCxxSelfContainedProgramPackage :: TcPackage -> MgMonad CxxPackage
mgPackageToCxxSelfContainedProgramPackage tcPkg =
  enter (nodeName tcPkg) $ do
    let allModules = join $
          map (getModules . snd) $ M.toList (_packageDecls $ _elem tcPkg)
        cxxPkgName = nodeName tcPkg
    cxxIncludes <- (mkCxxSystemInclude "cassert":) <$>
      gatherCxxIncludes allModules
    cxxPrograms <- gatherPrograms allModules
    return $ CxxPackage cxxPkgName cxxIncludes cxxPrograms
  where
    moduleType :: TcModule -> MModuleType
    moduleType (Ann _ (MModule moduleTy _ _ _)) = moduleTy
    moduleType (Ann _ (RefModule _ _ v)) = absurd v

    gatherCxxIncludes :: [TcModule] -> MgMonad [CxxInclude]
    gatherCxxIncludes = foldMAccumErrors (\acc tcM -> case moduleType tcM of
      External Cxx structName -> case _scopeName structName of
        Nothing -> throwLocatedE MiscErr (srcCtx $ _ann tcM) $
          "external C++ block " <> pshow structName <> " was specified " <>
          "without an include path"
        Just pathName -> return $ mkCxxRelativeIncludeFromName pathName : acc
      _ -> return acc) []

    gatherPrograms :: [TcModule] -> MgMonad [CxxModule]
    gatherPrograms = foldM
      (\acc -> ((:acc) <$>) . mgProgramToCxxProgramModule (nodeName tcPkg)) [] .
      filter ((== Program) . moduleType)

mgProgramToCxxProgramModule :: Name -> TcModule -> MgMonad CxxModule
mgProgramToCxxProgramModule _ (Ann _ (RefModule _ _ v)) = absurd v
mgProgramToCxxProgramModule pkgName
                            (Ann _ (MModule Program name decls deps)) =
  enter name $ do
    let moduleFqName = FullyQualifiedName (Just pkgName) name
        boundNames = S.fromList . map _name $
          name : M.keys decls <> map nodeName deps

        declsToGen = filter declFilter declsAsList
        typeDecls = getTypeDecls declsToGen

        (boundNames', returnTypeOverloadsNameAliasMap) =
          foldl (\(boundNs, nameMap) rTyO ->
                let (boundNs', newName) = registerFreshName boundNs (fst rTyO)
                in (boundNs', M.insert rTyO newName nameMap))
            (boundNames, M.empty) (S.toList returnTypeOverloads)

        (boundNames'', extObjectsNameMap) =
          foldl (\(boundNs, nameMap) inName ->
                let inName' = inName { _name = "__" <> _name inName }
                    (boundNs', newName) = registerFreshName boundNs inName'
                in (boundNs', M.insert inName newName nameMap))
            (boundNames', M.empty) referencedExternals

        callableDecls = getCallableDecls declsToGen

    moduleCxxNamespaces <- mkCxxNamespaces moduleFqName
    moduleCxxName <- mkCxxName name

    extObjectsCxxNameMap <- mapM mkCxxName extObjectsNameMap
    extObjectsDef <- mapMAccumErrors (\(structName, objCxxName) ->
          mkCxxType structName
        >>= \cxxTy -> return (CxxExternalInstance cxxTy objCxxName))
      (M.toList extObjectsCxxNameMap)

    cxxTyDefs <- mapMAccumErrors mgTyDeclToCxxTypeDef typeDecls

    cxxFnDefs <- mapMAccumErrors
      (mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsCxxNameMap)
      callableDecls

    defs <- ((map (CxxPrivate,) extObjectsDef <>) <$>) $
      (map (CxxPublic,) <$>) $
        ((cxxTyDefs <> cxxFnDefs) <>) <$>
          mgReturnTypeOverloadsToCxxTemplatedDefs boundNames''
            returnTypeOverloadsNameAliasMap
    return $ CxxModule moduleCxxNamespaces moduleCxxName $ L.sortOn snd defs
  where
    declFilter :: TcDecl -> Bool
    declFilter decl = case decl of
      MTypeDecl _ -> True
      MCallableDecl (Ann _ callable) ->
        -- Only accept callables if they:
        -- (1) are not called "_=_" (assignment is always pre-generated)
        -- (2) do not contain predicates as parameters (the only methods that
        --     allow this are pre-generated predicate functions)
        -- This is because we assume all of these to have a canonical
        -- external implementation.
        _name (nodeName callable) /= "_=_" &&
        all ((/= Pred) . _varType . _elem) (_callableArgs callable)

    returnTypeOverloads :: S.Set (Name, [MType])
    returnTypeOverloads =
      let extractDuplicateTypeLists =
            map L.head . filter (not . null . tail) . L.group . L.sort
          getInputTypeLists callables =
            map (map (_varType . _elem) . _callableArgs . _elem) callables
          extractOverloadsOnReturnType callablesName =
            map (callablesName,) . extractDuplicateTypeLists . getInputTypeLists
      in M.foldrWithKey (\k v acc ->
            acc `S.union` S.fromList (extractOverloadsOnReturnType k v)
        ) S.empty $ M.map getCallableDecls decls

    declsAsList :: [TcDecl]
    declsAsList = join (map snd (M.toList decls))

    referencedExternals :: [Name]
    referencedExternals = S.toList $ foldl accExtFqn S.empty declsAsList

    accExtFqn :: S.Set Name -> TcDecl -> S.Set Name
    accExtFqn acc (MTypeDecl (Ann (mconDeclO, _) _)) = case mconDeclO of
      Just (ConcreteExternalDecl _ _ fqn) ->
        let ~(Just scopeName) = _scopeName fqn in S.insert scopeName acc
      _ -> acc
    accExtFqn acc (MCallableDecl (Ann (mconDeclO, _) _)) = case mconDeclO of
      Just (ConcreteExternalDecl _ _ fqn) ->
        let ~(Just scopeName) = _scopeName fqn in S.insert scopeName acc
      _ -> acc

mgProgramToCxxProgramModule _ _ = error "expected program"

-- | Produces a fresh name based on an initial input name and a set of bound
-- strings. If the input name's String component is not in the set of bound
-- strings, it is returned as is.
freshName :: Name -> S.Set String -> Name
freshName name@(Name ns str) boundNs
  | str `S.member` boundNs = freshName' (0 :: Int)
  | otherwise = name
  where
    freshName' i | (str <> show i) `S.member` boundNs = freshName' (i + 1)
                 | otherwise = Name ns $ str <> show i

-- | Produces a fresh name and returns both an updated set of bound strings that
-- includes it, and the new name.
registerFreshName :: S.Set String -> Name -> (S.Set String, Name)
registerFreshName boundNs name = let newName = freshName name boundNs
                                 in (S.insert (_name newName) boundNs, newName)

mgTyDeclToCxxTypeDef :: TcTypeDecl -> MgMonad CxxDef
mgTyDeclToCxxTypeDef (Ann (conDeclO, absDeclOs) (Type targetTyName _)) = do
  cxxTargetTyName <- mkCxxName targetTyName
  let ~(Just (ConcreteExternalDecl _ backend extFqn)) = conDeclO
      ~(Just extStructName) = _scopeName extFqn
  unless (backend == Cxx) $
    throwLocatedE MiscErr (srcCtx $ NE.head absDeclOs) $
      "attempted to generate C++ code relying on external type " <>
      pshow extFqn <> " but it is declared as having a " <> pshow backend <>
      " backend"
  cxxSourceTyName <-
    mkCxxNamespaceMemberAccess <$> mkCxxName extStructName
                               <*> mkCxxName (_targetName extFqn)
  return $ CxxTypeDef cxxSourceTyName cxxTargetTyName


mgCallableDeclToCxx
  :: M.Map (Name, [MType]) Name
  -> M.Map Name CxxName
  -> TcCallableDecl
  -> MgMonad CxxDef
mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsMap
  mgFn@(Ann (conDeclO, absDeclOs) (Callable _ _ args retTy _ ExternalBody)) = do
    -- This case is hit when one of the API functions we want to expose
    -- is declared externally, and potentially renamed.
    -- In this case, to generate the API function, we need to perform
    -- an inline call to the external function.
    let ~(Just (ConcreteExternalDecl _ backend extFqn)) = conDeclO
        ~(Just extStructName) = _scopeName extFqn
    unless (backend == Cxx) $
      throwLocatedE MiscErr (srcCtx $ NE.head absDeclOs) $
        "attempted to generate C++ code relying on external function " <>
        pshow extFqn <> " but it is declared as having a " <>
        pshow backend <> " backend"
    let mgFnWithDummyMgBody = Ann (conDeclO, absDeclOs) $
          (_elem mgFn) { _callableBody = MagnoliaBody (Ann retTy MSkip) }
        ~(Just cxxExtObjectName) = M.lookup extStructName extObjectsMap

    ~(CxxFunctionDef cxxFnDef) <- mgCallableDeclToCxx
      returnTypeOverloadsNameAliasMap extObjectsMap mgFnWithDummyMgBody
    -- We synthesize a function call
    cxxExtFnName <- mkCxxObjectMemberAccess cxxExtObjectName <$>
      mkCxxName (_targetName extFqn)
    cxxArgExprs <- mapM ((CxxVarRef <$>) . mkCxxName . nodeName) args
    let cxxBody = [ CxxStmtInline . CxxReturn . Just $
                      CxxCall cxxExtFnName [] cxxArgExprs
                  ]
    return $ CxxFunctionDef cxxFnDef {_cxxFnBody = cxxBody }


mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsMap
                    mgFn@(Ann _ (Callable cty name args retTy mguard cbody))
  | isOverloadedOnReturnType = do
      mgFn' <- mutify mgFn
      mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsMap mgFn'
  | otherwise = do
      cxxFnName <- mkCxxName name
      cxxBody <- mgFnBodyToCxxStmtBlock returnTypeOverloadsNameAliasMap mgFn
      cxxRetTy <- case cty of
        Function -> mkClassMemberCxxType retTy
        Predicate -> mkCxxType Pred
        _ -> mkCxxType Unit -- Axiom and procedures
      cxxParams <- mapM mgTypedVarToCxx args
      -- TODO: what do we do with guard? Carry it in and use it as a
      -- precondition test?
      return . CxxFunctionDef $
        CxxFunction False cxxFnName [] cxxParams cxxRetTy
                    cxxBody
  where
    mgTypedVarToCxx :: TypedVar PhCheck -> MgMonad CxxVar
    mgTypedVarToCxx (Ann _ v) = do
      cxxVarName <- mkCxxName $ _varName v
      cxxVarType <- mkClassMemberCxxType $ _varType v
      return $ CxxVar (_varMode v == MObs) True cxxVarName cxxVarType

    isOverloadedOnReturnType :: Bool
    isOverloadedOnReturnType = isJust $
      M.lookup (name, map (_varType . _elem) args)
               returnTypeOverloadsNameAliasMap

    mutifiedFnName :: Name
    mutifiedFnName =
      let ~(Just mutFnName) = M.lookup (name, map (_varType . _elem) args)
                                       returnTypeOverloadsNameAliasMap
      in mutFnName

    mutify :: TcCallableDecl -> MgMonad TcCallableDecl
    mutify (Ann ann c) = mutifyBody >>= \mutifiedBody -> return $ Ann ann
      c { _callableType = Procedure
        , _callableName = mutifiedFnName
        , _callableArgs = args <> [Ann retTy mutifiedOutArg]
        , _callableReturnType = Unit
        , _callableGuard = mguard
        , _callableBody = mutifiedBody
        }

    mutifiedOutName =
      freshName (VarName "o") (S.fromList $ map (_name . _varName . _elem) args)

    mutifiedOutArg = Var MOut mutifiedOutName retTy
    mutifiedOutVar = Var MOut mutifiedOutName (Just retTy)

    mutifyBody :: MgMonad (CBody PhCheck)
    mutifyBody = case cbody of
      MagnoliaBody e -> return . MagnoliaBody . Ann Unit $
        MCall (ProcName "_=_")
              [Ann retTy (MVar (Ann retTy mutifiedOutVar)), e]
              (Just retTy)
      _ -> return cbody

-- | Generates a templated function for each set of functions overloaded on
-- their return types.
mgReturnTypeOverloadsToCxxTemplatedDefs
  :: S.Set String -- Not sure if necessary to pay attention to bound names here.
  -> M.Map (Name, [MType]) Name
  -> MgMonad [CxxDef]
mgReturnTypeOverloadsToCxxTemplatedDefs boundNs = mapM mkOverloadedFn . M.toList
  where
    mkOverloadedFn ((fnName, argTypes), mutifiedFnName) = do
      argCxxNames <- mapM mkCxxName $ snd $
        L.mapAccumL registerFreshName boundNs
                    (map (const (VarName "a")) argTypes)
      cxxArgTypes <- mapM mkClassMemberCxxType argTypes
      let templateTyName = freshName (TypeName "T") boundNs
      cxxTemplateTyName <- mkCxxName templateTyName
      cxxTemplateTy <- mkCxxType templateTyName
      outVarCxxName <- mkCxxName $ freshName (VarName "o") boundNs
      cxxFnName <- mkCxxName fnName
      mutifiedFnCxxName <- mkCxxName mutifiedFnName
      let cxxOutVar = CxxVar { _cxxVarIsConst = False
                             , _cxxVarIsRef = False
                             , _cxxVarName = outVarCxxName
                             , _cxxVarType = cxxTemplateTy
                             }
          cxxFnParams = zipWith (\cxxArgName cxxArgTy ->
            CxxVar { _cxxVarIsConst = True
                   , _cxxVarIsRef = True
                   , _cxxVarName = cxxArgName
                   , _cxxVarType = cxxArgTy
                   } ) argCxxNames cxxArgTypes
          cxxCallArgExprs = map CxxVarRef (argCxxNames <> [outVarCxxName])
          cxxFnBody = map CxxStmtInline
            [ -- T o;
              CxxVarDecl cxxOutVar Nothing
              -- mutifiedFnName(a0, a1, …, an, &o);
            , CxxExpr (CxxCall mutifiedFnCxxName []
                               cxxCallArgExprs)
              -- return o;
            , CxxReturn (Just (CxxVarRef outVarCxxName))
            ]
      return . CxxFunctionDef $
        CxxFunction { _cxxFnIsInline = True
                    , _cxxFnName = cxxFnName
                    , _cxxFnTemplateParameters = [cxxTemplateTyName]
                    , _cxxFnParams = cxxFnParams
                    , _cxxFnReturnType = cxxTemplateTy
                    , _cxxFnBody = cxxFnBody
                    }


mgFnBodyToCxxStmtBlock :: M.Map (Name, [MType]) Name -> TcCallableDecl
                       -> MgMonad CxxStmtBlock
mgFnBodyToCxxStmtBlock returnTypeOverloadsNameAliasMap
                       (Ann _ (Callable _ name _ _ _ body)) = case body of
  EmptyBody -> throwNonLocatedE CompilerErr $
    "attempted to generate implementation code for unimplemented callable " <>
    pshow name
  ExternalBody -> throwNonLocatedE CompilerErr $
    "attempted to generate implementation code for external callable " <>
    pshow name
  -- TODO: this is not extremely pretty. Can we easily do something better?
  MagnoliaBody (Ann _ (MBlockExpr _ exprs)) ->
    mapM (mgExprToCxxStmt returnTypeOverloadsNameAliasMap) (NE.toList exprs)
  MagnoliaBody expr -> (:[]) <$>
    mgExprToCxxStmt returnTypeOverloadsNameAliasMap (insertValueBlocks expr)

insertValueBlocks :: TcExpr -> TcExpr
insertValueBlocks inExpr@(Ann Unit _) = inExpr
insertValueBlocks inExpr@(Ann ty e) = case e of
  MValue _ -> inExpr
  MBlockExpr _ _ -> inExpr
  MIf cond trueExpr falseExpr -> Ann ty $
    MIf cond (insertValueBlocks trueExpr) (insertValueBlocks falseExpr)
  _ -> Ann ty $ MValue inExpr

mgExprToCxxStmt :: M.Map (Name, [MType]) Name -> TcExpr -> MgMonad CxxStmt
mgExprToCxxStmt returnTypeOverloadsNameAliasMap = goStmt
  where
    goStmt annInExpr@(Ann _ inExpr) = case inExpr of
      MVar _ -> CxxStmtInline . CxxExpr <$> goExpr annInExpr
      -- TODO: handle special case of assignment in a prettier way. ATM,
      --       this seems good enough.
      MCall (ProcName "_=_") [Ann tyLhs (MVar v), rhs@(Ann tyRhs _)] _ ->
        if tyLhs == tyRhs then do
          cxxVarName <- mkCxxName . _varName $ _elem v
          cxxRhs <- goExpr rhs
          return . CxxStmtInline $ CxxAssign cxxVarName cxxRhs
        else CxxStmtInline . CxxExpr <$> goExpr annInExpr
      MCall {} -> CxxStmtInline . CxxExpr <$> goExpr annInExpr
      MBlockExpr _ stmts -> CxxStmtBlock <$> mapM goStmt (NE.toList stmts)
      MValue _ -> CxxStmtInline . CxxReturn . Just <$> goExpr annInExpr
      MLet (Ann vty (Var mode name _)) mexpr -> do
        cxxVarName <- mkCxxName name
        cxxVarType <- mkClassMemberCxxType vty
        mcxxRhsExpr <- maybe (return Nothing) ((Just <$>) . goExpr) mexpr
        let cxxVarIsConst = mode == MObs
            cxxVarIsRef = False -- TODO: figure out when to use refs?
            cxxVar = CxxVar cxxVarIsConst cxxVarIsRef cxxVarName cxxVarType
        return . CxxStmtInline $ CxxVarDecl cxxVar mcxxRhsExpr
      MIf cond trueStmt falseStmt -> do
        cxxCond <- goExpr cond
        cxxTrueStmt <- goStmt trueStmt
        cxxFalseStmt <- goStmt falseStmt
        return . CxxStmtInline $ CxxIf cxxCond cxxTrueStmt cxxFalseStmt
      MAssert expr -> CxxStmtInline . CxxAssert <$> goExpr expr
      MSkip -> return $ CxxStmtInline CxxSkip
    goExpr = mgExprToCxxExpr returnTypeOverloadsNameAliasMap

mgExprToCxxExpr :: M.Map (Name, [MType]) Name -> TcExpr -> MgMonad CxxExpr
mgExprToCxxExpr returnTypeOverloadsNameAliasMap = goExpr
  where
    goExpr annInExpr@(Ann ty inExpr) = case inExpr of
      MVar (Ann _ v) -> CxxVarRef <$> mkCxxName (_varName v)
      MCall name args _ -> do
        parentModuleName <- getParentModuleName
        cxxModuleName <- mkCxxName parentModuleName
        cxxName <- mkCxxClassMemberAccess cxxModuleName <$> mkCxxName name
        let inputProto = (name, map _ann args)
        cxxTemplateArgs <- mapM mkCxxName
          [ty | inputProto `M.member` returnTypeOverloadsNameAliasMap]
        cxxArgs <- mapM goExpr args
        return $ CxxCall cxxName cxxTemplateArgs cxxArgs
      MBlockExpr blockTy exprs -> do
        cxxExprs <- mapM goStmt (NE.toList exprs)
        case blockTy of
          MValueBlock -> mgToCxxLambdaVal cxxExprs
          MEffectfulBlock -> mgToCxxLambdaRef cxxExprs
      MValue expr -> goExpr expr
      MLet {} -> goStmt annInExpr >>= mgToCxxLambdaVal . (:[])
      MIf cond trueExpr falseExpr -> CxxIfExpr <$> goExpr cond <*>
        goExpr trueExpr <*> goExpr falseExpr
      MAssert _ -> goStmt annInExpr >>= mgToCxxLambdaVal . (:[])
      MSkip -> mgToCxxLambdaRef []

    goStmt = mgExprToCxxStmt returnTypeOverloadsNameAliasMap
    mgToCxxLambdaRef = return . CxxLambdaCall CxxLambdaCaptureDefaultReference
    mgToCxxLambdaVal = return . CxxLambdaCall CxxLambdaCaptureDefaultValue

mkCxxNamespaces :: FullyQualifiedName -> MgMonad [CxxNamespaceName]
mkCxxNamespaces fqName = maybe (return []) split (_scopeName fqName)
  where split (Name namespace nameStr) = case break (== '.') nameStr of
          ("", "") -> return []
          (ns, "") -> (:[]) <$> mkCxxName (Name namespace ns)
          (ns, _:rest) -> (:) <$> mkCxxName (Name namespace ns)
                              <*> split (Name namespace rest)