{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module MgToCxx (
    mgPackageToCxxSelfContainedProgramPackage
  )
  where

import Control.Monad.State
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
import Monad

-- | A helper data type to wrap the requirements found in 'ExternalDeclDetails'.
data Requirement = Requirement { -- | The original required declaration
                                 _requiredDecl :: TcDecl
                                 -- | The declaration supposed to fullfill
                                 --   the requirement.
                               , _parameterDecl :: TcDecl
                               }
                   deriving (Eq, Ord)

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
        cxxIncludes = mkCxxSystemInclude "cassert" :
          gatherCxxIncludes allModules
    cxxPrograms <- gatherPrograms allModules
    return $ CxxPackage cxxPkgName cxxIncludes cxxPrograms
  where
    moduleType :: TcModule -> MModuleType
    moduleType (Ann _ (MModule moduleTy _ _)) = moduleTy

    gatherCxxIncludes :: [TcModule] -> [CxxInclude]
    gatherCxxIncludes tcModules = S.toList $ foldl
      (\acc (Ann _ (MModule _ _ tcModuleExpr)) ->
        let newCxxIncludes = extractIncludeFromModuleExpr tcModuleExpr
        in acc `S.union` newCxxIncludes) S.empty tcModules

    extractIncludeFromModuleExpr :: TcModuleExpr -> S.Set CxxInclude
    extractIncludeFromModuleExpr (Ann _ moduleExpr) = case moduleExpr of
      MModuleRef v _ -> absurd v
      MModuleAsSignature v _ -> absurd v
      MModuleExternal _ _ v -> absurd v
      MModuleDef decls _ _ -> do
        let innerFoldFn acc tcDecl =
              maybe acc (`S.insert` acc) (extractExternalReference tcDecl)
        foldl (foldl innerFoldFn) S.empty decls

    -- TODO(bchetioui, 2021/09/02): this does not work if there exists a fully
    -- parameterized external module (i.e. one that does not contain any
    -- concrete external definition). This edge case will be properly handled
    -- once some tweaks are made on {Concrete,Abstract}DeclOrigin annotations
    -- in a WIP change.
    extractExternalReference :: TcDecl -> Maybe CxxInclude
    extractExternalReference tcDecl = case tcDecl of
      MTypeDecl _ (Ann (mconDeclO, _) _) ->
        mconDeclO >>= mkCxxIncludeFromConDeclO
      MCallableDecl _ (Ann (mconDeclO, _) _) ->
        mconDeclO >>= mkCxxIncludeFromConDeclO

    mkCxxIncludeFromConDeclO :: ConcreteDeclOrigin -> Maybe CxxInclude
    mkCxxIncludeFromConDeclO (ConcreteExternalDecl _ extDeclDetails) = Just $
      mkCxxRelativeCxxIncludeFromPath (externalDeclFilePath extDeclDetails)
    mkCxxIncludeFromConDeclO _ = Nothing

    gatherPrograms :: [TcModule] -> MgMonad [CxxModule]
    gatherPrograms = foldM
      (\acc -> ((:acc) <$>) . mgProgramToCxxProgramModule (nodeName tcPkg)) [] .
      filter ((== Program) . moduleType)

mgProgramToCxxProgramModule :: Name -> TcModule -> MgMonad CxxModule
mgProgramToCxxProgramModule
  pkgName (Ann _ (MModule Program name (Ann _ (MModuleDef decls deps _)))) =
  enter name $ do
    let moduleFqName = FullyQualifiedName (Just pkgName) name
        boundNames = S.fromList . map _name $
          name : M.keys decls <> map nodeName deps

        declsToGen = filter declFilter declsAsList
        typeDecls = getTypeDecls declsToGen
        callableDecls = getCallableDecls declsToGen

        -- TODO: join boundNames handling into one op returning everything
        (boundNames', returnTypeOverloadsNameAliasMap) =
          foldl (\(boundNs, nameMap) rTyO ->
                let (boundNs', newName) = registerFreshName boundNs (fst rTyO)
                in (boundNs', M.insert rTyO newName nameMap))
            (boundNames, M.empty) (S.toList returnTypeOverloads)

        (extObjectsNameMap, boundNames'') =
          let (extObjectNames, boundNs) = mapM (freshExtObjectName . fst)
                referencedExternals `runState` boundNames'
          in (M.fromList $ zip referencedExternals extObjectNames, boundNs)

        -- We generate a number of empty structs in order to parameterize
        -- external modules with requirements so as to extract the types they
        -- define. This should be a valid way to do things in C++ thanks to
        -- SFINAE.
        (dummyStructNames, boundNames''') =
          let nbDummyStructs = safeMaximum 0 $
                map (length . snd) referencedExternals
          in runState (mapM (\_ -> freshNameM (TypeName "dummy_struct"))
                            [1..nbDummyStructs]) boundNames''

    moduleCxxNamespaces <- mkCxxNamespaces moduleFqName
    moduleCxxName <- mkCxxName name

    extObjectsCxxNameMap <- mapM mkCxxName extObjectsNameMap
    extObjectsDef <- mapMAccumErrors
      ((CxxExternalInstance <$>) . uncurry (uncurry mkCxxObject))
      (M.toList extObjectsCxxNameMap)

    dummyStructCxxNames <- mapM mkCxxName dummyStructNames

    cxxTyDefs <-
      mapMAccumErrors (mgTyDeclToCxxTypeDef dummyStructCxxNames) typeDecls

    cxxFnDefs <- mapMAccumErrors
      (mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsCxxNameMap)
      callableDecls

    let cxxDummyStructDefs = map CxxDummyModule dummyStructCxxNames

    defs <- ((map (CxxPrivate,) (extObjectsDef <> cxxDummyStructDefs) <>) <$>) $
      (map (CxxPublic,) <$>) $
        ((cxxTyDefs <> cxxFnDefs) <>) <$>
          mgReturnTypeOverloadsToCxxTemplatedDefs boundNames'''
            returnTypeOverloadsNameAliasMap
    return $ CxxModule moduleCxxNamespaces moduleCxxName $ L.sortOn snd defs
  where
    safeMaximum :: Ord a => a -> [a] -> a
    safeMaximum defaultValue l = maximum $ defaultValue : l

    declFilter :: TcDecl -> Bool
    declFilter decl = case decl of
      MTypeDecl _ _ -> True
      -- Only generate callables that are not assumed built-in.
      MCallableDecl _ (Ann _ callable) -> not $ isBuiltin callable

    isBuiltin :: MCallableDecl' p -> Bool
    isBuiltin cdecl = case _callableBody cdecl of BuiltinBody -> True
                                                  _ -> False

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
    declsAsList = join $ M.elems decls

    referencedExternals :: [(Name, [Requirement])]
    referencedExternals = S.toList $ foldl accExtReqs S.empty declsAsList

    -- TODO: extract ExternalDeclDetails module name, requirements, and
    accExtReqs :: S.Set (Name, [Requirement]) -> TcDecl
               -> S.Set (Name, [Requirement])
    accExtReqs acc (MTypeDecl _ (Ann (mconDeclO, _) _)) = case mconDeclO of
      Just (ConcreteExternalDecl _ extDeclDetails) ->
        S.insert ( externalDeclModuleName extDeclDetails
                 , map (uncurry Requirement) $
                    M.toList $ externalDeclRequirements extDeclDetails
                 ) acc
      _ -> acc
    accExtReqs acc (MCallableDecl _ (Ann (mconDeclO, _) _)) = case mconDeclO of
      Just (ConcreteExternalDecl _ extDeclDetails) ->
        S.insert ( externalDeclModuleName extDeclDetails
                 , map (uncurry Requirement) $
                    M.toList $ externalDeclRequirements extDeclDetails
                 ) acc
      _ -> acc

    freshNameM :: Name -> State (S.Set String) Name
    freshNameM n = do
      env <- get
      let freeName = freshName n env
      modify (S.insert $ _name freeName)
      pure freeName

    freshExtObjectName :: Name -> State (S.Set String) Name
    freshExtObjectName n = freshNameM n { _name = "__" <> _name n }


mgProgramToCxxProgramModule _ _ = error "expected program"

-- | Produces a C++ object from a data structure to instantiate along with
-- the required template parameters if necessary.
mkCxxObject :: Name          -- ^ the data structure to instantiate
            -> [Requirement] -- ^ the required arguments to the
                             --   data structure
            -> CxxName       -- ^ the name given to the resulting
                             --   object
            -> MgMonad CxxObject
mkCxxObject structName [] targetCxxName = do
  structCxxName <- mkCxxName structName
  pure $ CxxObject (CxxCustomType structCxxName) targetCxxName
mkCxxObject structName requirements@(_:_) targetCxxName = do
  let sortedRequirements = L.sortBy orderRequirements requirements
  structCxxName <- mkCxxName structName
  cxxTemplateParameters <- mapM (mkCxxName . nodeName . _parameterDecl)
                                sortedRequirements
  pure $ CxxObject (CxxCustomTemplatedType structCxxName cxxTemplateParameters)
                   targetCxxName

-- | Defines an ordering on 'Requirement'. The convention is that type
-- declarations are lower than callable declarations, and elements of the
-- same type are ordered lexicographically on their names.
orderRequirements :: Requirement -> Requirement -> Ordering
orderRequirements (Requirement (MTypeDecl _ tcTy1) _)
                  (Requirement tcDecl2 _) = case tcDecl2 of
  MTypeDecl _ tcTy2 -> nodeName tcTy1 `compare` nodeName tcTy2
  MCallableDecl _ _ -> LT
orderRequirements (Requirement (MCallableDecl _ tcCallable1) _)
                  (Requirement tcDecl2 _) = case tcDecl2 of
  MTypeDecl _ _ -> GT
  MCallableDecl _ tcCallable2 ->
    nodeName tcCallable1 `compare` nodeName tcCallable2

-- | Produces a list of ordered requirements from an 'ExternalDeclDetails'.
-- See 'orderRequirements' for details on the ordering.
orderedRequirements :: ExternalDeclDetails -> [Requirement]
orderedRequirements extDeclDetails = L.sortBy orderRequirements $
  map (uncurry Requirement) $ M.toList $ externalDeclRequirements extDeclDetails

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

-- | Produces a C++ typedef from a Magnolia type declaration. A list of dummy
-- structures is provided as a parameter, and can be used to parameterize the
-- structure from which to extract the source type if it has requirements.
mgTyDeclToCxxTypeDef :: [CxxName]  -- ^ the names of the dummy data structures
                     -> TcTypeDecl
                     -> MgMonad CxxDef
mgTyDeclToCxxTypeDef dummyStructCxxNames
                     (Ann (conDeclO, absDeclOs) (Type targetTyName)) = do
  cxxTargetTyName <- mkCxxName targetTyName
  let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
      backend = externalDeclBackend extDeclDetails
      extStructName = externalDeclModuleName extDeclDetails
      extTyName = externalDeclElementName extDeclDetails
      extTyNbRequirements = M.size $ externalDeclRequirements extDeclDetails
  unless (backend == Cxx) $
    throwLocatedE MiscErr (srcCtx $ NE.head absDeclOs) $
      "attempted to generate C++ code relying on external type " <>
      pshow (FullyQualifiedName (Just extStructName) extTyName) <> " but " <>
      "it is declared as having a " <> pshow backend <> " backend"
  cxxExtStructName <- mkCxxName extStructName
  cxxExtTyName <- mkCxxName extTyName
  let cxxSourceTyName = case extTyNbRequirements of
        0 ->
          mkCxxClassMemberAccess (CxxCustomType cxxExtStructName) cxxExtTyName
        _ -> let paramStructs = zipWith const dummyStructCxxNames
                  [1..extTyNbRequirements]
             in mkCxxClassMemberAccess
              (CxxCustomTemplatedType cxxExtStructName paramStructs)
              cxxExtTyName
  return $ CxxTypeDef cxxSourceTyName cxxTargetTyName


-- TODO: extObjectsMap here should also contain the requirements, and easy!
mgCallableDeclToCxx
  :: M.Map (Name, [MType]) Name
  -> M.Map (Name, [Requirement]) CxxName
  -> TcCallableDecl
  -> MgMonad CxxDef
mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsMap
  mgFn@(Ann (conDeclO, absDeclOs)
            (Callable _ _ args retTy _ (ExternalBody _))) = do
    -- This case is hit when one of the API functions we want to expose
    -- is declared externally, and potentially renamed.
    -- In this case, to generate the API function, we need to perform
    -- an inline call to the external function.
    let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
        backend = externalDeclBackend extDeclDetails
        extStructName = externalDeclModuleName extDeclDetails
        extCallableName = externalDeclElementName extDeclDetails
        extOrderedRequirements = orderedRequirements extDeclDetails
    unless (backend == Cxx) $
      throwLocatedE MiscErr (srcCtx $ NE.head absDeclOs) $
        "attempted to generate C++ code relying on external function " <>
        pshow (FullyQualifiedName (Just extStructName) extCallableName) <>
        " but it is declared as having a " <> pshow backend <> " backend"
    let mgFnWithDummyMgBody = Ann (conDeclO, absDeclOs) $
          (_elem mgFn) { _callableBody = MagnoliaBody (Ann retTy MSkip) }
        ~(Just cxxExtObjectName) =
          M.lookup (extStructName, extOrderedRequirements) extObjectsMap

    ~(CxxFunctionDef cxxFnDef) <- mgCallableDeclToCxx
      returnTypeOverloadsNameAliasMap extObjectsMap mgFnWithDummyMgBody
    -- We synthesize a function call
    cxxExtFnName <- mkCxxObjectMemberAccess cxxExtObjectName <$>
      mkCxxName extCallableName
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
  ExternalBody () -> throwNonLocatedE CompilerErr $
    "attempted to generate implementation code for external callable " <>
    pshow name
  BuiltinBody -> throwNonLocatedE CompilerErr $
    "attempted to generate implementation code for builtin callable " <>
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
        mCxxExpr <- tryMgCallToCxxSpecialOpExpr returnTypeOverloadsNameAliasMap
          name args ty
        case mCxxExpr of
          Just cxxExpr -> return cxxExpr
          Nothing -> do
            parentModuleName <- getParentModuleName
            cxxModuleName <- mkCxxName parentModuleName
            cxxName <- mkCxxClassMemberAccess (CxxCustomType cxxModuleName) <$>
              mkCxxName name
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

-- TODO: for the moment, this assumes no '_:T ==_:T' predicate is implemented
-- in Magnolia, although it will be possible to define one manually. Therefore,
-- this will have to be improved later.
-- | Takes the name of a function, its arguments and its return type, and
-- produces a unop or binop expression node if it can be expressed as such in
-- C++. For the moment, functions that can be expressed like that include the
-- equality predicate between two elements of the same type, predicate
-- combinators (such as '_&&_' and '_||_'), and the boolean constants 'TRUE',
-- and 'FALSE'.
tryMgCallToCxxSpecialOpExpr :: M.Map (Name, [MType]) Name
                            -> Name -> [TcExpr] -> MType
                            -> MgMonad (Maybe CxxExpr)
tryMgCallToCxxSpecialOpExpr returnTypeOverloadsNameAliasMap name args retTy = do
  cxxArgs <- mapM (mgExprToCxxExpr returnTypeOverloadsNameAliasMap) args
  case (cxxArgs, retTy : map _ann args ) of
    ([cxxExpr], [Pred, Pred]) -> pure $ unPredCombinator cxxExpr
    ([cxxLhsExpr, cxxRhsExpr], [Pred, Pred, Pred]) ->
      pure $ binPredCombinator cxxLhsExpr cxxRhsExpr
    ([cxxLhsExpr, cxxRhsExpr], [Pred, a, b]) -> return $
      if a == b && name == FuncName "_==_"
      then pure $ CxxBinOp CxxEqual cxxLhsExpr cxxRhsExpr
      else Nothing
    ([], [Pred]) -> pure constPred
    _ -> return Nothing
  where
    constPred = case name of FuncName "FALSE" -> Just CxxFalse
                             FuncName "TRUE"  -> Just CxxTrue
                             _      -> Nothing

    unPredCombinator cxxExpr =
      let mCxxUnOp = case name of
            FuncName "!_" -> Just CxxLogicalNot
            _ -> Nothing
      in mCxxUnOp >>= \op -> Just $ CxxUnOp op cxxExpr

    binPredCombinator cxxLhsExpr cxxRhsExpr =
      let mCxxBinOp = case name of
            FuncName "_&&_" -> Just CxxLogicalAnd
            FuncName "_||_" -> Just CxxLogicalOr
            _ -> Nothing
      in mCxxBinOp >>= \op -> Just $ CxxBinOp op cxxLhsExpr cxxRhsExpr

mkCxxNamespaces :: FullyQualifiedName -> MgMonad [CxxNamespaceName]
mkCxxNamespaces fqName = maybe (return []) split (_scopeName fqName)
  where split (Name namespace nameStr) = case break (== '.') nameStr of
          ("", "") -> return []
          (ns, "") -> (:[]) <$> mkCxxName (Name namespace ns)
          (ns, _:rest) -> (:) <$> mkCxxName (Name namespace ns)
                              <*> split (Name namespace rest)