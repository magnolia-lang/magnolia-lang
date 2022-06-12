{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module MgToCuda (
    mgPackageToCudaSelfContainedProgramPackage
  )
  where

import Control.Monad.State
--import Control.Monad.IO.Class (liftIO)
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isJust)
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import Data.Tuple (swap)
import Data.Void (absurd)

import Cuda.Syntax
import Env
import Magnolia.PPrint
import Magnolia.Syntax
import MgToUtil
import Monad

-- TODOs left:
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

-- | Transforms a Magnolia typechecked package into a CUDA self-contained
-- package. Self-contained here means that a CUDA module (a struct or class) is
-- generated for each program defined in the Magnolia package. No code is
-- generated for other types of modules, and the same function exposed in two
-- different programs is code generated twice – once for each program.
mgPackageToCudaSelfContainedProgramPackage :: TcPackage -> MgMonad CudaPackage
mgPackageToCudaSelfContainedProgramPackage tcPkg =
  enter (nodeName tcPkg) $ do
    let allModules = join $
          map (getModules . snd) $ M.toList (_packageDecls $ _elem tcPkg)
        cudaPkgName = nodeName tcPkg
        cudaIncludes = mkCudaSystemInclude "cassert" :
          gatherCudaIncludes allModules
    cudaPrograms <- gatherPrograms allModules
    return $ CudaPackage cudaPkgName cudaIncludes cudaPrograms
  where
    moduleType :: TcModule -> MModuleType
    moduleType (Ann _ (MModule moduleTy _ _)) = moduleTy

    gatherCudaIncludes :: [TcModule] -> [CudaInclude]
    gatherCudaIncludes tcModules = S.toList $ foldl
      (\acc (Ann _ (MModule _ _ tcModuleExpr)) ->
        let newCudaIncludes = extractIncludeFromModuleExpr tcModuleExpr
        in acc `S.union` newCudaIncludes) S.empty tcModules

    extractIncludeFromModuleExpr :: TcModuleExpr -> S.Set CudaInclude
    extractIncludeFromModuleExpr (Ann _ moduleExpr) = case moduleExpr of
      MModuleRef v _ -> absurd v
      MModuleAsSignature v _ -> absurd v
      MModuleTransform _ v -> absurd v
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
    extractExternalReference :: TcDecl -> Maybe CudaInclude
    extractExternalReference tcDecl = case tcDecl of
      MTypeDecl _ (Ann (mconDeclO, _) _) ->
        mconDeclO >>= mkCudaIncludeFromConDeclO
      MCallableDecl _ (Ann (mconDeclO, _) _) ->
        mconDeclO >>= mkCudaIncludeFromConDeclO

    mkCudaIncludeFromConDeclO :: ConcreteDeclOrigin -> Maybe CudaInclude
    mkCudaIncludeFromConDeclO (ConcreteExternalDecl _ extDeclDetails) = Just $
      mkCudaRelativeCudaIncludeFromPath (externalDeclFilePath extDeclDetails)
    mkCudaIncludeFromConDeclO _ = Nothing

    gatherPrograms :: [TcModule] -> MgMonad [CudaModule]
    gatherPrograms = foldMAccumErrors
      (\acc -> ((:acc) <$>) . mgProgramToCudaProgramModule (nodeName tcPkg)) [] .
      filter ((== Program) . moduleType)

mgProgramToCudaProgramModule :: Name -> TcModule -> MgMonad CudaModule
mgProgramToCudaProgramModule
  pkgName (Ann declO (MModule Program name
                              tcModuleExpr@(Ann _ (MModuleDef decls deps _)))) =
  enter name $ do
    let moduleFqName = FullyQualifiedName (Just pkgName) name
        boundNames = S.fromList . map _name $
          name : M.keys decls <>
          join (map (map _targetName . snd . _ann) deps)

        declsToGen = filter declFilter declsAsList
        typeDecls = getTypeDecls declsToGen
        callableDecls = getCallableDecls declsToGen
        uniqueCallableNames = S.toList . S.fromList $
          map nodeName callableDecls

        (( returnTypeOverloadsNames
         , extObjectsNames
         , callableNamesAndFreshFnOpStructNames
         ), boundNames') = runState (do
            returnTypeOverloadsNames' <-
              mapM (freshNameM . fst) returnTypeOverloads
            -- External objects are completely synthetic, and therefore need
            -- to be given fresh names.
            extObjectsNames' <-
              mapM (freshObjectNameM . fst) referencedExternals

            callableNamesAndFreshFnOpStructNames' <-
              mapM (\callableName -> (callableName,) <$>
                      freshFunctionClassNameM callableName)
                    uniqueCallableNames

            pure ( returnTypeOverloadsNames'
                 , extObjectsNames'
                 , callableNamesAndFreshFnOpStructNames'
                 )) boundNames

        returnTypeOverloadsNameAliasMap =
          M.fromList (zip returnTypeOverloads returnTypeOverloadsNames)

        callableNamesToFreshFnOpStructNames =
          M.fromList callableNamesAndFreshFnOpStructNames

        toFnOpStructRequirement req = case _parameterDecl req of
          MTypeDecl {} -> pure req
          MCallableDecl mods (Ann src callable) -> do
            case M.lookup (_callableName callable)
                          callableNamesToFreshFnOpStructNames of
              Nothing -> throwNonLocatedE CompilerErr $
                "could not find callable " <> pshow (_callableName callable) <>
                " in supposedly exhaustive map of callables"
              Just fnOpStructName ->
                let newCallable = callable { _callableName = fnOpStructName }
                in pure $ req { _parameterDecl =
                                MCallableDecl mods (Ann src newCallable)
                              }

        extObjectsNameMap = M.fromList $ zip referencedExternals extObjectsNames

    moduleCudaNamespaces <- mkCudaNamespaces moduleFqName
    moduleCudaName <- mkCudaName name

    extObjectsCudaNameMap <- mapM mkCudaName extObjectsNameMap
    extObjectsDef <- mapMAccumErrors
      (\((structName, requirements), targetCudaName) -> do
          fnOpReqs <- mapM toFnOpStructRequirement requirements
          cudaStructName <- mkCudaName structName
          CudaInstance <$> mkCudaObject cudaStructName fnOpReqs targetCudaName)
      (M.toList extObjectsCudaNameMap)

    cudaTyDefs <-
      mapMAccumErrors (mgTyDeclToCudaTypeDef callableNamesToFreshFnOpStructNames)
                      typeDecls

    -- These CUDA functions are used to create data structures implementing the
    -- function call operator. An instance of these data structures is
    -- then created where needed invoke the relevant functions.
    -- We accumulate errors here, but fail if any was thrown, as the rest of
    -- the computation would not be meaningful anymore.
    -- TODO: overloading on return type, at the moment, requires writing
    --       something like f::operator()<T1, …, Tn>(a1, …, an) instead of
    --       the desired f<T1, …, Tn>(a1, …, an). This will eventually be fixed
    --       by generating functions wrapping calls to the operator, but is
    --       for the moment left out of the compiler.
    cudaFnDefs <- mapMAccumErrorsAndFail
      (mgCallableDeclToCuda returnTypeOverloadsNameAliasMap extObjectsCudaNameMap)
      callableDecls

    cudaFnTemplatedDefs <- mapM
      (uncurry . uncurry $ mgReturnTypeOverloadToCudaTemplatedDef boundNames')
      (M.toList returnTypeOverloadsNameAliasMap)

    let extObjectsCudaNameRMap = M.fromList $
          map swap $ M.toList extObjectsCudaNameMap
        -- TODO: do better at error reporting
        checkExtObjectExists e = case M.lookup e extObjectsCudaNameRMap of
          Nothing -> throwNonLocatedE CompilerErr $
            pshow e <> " is not an external object"
          Just einfo -> pure einfo
        mkFunctionCallOperatorStruct (callableName, fnOpStructName) = do
          cudaCallableName <- mkCudaName callableName
          let mustKeep cudaFnDef = _cudaFnName cudaFnDef == cudaCallableName
              (cudaFns', cudaFns'MgDecls) = unzip $
                filter (mustKeep . fst) (zip cudaFnDefs callableDecls)
              cudaFns = cudaFns' <> filter mustKeep cudaFnTemplatedDefs
              extObjectDependencyNames = foldl S.union S.empty $
                map extractCudaFnObjectDependencies cudaFns
          localDependencies <- do
            allDependencies <- S.toList <$>
              foldM (\acc callable -> S.union acc <$>
                extractCallableDependencies callable) S.empty cudaFns'MgDecls
            allDependencies'Cuda <- zip allDependencies <$>
              mapM (\n -> mkCudaName $ n { _name = tail $ _name n })
                allDependencies
            -- TODO: not sure if this is necessary anymore
            pure . S.fromList . map fst $ filter (\(_, cudaName) ->
              not $ cudaName `S.member` extObjectDependencyNames) allDependencies'Cuda
          extObjectDependencies <-
            let extObjectDependencyNames' = S.toList extObjectDependencyNames in
              flip zip extObjectDependencyNames' <$>
                mapM checkExtObjectExists extObjectDependencyNames'
          extObjectDependenciesDefs <-
            mapMAccumErrors (\((structName, requirements), targetCudaName) ->do
                fnOpReqs <- mapM toFnOpStructRequirement requirements
                cudaStructName <- mkCudaName structName
                CudaInstance <$> mkCudaObject cudaStructName fnOpReqs
                                              targetCudaName)
              extObjectDependencies
          case cudaFns of
            [] -> throwNonLocatedE CompilerErr $
              "could not find any CUDA generated implementation for " <>
              pshow cudaCallableName <> " but function generation supposedly " <>
              "succeeded"
            _ -> cudaFnsToFunctionCallOperatorStruct (fnOpStructName, cudaFns)
              extObjectDependenciesDefs localDependencies

    cudaFnCallOperatorStructs <- foldM
      (\acc (cn, sn) -> (:acc) <$> mkFunctionCallOperatorStruct (cn, sn)) []
      callableNamesAndFreshFnOpStructNames

    cudaFnCallOperatorObjects <- mapM (\(callableName, structName) -> do
        callableCudaName <- mkCudaName callableName
        cudaStructName <-
          mkCudaClassMemberAccess (CudaCustomType moduleCudaName) <$>
            mkCudaName structName
        mkCudaObject cudaStructName [] callableCudaName)
      callableNamesAndFreshFnOpStructNames

    cudaGeneratedFunctions <- do
      uniqueCallableCudaNames <-
        S.fromList <$> mapM mkCudaName uniqueCallableNames
      pure $ filter (\f -> _cudaFnName f `S.notMember` uniqueCallableCudaNames)
                    cudaFnDefs

    let allDefs =
             map (CudaPrivate,) (  extObjectsDef
                               <> map CudaFunctionDef cudaGeneratedFunctions)
          <> map (CudaPublic,) (  map (uncurry CudaTypeDef) cudaTyDefs
                              <> map CudaInstance cudaFnCallOperatorObjects
                              <> map CudaNestedModule cudaFnCallOperatorStructs)
    topSortedDefs <- topSortCudaDefs (srcCtx declO) moduleCudaName allDefs
    pure $ CudaModule moduleCudaNamespaces moduleCudaName topSortedDefs
  where
    declFilter :: TcDecl -> Bool
    declFilter decl = case decl of
      MTypeDecl _ _ -> True
      -- Only generate callables that are not assumed built-in.
      MCallableDecl _ (Ann _ callable) -> not $ isBuiltin callable

    isBuiltin :: MCallableDecl' p -> Bool
    isBuiltin cdecl = case _callableBody cdecl of BuiltinBody -> True
                                                  _ -> False

    returnTypeOverloads :: [(Name, [MType])]
    returnTypeOverloads = S.toList $
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
    referencedExternals = gatherUniqueExternalRequirements tcModuleExpr

mgProgramToCudaProgramModule _ _ = error "expected program"

-- | Sorts topologically a list of CUDA definitions with access specifiers
-- based on their dependencies. We assume the definitions to all belong to the
-- same module, whose name is passed as a parameter. For this to work, it is
-- crucial that each definition refers to other definitions of the same module
-- in a fully qualified way, i.e., if the parent module is called \'M\', and
-- contains a type \'T\', and a function \'f\' that returns an element of
-- type \'T\', the return type of \'f\' must be given as \'M::T\'.
topSortCudaDefs :: SrcCtx  -- ^ a source location for error reporting
               -> CudaName -- ^ the name of the parent module in CUDA
               -> [(CudaAccessSpec, CudaDef)]
               -> MgMonad [(CudaAccessSpec, CudaDef)]
topSortCudaDefs src parentModuleCudaName defs = do
  mapM checkAcyclic sccs
  where
    extractInName = extractAllChildrenNamesFromCudaClassTypeInCudaName
      (CudaCustomType parentModuleCudaName)
    extractInType = extractAllChildrenNamesFromCudaClassTypeInCudaType
      (CudaCustomType parentModuleCudaName)

    sccs =  let snd3 (_, b, _) = b
                -- TODO(not that important): check if the assessment below
                --
                -- I am not 100% sure if a call to sortOn is needed. The reason
                -- it is here is to ensure that across different runs, the
                -- output is always the same, even if the definitions are
                -- processed in a different order (which I think could be the
                -- case, if the implementation of Maps is different across
                -- platforms, which I assume it may be – but haven't checked.
                -- This is important so our tests don't become flakey on
                -- other platforms, though not for actual runs of the compiler.
                -- Note that since 'G.stronglyConnComp' makes no promise about
                -- stability afaik, tests may also become flaky if that
                -- dependency is updated.
                incidenceGraphInformation = L.sortOn snd3 $
                  map extractIncidenceGraphInformation defs
            in G.stronglyConnComp incidenceGraphInformation

    extractIncidenceGraphInformation
      :: (CudaAccessSpec, CudaDef)
      -> ((CudaAccessSpec, CudaDef), CudaName, [CudaName])
    extractIncidenceGraphInformation defAndAs@(_, def) =

      case def of
        CudaTypeDef sourceName targetName ->
          ( defAndAs
          , targetName
          , extractInName sourceName
          )
        CudaFunctionDef cudaFn ->
          ( defAndAs
          , _cudaFnName cudaFn
          , extractInType (_cudaFnReturnType cudaFn) <>
            join (map (extractInType . _cudaVarType) (_cudaFnParams cudaFn))
          )
        CudaNestedModule cudaMod ->
          let (_, _, allDependencies) = unzip3 $
                map extractIncidenceGraphInformation
                    (_cudaModuleDefinitions cudaMod)
          in ( defAndAs
             , _cudaModuleName cudaMod
             , join allDependencies
             )
        CudaInstance cudaObj ->
          ( defAndAs
          , _cudaObjectName cudaObj
          , extractInType (_cudaObjectType cudaObj)
          )

    checkAcyclic :: G.SCC (CudaAccessSpec, CudaDef)
                 -> MgMonad (CudaAccessSpec, CudaDef)
    checkAcyclic comp = case comp of
      G.AcyclicSCC defAndAs -> pure defAndAs
      G.CyclicSCC circularDepDefsAndAs ->
        let extractName def = case def of
              CudaTypeDef _ cudaTargetName -> cudaTargetName
              CudaFunctionDef cudaFn -> _cudaFnName cudaFn
              CudaNestedModule cudaMod -> _cudaModuleName cudaMod
              CudaInstance cudaObj -> _cudaObjectName cudaObj
            circularDepDefs = map snd circularDepDefsAndAs
        in throwLocatedE MiscErr src $
              "can not sort elements " <>
              T.intercalate ", " (map (pshow . extractName) circularDepDefs) <>
              " topologically"

-- | Produces a CUDA object from a data structure to instantiate along with
-- the required template parameters if necessary. All the generated objects are
-- static at the moment.
mkCudaObject :: CudaName       -- ^ the data structure to instantiate
            -> [Requirement] -- ^ the required arguments to the
                             --   data structure
            -> CudaName       -- ^ the name given to the resulting
                             --   object
            -> MgMonad CudaObject
mkCudaObject cudaStructName [] targetCudaName = do
  pure $ CudaObject (CudaCustomType cudaStructName) targetCudaName
mkCudaObject cudaStructName requirements@(_:_) targetCudaName = do
  let sortedRequirements = L.sortBy orderRequirements requirements
  cudaTemplateParameters <-
    mapM (mkClassMemberCudaType . nodeName . _parameterDecl)
         sortedRequirements
  pure $ CudaObject
            (CudaCustomTemplatedType cudaStructName cudaTemplateParameters)
            targetCudaName

-- | Takes a list of overloaded function definitions, a fresh name for the
-- module in Magnolia, and produces a function call operator module.
-- TODO: out of sync comment
cudaFnsToFunctionCallOperatorStruct :: (Name, [CudaFunctionDef])
                                    -> [CudaDef]
                                    -> S.Set Name
                                    -> MgMonad CudaModule
cudaFnsToFunctionCallOperatorStruct (moduleName, cudaFns) extObjDefs deps = do
  cudaModuleName <- mkCudaName moduleName
  let renamedCudaFns = map
        (\cudaFn -> cudaFn { _cudaFnName = cudaFunctionCallOperatorName
                           , _cudaFnType = CudaFunctionType'DeviceHost
                           }) cudaFns
  cudaParentModuleName <- getParentModuleName >>= mkCudaType
  cudaDependencyObjects <- mapM
    (\name -> do
        cudaStructName <- mkCudaClassMemberAccess cudaParentModuleName <$>
          mkCudaName (TypeName ("_" <> _name name))
        mkCudaName name >>= mkCudaObject cudaStructName [])
    (S.toList deps)
  pure $ CudaModule [] cudaModuleName $
    map ((CudaPrivate,) . CudaInstance) cudaDependencyObjects <>
    map (CudaPrivate,) extObjDefs <>
    map ((CudaPublic,) . CudaFunctionDef) renamedCudaFns

-- | Produces a CUDA typedef from a Magnolia type declaration. A map from
-- callable names to their function operator-implementing struct name is
-- provided to look up template parameters as needed.
mgTyDeclToCudaTypeDef :: M.Map Name Name
                     -> TcTypeDecl
                     -> MgMonad (CudaName, CudaName)
mgTyDeclToCudaTypeDef callableNamesToFnOpStructNames
    (Ann (conDeclO, absDeclOs) (Type targetTyName)) = do
  cudaTargetTyName <- mkCudaName targetTyName
  let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
      extStructName = externalDeclModuleName extDeclDetails
      extTyName = externalDeclElementName extDeclDetails
  sortedRequirements <- mapM (makeReqTypeName . _parameterDecl)
    (orderedRequirements extDeclDetails)
  checkCudaBackend errorLoc "type" extDeclDetails
  cudaExtStructName <- mkCudaName extStructName
  cudaExtTyName <- mkCudaName extTyName
  cudaSourceTyName <-
    if null sortedRequirements
    then pure $
      mkCudaClassMemberAccess (CudaCustomType cudaExtStructName) cudaExtTyName
    else do
      cudaReqTypes <- mapM mkClassMemberCudaType sortedRequirements
      pure $ mkCudaClassMemberAccess
        (CudaCustomTemplatedType cudaExtStructName cudaReqTypes) cudaExtTyName
  pure (cudaSourceTyName, cudaTargetTyName)
  where
    errorLoc = srcCtx $ NE.head absDeclOs

    makeReqTypeName decl = case decl of
      MCallableDecl {} ->
        case M.lookup (nodeName decl) callableNamesToFnOpStructNames of
          Nothing -> throwLocatedE CompilerErr errorLoc $
            "did not find a function operator-implementing struct for " <>
            "callable " <> pshow (nodeName decl)
          Just fnOpStructName -> pure fnOpStructName
      MTypeDecl {} -> pure $ nodeName decl

mgCallableDeclToCuda
  :: M.Map (Name, [MType]) Name
  -> M.Map (Name, [Requirement]) CudaName
  -> TcCallableDecl
  -> MgMonad CudaFunctionDef
mgCallableDeclToCuda returnTypeOverloadsNameAliasMap extObjectsMap
  mgFn@(Ann (conDeclO, absDeclOs)
            (Callable _ _ args retTy _ (ExternalBody _))) = do
    -- This case is hit when one of the API functions we want to expose
    -- is declared externally, and potentially renamed.
    -- In this case, to generate the API function, we need to perform
    -- an inline call to the external function.
    let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
        extStructName = externalDeclModuleName extDeclDetails
        extCallableName = externalDeclElementName extDeclDetails
        extOrderedRequirements = orderedRequirements extDeclDetails
        mgFnWithDummyMgBody = Ann (conDeclO, absDeclOs) $
          (_elem mgFn) { _callableBody = MagnoliaBody (Ann retTy MSkip) }
    cudaExtObjectName <-
      case M.lookup (extStructName, extOrderedRequirements) extObjectsMap of
        Nothing -> throwLocatedE CompilerErr (srcCtx $ NE.head absDeclOs) $
          "could not find matching external object definition for " <>
          pshow extStructName <> " with requirements " <>
          pshow (map _parameterDecl extOrderedRequirements)
        Just cudaExtObjName -> pure cudaExtObjName

    checkCudaBackend (srcCtx $ NE.head absDeclOs) "function" extDeclDetails

    cudaFnDef <- mgCallableDeclToCuda
      returnTypeOverloadsNameAliasMap extObjectsMap mgFnWithDummyMgBody
    cudaExtFnName <- mkCudaObjectMemberAccess cudaExtObjectName <$>
      mkCudaName extCallableName
    cudaArgExprs <- mapM ((CudaVarRef <$>) . mkCudaName . nodeName) args
    -- If mutification occurred, then it means that we must not return the
    -- value but instead assign it to the last variable in the argument list.
    -- Checking if mutification occur can be done by checking if the length
    -- of the argument list changed.
    let cudaBody = if length (_cudaFnParams cudaFnDef) == length args
                  then -- We synthesize a return function call
                    [ CudaStmtInline . CudaReturn . Just $
                      CudaCall cudaExtFnName [] cudaArgExprs
                    ]
                  else
                    let outVarCudaName =
                          _cudaVarName $ last (_cudaFnParams cudaFnDef)
                    in [ CudaStmtInline $
                           CudaAssign outVarCudaName
                                     (CudaCall cudaExtFnName [] cudaArgExprs)
                       ]
    pure cudaFnDef {_cudaFnBody = cudaBody }


mgCallableDeclToCuda returnTypeOverloadsNameAliasMap extObjectsMap
                    mgFn@(Ann _ (Callable cty name args retTy mguard cbody))
  | isOverloadedOnReturnType = do
      mgFn' <- mutify mgFn
      mgCallableDeclToCuda returnTypeOverloadsNameAliasMap extObjectsMap mgFn'
  | otherwise = do
      cudaFnName <- mkCudaName name
      cudaBody <- mgFnBodyToCudaStmtBlock returnTypeOverloadsNameAliasMap mgFn
      cudaRetTy <- case cty of
        Function -> mkClassMemberCudaType retTy
        Predicate -> mkCudaType Pred
        _ -> mkCudaType Unit -- Axiom and procedures
      cudaParams <- mapM mgTypedVarToCuda args
      -- TODO: what do we do with guard? Carry it in and use it as a
      -- precondition test?
      pure $ CudaFunction CudaFunctionType'DeviceHost True
        cudaFnName [] cudaParams cudaRetTy cudaBody
  where
    mgTypedVarToCuda :: TypedVar PhCheck -> MgMonad CudaVar
    mgTypedVarToCuda (Ann _ v) = do
      cudaVarName <- mkCudaName $ _varName v
      cudaVarType <- mkClassMemberCudaType $ _varType v
      return $ CudaVar (_varMode v == MObs) True cudaVarName cudaVarType

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
      MagnoliaBody e -> pure . MagnoliaBody . Ann Unit $
        MCall (ProcName "_=_")
              [Ann retTy (MVar (Ann retTy mutifiedOutVar)), e]
              Nothing
      _ -> pure cbody

-- | Generates a templated function given a set of functions overloaded on their
-- return type.
mgReturnTypeOverloadToCudaTemplatedDef :: S.Set String
                                      -- ^ the bound strings in the environment
                                      -> Name
                                      -- ^ the name of the overloaded function
                                      -> [MType]
                                      -- ^ the type of the arguments to the
                                      --   overloaded function
                                      -> Name
                                      -- ^ the name to give to the resulting
                                      --   mutified definition
                                      -> MgMonad CudaFunctionDef
mgReturnTypeOverloadToCudaTemplatedDef boundNs fnName argTypes mutifiedFnName =
  do
    argCudaNames <- mapM mkCudaName $
      evalState (mapM (freshNameM . const (VarName "a")) argTypes) boundNs
    cudaArgTypes <- mapM mkClassMemberCudaType argTypes
    let templateTyName = freshName (TypeName "T") boundNs
    cudaTemplateTy <- mkCudaType templateTyName
    outVarCudaName <- mkCudaName $ freshName (VarName "o") boundNs
    cudaFnName <- mkCudaName fnName
    mutifiedFnCudaName <- mkCudaName mutifiedFnName
    moduleCudaName <- getParentModuleName >>= mkCudaName
    let fullyQualifiedMutifiedFnCudaName = mkCudaClassMemberAccess
          (CudaCustomType moduleCudaName) mutifiedFnCudaName
        cudaOutVar = CudaVar { _cudaVarIsConst = False
                           , _cudaVarIsRef = False
                           , _cudaVarName = outVarCudaName
                           , _cudaVarType = cudaTemplateTy
                           }
        cudaFnParams = zipWith (\cudaArgName cudaArgTy ->
          CudaVar { _cudaVarIsConst = True
                 , _cudaVarIsRef = True
                 , _cudaVarName = cudaArgName
                 , _cudaVarType = cudaArgTy
                 } ) argCudaNames cudaArgTypes
        cudaCallArgExprs = map CudaVarRef (argCudaNames <> [outVarCudaName])
        cudaFnBody = map CudaStmtInline
          [ -- T o;
            CudaVarDecl cudaOutVar Nothing
            -- mutifiedFnName(a0, a1, …, an, &o);
          , CudaExpr (CudaCall fullyQualifiedMutifiedFnCudaName []
                             cudaCallArgExprs)
            -- return o;
          , CudaReturn (Just (CudaVarRef outVarCudaName))
          ]
    -- TODO: as of June 2022, we only support __device__ __host__ functions and
    -- __global__ methods. Methods can *not* be overloaded, and this must
    -- therefore be a __device__ __host__ function.
    pure $
      CudaFunction { _cudaFnType = CudaFunctionType'DeviceHost
                   , _cudaFnIsInline = True
                   , _cudaFnName = cudaFnName
                   , _cudaFnTemplateParameters = [cudaTemplateTy]
                   , _cudaFnParams = cudaFnParams
                   , _cudaFnReturnType = cudaTemplateTy
                   , _cudaFnBody = cudaFnBody
                   }

mgFnBodyToCudaStmtBlock :: M.Map (Name, [MType]) Name -> TcCallableDecl
                       -> MgMonad CudaStmtBlock
mgFnBodyToCudaStmtBlock returnTypeOverloadsNameAliasMap
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
    mapM (mgExprToCudaStmt returnTypeOverloadsNameAliasMap) (NE.toList exprs)
  MagnoliaBody expr -> (:[]) <$>
    mgExprToCudaStmt returnTypeOverloadsNameAliasMap (insertValueBlocks expr)

insertValueBlocks :: TcExpr -> TcExpr
insertValueBlocks inExpr@(Ann Unit _) = inExpr
insertValueBlocks inExpr@(Ann ty e) = case e of
  MValue _ -> inExpr
  MBlockExpr _ _ -> inExpr
  MIf cond trueExpr falseExpr -> Ann ty $
    MIf cond (insertValueBlocks trueExpr) (insertValueBlocks falseExpr)
  _ -> Ann ty $ MValue inExpr

mgExprToCudaStmt :: M.Map (Name, [MType]) Name -> TcExpr -> MgMonad CudaStmt
mgExprToCudaStmt returnTypeOverloadsNameAliasMap = goStmt
  where
    goStmt annInExpr@(Ann _ inExpr) = case inExpr of
      MVar _ -> CudaStmtInline . CudaExpr <$> goExpr annInExpr
      -- TODO: handle special case of assignment in a prettier way. ATM,
      --       this seems good enough.
      MCall (ProcName "_=_") [Ann tyLhs (MVar v), rhs@(Ann tyRhs _)] _ ->
        if tyLhs == tyRhs then do
          cudaVarName <- mkCudaName . _varName $ _elem v
          cudaRhs <- goExpr rhs
          return . CudaStmtInline $ CudaAssign cudaVarName cudaRhs
        else CudaStmtInline . CudaExpr <$> goExpr annInExpr
      MCall {} -> CudaStmtInline . CudaExpr <$> goExpr annInExpr
      MBlockExpr _ stmts -> CudaStmtBlock <$> mapM goStmt (NE.toList stmts)
      MValue _ -> CudaStmtInline . CudaReturn . Just <$> goExpr annInExpr
      MLet (Ann vty (Var mode name _)) mexpr -> do
        cudaVarName <- mkCudaName name
        cudaVarType <- mkClassMemberCudaType vty
        mcudaRhsExpr <- maybe (return Nothing) ((Just <$>) . goExpr) mexpr
        let cudaVarIsConst = mode == MObs
            cudaVarIsRef = False -- TODO: figure out when to use refs?
            cudaVar = CudaVar cudaVarIsConst cudaVarIsRef cudaVarName cudaVarType
        return . CudaStmtInline $ CudaVarDecl cudaVar mcudaRhsExpr
      MIf cond trueStmt falseStmt -> do
        cudaCond <- goExpr cond
        cudaTrueStmt <- goStmt trueStmt
        cudaFalseStmt <- goStmt falseStmt
        return . CudaStmtInline $ CudaIf cudaCond cudaTrueStmt cudaFalseStmt
      MAssert expr -> CudaStmtInline . CudaAssert <$> goExpr expr
      MSkip -> return $ CudaStmtInline CudaSkip
    goExpr = mgExprToCudaExpr returnTypeOverloadsNameAliasMap

mgExprToCudaExpr :: M.Map (Name, [MType]) Name -> TcExpr -> MgMonad CudaExpr
mgExprToCudaExpr returnTypeOverloadsNameAliasMap = goExpr
  where
    goExpr annInExpr@(Ann ty inExpr) = case inExpr of
      MVar (Ann _ v) -> CudaVarRef <$> mkCudaName (_varName v)
      MCall name args _ -> do
        mCudaExpr <- tryMgCallToCudaSpecialOpExpr returnTypeOverloadsNameAliasMap
          name args ty
        case mCudaExpr of
          Just cudaExpr -> return cudaExpr
          Nothing -> do
            let inputProto = (name, map _ann args)
            cudaTemplateArgs <- mapM mkCudaType
              [ty | inputProto `M.member` returnTypeOverloadsNameAliasMap]
            cudaArgs <- mapM goExpr args
            -- TODO: right now, we need to call operator() with templated types.
            --       This is not great from the point of view of the exposed
            --       and it will be fixed at some point.
            case cudaTemplateArgs of
              [] -> do
                cudaFnName <- mkCudaName name
                pure $ CudaCall cudaFnName cudaTemplateArgs cudaArgs
              _:_ -> do
                cudaFunctionObjectName <- mkCudaName name
                let cudaOperatorName = mkCudaObjectMemberAccess
                      cudaFunctionObjectName
                      cudaFunctionCallOperatorName
                pure $ CudaCall cudaOperatorName cudaTemplateArgs cudaArgs
      MBlockExpr blockTy exprs -> do
        cudaExprs <- mapM goStmt (NE.toList exprs)
        case blockTy of
          -- TODO: check properly that ref is okay here (it should be)
          MValueBlock -> mgToCudaLambdaRef cudaExprs
          MEffectfulBlock -> mgToCudaLambdaRef cudaExprs
      MValue expr -> goExpr expr
      -- TODO: check properly that ref is okay here (it should be)
      MLet {} -> goStmt annInExpr >>= mgToCudaLambdaRef . (:[])
      MIf cond trueExpr falseExpr -> CudaIfExpr <$> goExpr cond <*>
        goExpr trueExpr <*> goExpr falseExpr
      -- TODO: check properly that ref is okay here (it should be)
      MAssert _ -> goStmt annInExpr >>= mgToCudaLambdaRef . (:[])
      MSkip -> mgToCudaLambdaRef []

    goStmt = mgExprToCudaStmt returnTypeOverloadsNameAliasMap
    mgToCudaLambdaRef = return . CudaLambdaCall CudaLambdaCaptureDefaultReference
    --mgToCudaLambdaVal = return . CudaLambdaCall CudaLambdaCaptureDefaultValue

-- TODO: for the moment, this assumes no '_:T ==_:T' predicate is implemented
-- in Magnolia, although it will be possible to define one manually. Therefore,
-- this will have to be improved later.
-- | Takes the name of a function, its arguments and its return type, and
-- produces a unop or binop expression node if it can be expressed as such in
-- CUDA. For the moment, functions that can be expressed like that include the
-- equality predicate between two elements of the same type, predicate
-- combinators (such as '_&&_' and '_||_'), and the boolean constants 'TRUE',
-- and 'FALSE'.
tryMgCallToCudaSpecialOpExpr :: M.Map (Name, [MType]) Name
                            -> Name -> [TcExpr] -> MType
                            -> MgMonad (Maybe CudaExpr)
tryMgCallToCudaSpecialOpExpr returnTypeOverloadsNameAliasMap name args retTy = do
  cudaArgs <- mapM (mgExprToCudaExpr returnTypeOverloadsNameAliasMap) args
  case (cudaArgs, retTy : map _ann args ) of
    ([cudaExpr], [Pred, Pred]) -> pure $ unPredCombinator cudaExpr
    ([cudaLhsExpr, cudaRhsExpr], [Pred, Pred, Pred]) ->
      pure $ binPredCombinator cudaLhsExpr cudaRhsExpr
    ([cudaLhsExpr, cudaRhsExpr], [Pred, a, b]) -> return $
      if a == b
      then case name of
        FuncName "_==_" -> pure $ CudaBinOp CudaEqual cudaLhsExpr cudaRhsExpr
        FuncName "_!=_" -> pure $ CudaBinOp CudaNotEqual cudaLhsExpr cudaRhsExpr
        _ -> Nothing
      else Nothing
    ([], [Pred]) -> pure constPred
    _ -> return Nothing
  where
    constPred = case name of FuncName "FALSE" -> Just CudaFalse
                             FuncName "TRUE"  -> Just CudaTrue
                             _      -> Nothing

    unPredCombinator cudaExpr =
      let mCudaUnOp = case name of
            FuncName "!_" -> Just CudaLogicalNot
            _ -> Nothing
      in mCudaUnOp >>= \op -> Just $ CudaUnOp op cudaExpr

    binPredCombinator cudaLhsExpr cudaRhsExpr =
      let mCudaBinOp = case name of
            FuncName "_&&_" -> Just CudaLogicalAnd
            FuncName "_||_" -> Just CudaLogicalOr
            _ -> Nothing
      in mCudaBinOp >>= \op -> Just $ CudaBinOp op cudaLhsExpr cudaRhsExpr

mkCudaNamespaces :: FullyQualifiedName -> MgMonad [CudaNamespaceName]
mkCudaNamespaces fqName = maybe (return []) split (_scopeName fqName)
  where split (Name namespace nameStr) = case break (== '.') nameStr of
          ("", "") -> return []
          (ns, "") -> (:[]) <$> mkCudaName (Name namespace ns)
          (ns, _:rest) -> (:) <$> mkCudaName (Name namespace ns)
                              <*> split (Name namespace rest)

-- | Finds the name of all the callables upon which the input callable depends.
-- TODO: functions should not depend on overloads with the same name for this
-- to work as intended. Let's fix this later, no time now.
extractCallableDependencies :: TcCallableDecl -> MgMonad (S.Set Name)
extractCallableDependencies (Ann _ callable) =
  case _callableBody callable of
    MagnoliaBody expr -> pure $ go expr
    ExternalBody () -> pure S.empty
    _ -> throwNonLocatedE CompilerErr $
      "tried to extract dependencies for codegen of unimplemented " <>
      pshow (_callableName callable)
  where
    go :: TcExpr -> S.Set Name
    go (Ann _ expr) = case expr of
      MVar {} -> noDeps
      -- TODO: what happens if f depends on another function also named f?
      -- Probably a bug. We ignore it for the prototype, and will fix it
      -- later.
      MCall name tcArgs _ -> foldl S.union (ignoreBuiltin name) $ map go tcArgs
      MBlockExpr _ tcStmts -> foldl S.union S.empty $ NE.map go tcStmts
      MValue tcExpr' -> go tcExpr'
      MLet _ tcExpr' -> maybe S.empty go tcExpr'
      MIf tcCond tcTrue tcFalse ->
        go tcCond `S.union` go tcTrue `S.union` go tcFalse
      MAssert tcExpr' -> go tcExpr'
      MSkip -> noDeps

    noDeps = S.empty

    ignoreBuiltin :: Name -> S.Set Name
    ignoreBuiltin name =
      if _name name `S.member` builtins then S.empty else S.singleton name

    builtins = S.fromList [ "_&&_", "_||_", "_!=_", "_==_", "_=>_", "_<=>_"
                          , "_=_", "!_"]

-- | Finds the name of all the objects upon which the input CUDA function
-- definition depends. A CUDA function definition depends on an object if it
-- contains a call to a member of an object.
extractCudaFnObjectDependencies :: CudaFunctionDef -> S.Set CudaName
extractCudaFnObjectDependencies (CudaFunction _ _ _ _ _ _ body) = goBlock body
  where
    goStmt :: CudaStmt -> S.Set CudaName
    goStmt cuStmt = case cuStmt of
      CudaStmtBlock block -> goBlock block
      CudaStmtInline inlineStmt -> case inlineStmt of
        CudaAssign _ cuRhsExpr -> goExpr cuRhsExpr
        CudaVarDecl _ mcuRhsExpr -> maybe S.empty goExpr mcuRhsExpr
        CudaAssert cuExpr -> goExpr cuExpr
        CudaIf cuCond cuBTrue cuBFalse ->
          goExpr cuCond `S.union` goStmt cuBTrue `S.union` goStmt cuBFalse
        CudaExpr cuExpr -> goExpr cuExpr
        CudaReturn mcuExpr -> maybe S.empty goExpr mcuExpr
        CudaSkip -> S.empty

    goBlock :: CudaStmtBlock -> S.Set CudaName
    goBlock block = foldl (\acc stmt -> S.union acc $ goStmt stmt) S.empty block

    goExpr :: CudaExpr -> S.Set CudaName
    goExpr cuExpr = case cuExpr of
      CudaCall cuFnName _ cuFnArgs ->
        foldl (\acc expr -> S.union acc $ goExpr expr) (memitDep cuFnName)
              cuFnArgs
      CudaGlobalCall _ cuFnName _ cuFnArgs ->
        foldl (\acc expr -> S.union acc $ goExpr expr) (memitDep cuFnName)
              cuFnArgs
      CudaLambdaCall _ cuLambdaBlock -> goBlock cuLambdaBlock
      CudaVarRef cuVarName -> memitDep cuVarName
      CudaIfExpr cuCond cuBTrue cuBFalse ->
        goExpr cuCond `S.union` goExpr cuBTrue `S.union` goExpr cuBFalse
      CudaUnOp _ cuExpr' -> goExpr cuExpr'
      CudaBinOp _ cuLhsExpr cuRhsExpr ->
        goExpr cuLhsExpr `S.union` goExpr cuRhsExpr
      CudaTrue -> noDeps
      CudaFalse -> noDeps

    noDeps = S.empty

    memitDep :: CudaName -> S.Set CudaName
    memitDep cuName = case getCudaObjectName cuName of
      Nothing -> S.empty
      Just cuObjName -> case getCudaMemberName cuName of
        Nothing -> unreachable -- should not happen
        Just cuMemberName ->
          if cuMemberName == cudaFunctionCallOperatorName
          then S.empty
          else S.singleton cuObjName

    unreachable = error "unreachable code in memitDep (MgToCuda)"