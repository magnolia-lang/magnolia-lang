{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module MgToCxx (
    mgPackageToCxxSelfContainedProgramPackage
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

type BoundNames = S.Set String

-- | Not really a map, but a wrapper over 'foldMAccumErrors' that allows writing
-- code like
--
-- >>> mapM f t
--
-- on lists that accumulates errors within the 'MgMonad' and returns a list of
-- items.
mapMAccumErrors :: (a -> MgMonad b) -> [a] -> MgMonad [b]
mapMAccumErrors f = foldMAccumErrors (\acc a -> (acc <>) . (:[]) <$> f a) []

-- | Like 'mapMAccumErrors' but fails if any error was thrown by the time the
-- end of the list is reached.
mapMAccumErrorsAndFail :: (a -> MgMonad b) -> [a] -> MgMonad [b]
mapMAccumErrorsAndFail f = foldMAccumErrorsAndFail
  (\acc a -> (acc <>) . (:[]) <$> f a) []

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
    gatherPrograms = foldMAccumErrors
      (\acc -> ((:acc) <$>) . mgProgramToCxxProgramModule (nodeName tcPkg)) [] .
      filter ((== Program) . moduleType)

mgProgramToCxxProgramModule :: Name -> TcModule -> MgMonad CxxModule
mgProgramToCxxProgramModule
  pkgName (Ann declO (MModule Program name (Ann _ (MModuleDef decls deps _)))) =
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

        -- TODO: join boundNames handling into one op returning everything
        (boundNames', returnTypeOverloadsNameAliasMap) =
          foldl (\(boundNs, nameMap) rTyO ->
                let (boundNs', newName) = registerFreshName boundNs (fst rTyO)
                in (boundNs', M.insert rTyO newName nameMap))
            (boundNames, M.empty) (S.toList returnTypeOverloads)

        (( extObjectsNames
         , callableNamesAndFreshFnOpStructNames
         ), _) = runState (do -- TODO: change to execState once sure we do
                              --       not need to retrieve the bound names.
            -- External objects are completely synthetic, and therefore need
            -- to be given fresh names.
            extObjectNames <- mapM (freshObjectName . fst) referencedExternals

            callableNamesAndFreshFnOpStructNames' <-
              mapM (\callableName -> (callableName,) <$>
                      freshFunctionClassName callableName)
                    uniqueCallableNames

            pure ( extObjectNames
                 , callableNamesAndFreshFnOpStructNames'
                 )) boundNames'

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

    moduleCxxNamespaces <- mkCxxNamespaces moduleFqName
    moduleCxxName <- mkCxxName name

    extObjectsCxxNameMap <- mapM mkCxxName extObjectsNameMap
    extObjectsDef <- mapMAccumErrors
      (\((structName, requirements), targetCxxName) -> do
          fnOpReqs <- mapM toFnOpStructRequirement requirements
          cxxStructName <- mkCxxName structName
          CxxInstance <$> mkCxxObject cxxStructName fnOpReqs targetCxxName)
      (M.toList extObjectsCxxNameMap)

    cxxTyDefs <-
      mapMAccumErrors (mgTyDeclToCxxTypeDef callableNamesToFreshFnOpStructNames)
                      typeDecls

    -- These C++ functions are used to create data structures implemention the
    -- function call operator. A static instance of these data structures is
    -- then created which can be used to invoked the relevant functions.
    -- We accumulate errors here, but fail if any was thrown, as the rest of
    -- the computation would not be meaningful anymore.
    -- TODO: overloading on return type, at the moment, requires writing
    --       something like f::operator()<T1, …, Tn>(a1, …, an) instead of
    --       the desired f<T1, …, Tn>(a1, …, an). This will eventually be fixed
    --       by generating functions wrapping calls to the operator, but is
    --       for the moment left out of the compiler.
    cxxFnDefs <- mapMAccumErrorsAndFail
      (mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsCxxNameMap)
      callableDecls

    cxxFnTemplatedDefs <- mapM
      (uncurry . uncurry $ mgReturnTypeOverloadToCxxTemplatedDef boundNames')
      (M.toList returnTypeOverloadsNameAliasMap)

    let mkFunctionCallOperatorStruct (callableName, fnOpStructName) = do
          cxxCallableName <- mkCxxName callableName
          let cxxFns = filter (\cxxFn -> _cxxFnName cxxFn == cxxCallableName)
                              (cxxFnDefs <> cxxFnTemplatedDefs)
          case cxxFns of
            [] -> throwNonLocatedE CompilerErr $
              "could not find any C++ generated implementation for " <>
              pshow cxxCallableName <> " but function generation supposedly " <>
              "succeeded"
            _ -> cxxFnsToFunctionCallOperatorStruct (fnOpStructName, cxxFns)

    cxxFnCallOperatorStructs <- foldM
      (\acc (cn, sn) -> (:acc) <$> mkFunctionCallOperatorStruct (cn, sn)) []
      callableNamesAndFreshFnOpStructNames

    cxxFnCallOperatorObjects <- mapM (\(callableName, structName) -> do
        callableCxxName <- mkCxxName callableName
        structCxxName <-
          mkCxxClassMemberAccess (CxxCustomType moduleCxxName) <$>
            mkCxxName structName
        mkCxxObject structCxxName [] callableCxxName)
      callableNamesAndFreshFnOpStructNames

    cxxGeneratedFunctions <- do
      uniqueCallableCxxNames <-
        S.fromList <$> mapM mkCxxName uniqueCallableNames
      pure $ filter (\f -> _cxxFnName f `S.notMember` uniqueCallableCxxNames)
                    cxxFnDefs

    topSortedCxxTyDefs <- topSortCxxTypeDefs (srcCtx declO) cxxTyDefs

    let allDefs =
             map (CxxPrivate,) (  extObjectsDef
                               <> map CxxFunctionDef cxxGeneratedFunctions)
          <> map (CxxPublic,) (  map (uncurry CxxTypeDef) topSortedCxxTyDefs
                              <> map CxxInstance cxxFnCallOperatorObjects
                              <> map CxxNestedModule cxxFnCallOperatorStructs)
    pure $ CxxModule moduleCxxNamespaces moduleCxxName (L.sortOn snd allDefs)
  where
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

mgProgramToCxxProgramModule _ _ = error "expected program"

-- | Checks that a list of C++ type definitions can be ordered topologically
-- and returns one such sort.
-- TODO: check if this is stable
topSortCxxTypeDefs :: SrcCtx -- ^ the source information of the program being
                             --   generated for error reporting purposes
                   -> [(CxxName, CxxName)]
                   -> MgMonad [(CxxName, CxxName)]
topSortCxxTypeDefs src typeDefs = do
  --liftIO $ pprint cs
  mapM checkAcyclic sccs
  where
    sccs = G.stronglyConnComp
      [ ( tyDef
        , targetName
        , extractTemplateParametersFromCxxName sourceName
        )
        | tyDef@(sourceName, targetName) <- typeDefs
      ]

    checkAcyclic :: G.SCC (CxxName, CxxName) -> MgMonad (CxxName, CxxName)
    checkAcyclic comp = case comp of
      G.AcyclicSCC tyDef -> pure tyDef
      G.CyclicSCC circularDepTyDefs ->
        throwLocatedE MiscErr src $
          "can not sort types " <>
          T.intercalate ", " (map (pshow . snd) circularDepTyDefs) <>
          " topologically"

-- | Produces a C++ object from a data structure to instantiate along with
-- the required template parameters if necessary. All the generated objects are
-- static at the moment.
mkCxxObject :: CxxName       -- ^ the data structure to instantiate
            -> [Requirement] -- ^ the required arguments to the
                             --   data structure
            -> CxxName       -- ^ the name given to the resulting
                             --   object
            -> MgMonad CxxObject
mkCxxObject structCxxName [] targetCxxName = do
  pure $ CxxObject CxxStaticMember (CxxCustomType structCxxName) targetCxxName
mkCxxObject structCxxName requirements@(_:_) targetCxxName = do
  let sortedRequirements = L.sortBy orderRequirements requirements
  cxxTemplateParameters <-
    mapM (mkClassMemberCxxType . nodeName . _parameterDecl)
         sortedRequirements
  pure $ CxxObject CxxStaticMember
                   (CxxCustomTemplatedType structCxxName cxxTemplateParameters)
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

-- | Takes a list of overloaded function definitions, a fresh name for the
-- module in Magnolia, and produces a function call operator module.
cxxFnsToFunctionCallOperatorStruct :: (Name, [CxxFunctionDef])
                                   -> MgMonad CxxModule
cxxFnsToFunctionCallOperatorStruct (moduleName, cxxFns) = do
  cxxModuleName <- mkCxxName moduleName
  let renamedCxxFns = map
        (\cxxFn -> cxxFn { _cxxFnName = cxxFunctionCallOperatorName
                         , _cxxFnModuleMemberType = CxxNonStaticMember
                         }) cxxFns
  pure $ CxxModule [] cxxModuleName $
    map ((CxxPublic,) . CxxFunctionDef) renamedCxxFns

-- | Produces a fresh name based on an initial input name and a set of bound
-- strings. If the input name's String component is not in the set of bound
-- strings, it is returned as is.
freshName :: Name -> BoundNames -> Name
freshName name@(Name ns str) boundNs
  | str `S.member` boundNs = freshName' (0 :: Int)
  | otherwise = name
  where
    freshName' i | (str <> show i) `S.member` boundNs = freshName' (i + 1)
                 | otherwise = Name ns $ str <> show i

-- | Like 'freshName' but using a State monad.
freshNameM :: Monad m => Name -> StateT BoundNames m Name
freshNameM n = do
  env <- get
  let freeName = freshName n env
  modify (S.insert $ _name freeName)
  pure freeName

-- | Like 'freshNameM', following the naming convention for objects.
freshObjectName :: Monad m => Name -> StateT BoundNames m Name
freshObjectName n = freshNameM n { _name = "__" <> _name n }

-- | Like 'freshNameM', following the naming convention for function classes.
freshFunctionClassName :: Monad m => Name -> StateT BoundNames m Name
freshFunctionClassName n = freshNameM n { _name = "_" <> _name n }

-- TODO: replace uses with freshNameM
-- | Produces a fresh name and returns both an updated set of bound strings that
-- includes it, and the new name.
registerFreshName :: S.Set String -> Name -> (S.Set String, Name)
registerFreshName boundNs name = let newName = freshName name boundNs
                                 in (S.insert (_name newName) boundNs, newName)

-- | Produces a C++ typedef from a Magnolia type declaration. A map from
-- callable names to their function operator-implementing struct name is
-- provided to look up template parameters as needed.
mgTyDeclToCxxTypeDef :: M.Map Name Name
                     -> TcTypeDecl
                     -> MgMonad (CxxName, CxxName)
mgTyDeclToCxxTypeDef callableNamesToFnOpStructNames
    (Ann (conDeclO, absDeclOs) (Type targetTyName)) = do
  cxxTargetTyName <- mkCxxName targetTyName
  let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
      extStructName = externalDeclModuleName extDeclDetails
      extTyName = externalDeclElementName extDeclDetails
  sortedRequirements <- mapM (makeReqTypeName . _parameterDecl) $
    orderedRequirements extDeclDetails
  checkCxxBackend errorLoc "type" extDeclDetails
  cxxExtStructName <- mkCxxName extStructName
  cxxExtTyName <- mkCxxName extTyName
  cxxSourceTyName <-
    if null sortedRequirements
    then pure $
      mkCxxClassMemberAccess (CxxCustomType cxxExtStructName) cxxExtTyName
    else do
      cxxReqTypes <- mapM ((CxxCustomType <$>) . mkCxxName) sortedRequirements
      pure $ mkCxxClassMemberAccess
        (CxxCustomTemplatedType cxxExtStructName cxxReqTypes) cxxExtTyName
  pure (cxxSourceTyName, cxxTargetTyName)
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

-- | Checks whether some 'ExternalDeclDetails' correspond to a declaration with
-- a C++ backend. Throws an error if it is not the case.
checkCxxBackend :: SrcCtx -- ^ source information for error reporting
                -> T.Text -- ^ the type of declaration we check; this is
                          --   used verbatim in the error message
                -> ExternalDeclDetails
                -> MgMonad ()
checkCxxBackend src prettyDeclTy extDeclDetails =
  let backend = externalDeclBackend extDeclDetails
      extStructName = externalDeclModuleName extDeclDetails
      extTyName = externalDeclElementName extDeclDetails
  in unless (backend == Cxx) $ throwLocatedE MiscErr src $
      "attempted to generate C++ code relying on external " <> prettyDeclTy <>
      " " <> pshow (FullyQualifiedName (Just extStructName) extTyName) <>
      " but it is declared as having a " <> pshow backend <> " backend"

mgCallableDeclToCxx
  :: M.Map (Name, [MType]) Name
  -> M.Map (Name, [Requirement]) CxxName
  -> TcCallableDecl
  -> MgMonad CxxFunctionDef
mgCallableDeclToCxx returnTypeOverloadsNameAliasMap extObjectsMap
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
        ~(Just cxxExtObjectName) =
          M.lookup (extStructName, extOrderedRequirements) extObjectsMap

    checkCxxBackend (srcCtx $ NE.head absDeclOs) "function" extDeclDetails

    cxxFnDef <- mgCallableDeclToCxx
      returnTypeOverloadsNameAliasMap extObjectsMap mgFnWithDummyMgBody
    cxxExtFnName <- mkCxxObjectMemberAccess cxxExtObjectName <$>
      mkCxxName extCallableName
    cxxArgExprs <- mapM ((CxxVarRef <$>) . mkCxxName . nodeName) args
    -- If mutification occurred, then it means that we must not return the
    -- value but instead assign it to the last variable in the argument list.
    -- Checking if mutification occur can be done by checking if the length
    -- of the argument list changed.
    let cxxBody = if length (_cxxFnParams cxxFnDef) == length args
                  then -- We synthesize a return function call
                    [ CxxStmtInline . CxxReturn . Just $
                      CxxCall cxxExtFnName [] cxxArgExprs
                    ]
                  else
                    let outVarCxxName =
                          _cxxVarName $ last (_cxxFnParams cxxFnDef)
                    in [ CxxStmtInline $
                           CxxAssign outVarCxxName
                                     (CxxCall cxxExtFnName [] cxxArgExprs)
                       ]
    pure cxxFnDef {_cxxFnBody = cxxBody }


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
      -- TODO: for now, all functions are generated as static.
      pure $ CxxFunction CxxStaticMember False cxxFnName [] cxxParams cxxRetTy
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
      MagnoliaBody e -> pure . MagnoliaBody . Ann Unit $
        MCall (ProcName "_=_")
              [Ann retTy (MVar (Ann retTy mutifiedOutVar)), e]
              Nothing
      _ -> pure cbody

-- | Generates a templated function given a set of functions overloaded on their
-- return type.
mgReturnTypeOverloadToCxxTemplatedDef :: S.Set String
                                      -- ^ the bound strings in the environment
                                      -> Name
                                      -- ^ the name of the overloaded function
                                      -> [MType]
                                      -- ^ the type of the arguments to the
                                      --   overloaded function
                                      -> Name
                                      -- ^ the name to give to the resulting
                                      --   mutified definition
                                      -> MgMonad CxxFunctionDef
mgReturnTypeOverloadToCxxTemplatedDef boundNs fnName argTypes mutifiedFnName =
  do
    argCxxNames <- mapM mkCxxName $ snd $
      L.mapAccumL registerFreshName boundNs (map (const (VarName "a")) argTypes)
    cxxArgTypes <- mapM mkClassMemberCxxType argTypes
    let templateTyName = freshName (TypeName "T") boundNs
    cxxTemplateTy <- mkCxxType templateTyName
    outVarCxxName <- mkCxxName $ freshName (VarName "o") boundNs
    cxxFnName <- mkCxxName fnName
    mutifiedFnCxxName <- mkCxxName mutifiedFnName
    moduleCxxName <- getParentModuleName >>= mkCxxName
    let fullyQualifiedMutifiedFnCxxName = mkCxxClassMemberAccess
          (CxxCustomType moduleCxxName) mutifiedFnCxxName
        cxxOutVar = CxxVar { _cxxVarIsConst = False
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
          , CxxExpr (CxxCall fullyQualifiedMutifiedFnCxxName []
                             cxxCallArgExprs)
            -- return o;
          , CxxReturn (Just (CxxVarRef outVarCxxName))
          ]
    pure $
      CxxFunction { _cxxFnModuleMemberType = CxxStaticMember
                  , _cxxFnIsInline = True
                  , _cxxFnName = cxxFnName
                  , _cxxFnTemplateParameters = [cxxTemplateTy]
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
            cxxModuleName <- getParentModuleName >>= mkCxxName
            let inputProto = (name, map _ann args)
            cxxTemplateArgs <- mapM mkCxxType
              [ty | inputProto `M.member` returnTypeOverloadsNameAliasMap]
            cxxArgs <- mapM goExpr args
            -- TODO: right now, we need to call operator() with templated types.
            --       This is not great from the point of view of the exposed
            --       and it will be fixed at some point.
            case cxxTemplateArgs of
              [] -> do
                cxxName <-
                  mkCxxClassMemberAccess (CxxCustomType cxxModuleName) <$>
                    mkCxxName name
                pure $ CxxCall cxxName cxxTemplateArgs cxxArgs
              _:_ -> do
                cxxFunctionObjectName <- mkCxxName name
                let cxxOperatorName = mkCxxObjectMemberAccess
                      cxxFunctionObjectName
                      cxxFunctionCallOperatorName
                    cxxName = mkCxxClassMemberAccess
                      (CxxCustomType cxxModuleName)
                      cxxOperatorName
                pure $ CxxCall cxxName cxxTemplateArgs cxxArgs
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