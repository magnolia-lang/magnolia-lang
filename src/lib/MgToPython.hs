{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module MgToPython (
    mgPackageToPySelfContainedProgramPackage
  )
  where

import Control.Monad.State
import qualified Data.Graph as G
import qualified Data.List as L
import Data.List.NonEmpty (NonEmpty ((:|)))
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (fromMaybe)
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import Data.Void (absurd)

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import MgToUtil
import Monad

import Python.Syntax

type ArgScope = S.Set Name

-- TODOs left:
--  - document assumptions
--  - document initializing out variables and then mutating them

mgPackageToPySelfContainedProgramPackage :: TcPackage
                                         -> MgMonad PyPackage
mgPackageToPySelfContainedProgramPackage tcPkg =
  enter (nodeName tcPkg) $ do
    let allModules = join $
          map (getModules . snd) $ M.toList (_packageDecls $ _elem tcPkg)
        pyPkgName = nodeName tcPkg
        -- We need to import our 'multiple_dispatch.py' file located at the
        -- root of the project (for the moment), as well as 'namedtuple' from
        -- collections.
        pyImports = PyImport RelativePyImport "functools" :
                    PyImport RelativePyImport "multiple_dispatch" :
                    PyImportFrom AbsolutePyImport "collections" ["namedtuple"] :
                    gatherPyImports allModules
    pyPrograms <- gatherPrograms allModules
    pure $ PyPackage pyPkgName pyImports pyPrograms
  where
    moduleType :: TcModule -> MModuleType
    moduleType (Ann _ (MModule moduleTy _ _)) = moduleTy

    gatherPyImports :: [TcModule] -> [PyImport]
    gatherPyImports tcModules = S.toList $ foldl
      (\acc (Ann _ (MModule _ _ tcModuleExpr)) ->
        let newPyImports = extractImportFromModuleExpr tcModuleExpr
        in acc `S.union` newPyImports) S.empty tcModules

    extractImportFromModuleExpr :: TcModuleExpr -> S.Set PyImport
    extractImportFromModuleExpr (Ann _ moduleExpr) = case moduleExpr of
      MModuleRef v _ -> absurd v
      MModuleAsSignature v _ -> absurd v
      MModuleExternal _ _ v -> absurd v
      MModuleDef decls _ _ -> do
        let innerFoldFn acc tcDecl =
              maybe acc (`S.insert` acc) (extractExternalReference tcDecl)
        foldl (foldl innerFoldFn) S.empty decls

    extractExternalReference :: TcDecl -> Maybe PyImport
    extractExternalReference tcDecl = case tcDecl of
      MTypeDecl _ (Ann (mconDeclO, _) _) ->
        mconDeclO >>= mkPyImportFromConDeclO
      MCallableDecl _ (Ann (mconDeclO, _) _) ->
        mconDeclO >>= mkPyImportFromConDeclO

    mkPyImportFromConDeclO :: ConcreteDeclOrigin -> Maybe PyImport
    mkPyImportFromConDeclO (ConcreteExternalDecl _ extDeclDetails)
        | externalDeclBackend extDeclDetails == Python =
      let extFilePath = externalDeclFilePath extDeclDetails
          extStructString = _name $ externalDeclModuleName extDeclDetails
      in Just $
        PyImportFrom RelativePyImport extFilePath [extStructString]
    mkPyImportFromConDeclO _ = Nothing

    gatherPrograms :: [TcModule] -> MgMonad [PyModule]
    gatherPrograms = foldMAccumErrors
      (\acc -> ((:acc) <$>) . mgProgramToPyProgramModule) [] .
      filter ((== Program) . moduleType)

-- | Transforms a Magnolia program into a Python module.
mgProgramToPyProgramModule :: TcModule -> MgMonad PyModule
mgProgramToPyProgramModule
  (Ann declO (MModule Program name
                      tcModuleExpr@(Ann _ (MModuleDef decls deps _)))) =
  enter name $ do
    let boundNames = S.fromList . map _name $
          name : M.keys decls <>
          join (map (map _targetName . snd . _ann) deps)

        declsToGen = filter declFilter declsAsList
        typeDecls = getTypeDecls declsToGen
        callableDecls = getCallableDecls declsToGen

        (extObjectsNames, boundNames') = runState
          (mapM (freshObjectNameM . fst) referencedExternals)
          boundNames

        extObjectsNameMap = M.fromList $ zip referencedExternals extObjectsNames

    modulePyName <- mkPyName name

    extObjectsPyNameMap <- mapM mkPyName extObjectsNameMap
    extObjectsDef <- mapMAccumErrors
      (uncurry $ uncurry mgExtObjectToPyInstanceDef)
      (M.toList extObjectsPyNameMap)

    pyClassDefs <- mapMAccumErrors (mgTyDeclToPyClassDef extObjectsPyNameMap)
                                   typeDecls

    pyFnDefs <- mapMAccumErrorsAndFail
      (mgCallableDeclToPyDecoratedFunctionDef boundNames' extObjectsPyNameMap)
      callableDecls

    -- TODO: there will be problems if types and callables share the same
    -- name string here, since namespaces will not be separated in python.
    -- One way could be to generate closures for each type, such that the
    -- namespaces are separated in some way. This will be fixed later.
    let allDefs = extObjectsDef <>
          (map (uncurry PyClassDef) pyClassDefs <>
           map (uncurry PyDecoratedFunctionDef) pyFnDefs)
        uniqueDeclNamesInScope = S.toList . S.fromList $ map nodeName declsToGen
        namedTupleClassDefExpr = mkNamedTupleClass (_name name)
          (map _name uniqueDeclNamesInScope)
        namedTupleClassName = evalState
          (freshNameM (UnspecName "__namedtuple")) boundNames'

    namedTupleClassPyName <- mkPyName namedTupleClassName
    namedTupleInstanceFields <- mapM ((PyVarRef <$>) . mkPyName)
      uniqueDeclNamesInScope

    topSortedDefsAsStmtList <- map PyLocalDef <$>
      topSortPyDefs (srcCtx declO) allDefs

    let namedTupleDefStmt = PyLocalDef $
          PyClassDef namedTupleClassPyName namedTupleClassDefExpr
        namedTupleReturnStmt = PyReturn . Just $
          PyCall (PyFnCall namedTupleClassPyName namedTupleInstanceFields [])
        pyModuleConstructorBody = PyStmtBlock . NE.fromList $
          pyModuleBaseStmt : topSortedDefsAsStmtList <>
          [namedTupleDefStmt, namedTupleReturnStmt]
        pyModuleConstructor =
          PyFunction modulePyName [] PyNamedTuple pyModuleConstructorBody

    pure $ PyModule modulePyName pyModuleConstructor
  where
    declFilter :: TcDecl -> Bool
    declFilter decl = case decl of
      MTypeDecl _ _ -> True
      -- Only generate callables that are not assumed built-in.
      MCallableDecl _ (Ann _ callable) -> not $ isBuiltin callable

    isBuiltin :: MCallableDecl' p -> Bool
    isBuiltin cdecl = case _callableBody cdecl of BuiltinBody -> True
                                                  _ -> False

    declsAsList :: [TcDecl]
    declsAsList = join $ M.elems decls

    referencedExternals :: [(Name, [Requirement])]
    referencedExternals = gatherUniqueExternalRequirements tcModuleExpr

mgProgramToPyProgramModule _ = error "expected program"

-- | Sorts topologically a list of Python definitions based on their
-- dependencies.
topSortPyDefs :: SrcCtx -- ^ a source location for error reporting
              -> [PyDef]
              -> MgMonad [PyDef]
topSortPyDefs src defs = do
  mapM checkAcyclic sccs
  where
    sccs =  let snd3 (_, b, _) = b
                -- TODO(not that important): check if the assessment below is
                -- correct.
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

    extractVarNamesFromExpr :: PyExpr -> S.Set PyName
    extractVarNamesFromExpr pyExpr = case pyExpr of
      PyCall (PyFnCall pyFnName pyFnArgs pyFnKwargs) ->
        let varNamesFromArgs = foldl S.union S.empty
              (map extractVarNamesFromExpr pyFnArgs)
            varNamesFromKwargs = foldl S.union S.empty
              (map (extractVarNamesFromExpr . snd) pyFnKwargs)
        in S.singleton pyFnName `S.union`
           varNamesFromArgs     `S.union`
           varNamesFromKwargs
      PyVarRef pyName ->
        S.singleton $ fromMaybe pyName (extractParentObjectName pyName)
      PyIfExpr cond trueExpr falseExpr ->
        extractVarNamesFromExpr cond `S.union`
        extractVarNamesFromExpr trueExpr `S.union`
        extractVarNamesFromExpr falseExpr
      PyUnOp _ pyExpr' -> extractVarNamesFromExpr pyExpr'
      PyBinOp _ lhsExpr rhsExpr -> extractVarNamesFromExpr lhsExpr `S.union`
                                   extractVarNamesFromExpr rhsExpr
      PyTrue -> S.empty
      PyFalse -> S.empty
      PyString _ -> S.empty
      PyEmptyDict -> S.empty
      PyList exprs -> foldl S.union S.empty (map extractVarNamesFromExpr exprs)

    extractIncidenceGraphInformation :: PyDef -> (PyDef, PyName, [PyName])
    extractIncidenceGraphInformation def = case def of
      PyClassDef targetName paramExpr ->
        ( def
        , targetName
        , S.toList $ extractVarNamesFromExpr paramExpr
        )
      PyFunctionDef (PyFunction pyFnName _ _ _) ->
        ( def
        , pyFnName
        , []
        )
      PyDecoratedFunctionDef
          (PyFnCall pyDecoratorName pyDecoratorArgs pyDecoratorKwargs)
          (PyFunction pyFnName _ _ _) ->
        ( def
        , pyFnName
        , [pyDecoratorName] <>
          S.toList (foldl S.union S.empty
                          (map extractVarNamesFromExpr pyDecoratorArgs)) <>
          S.toList (foldl S.union S.empty
                          (map (extractVarNamesFromExpr . snd)
                               pyDecoratorKwargs))
        )
      PyInstanceDef pyInstanceName pyExpr ->
        ( def
        , pyInstanceName
        , S.toList $ extractVarNamesFromExpr pyExpr
        )

    checkAcyclic :: G.SCC PyDef
                 -> MgMonad PyDef
    checkAcyclic comp = case comp of
      G.AcyclicSCC def -> pure def
      G.CyclicSCC circularDepDefs ->
        let extractName def = case def of
              PyClassDef pyTargetName _  -> pyTargetName
              PyFunctionDef pyFn -> _pyFnName pyFn
              PyDecoratedFunctionDef _ pyFn -> _pyFnName pyFn
              PyInstanceDef pyTargetName _ -> pyTargetName
        in throwLocatedE MiscErr src $
              "can not sort elements " <>
              T.intercalate ", " (map (pshow . extractName) circularDepDefs) <>
              " topologically"


mgExtObjectToPyInstanceDef :: Name
                           -> [Requirement]
                           -> PyName
                           -> MgMonad PyDef
mgExtObjectToPyInstanceDef structName requirements targetPyName = do
  structPyName <- mkPyName structName
  pyArgs <- mapM ((PyVarRef <$>) . mkPyName . nodeName . _parameterDecl) $
    L.sortBy orderRequirements requirements
  let pyFnCall = PyCall $ PyFnCall structPyName pyArgs []
  pure $ PyInstanceDef targetPyName pyFnCall


mgTyDeclToPyClassDef :: M.Map (Name, [Requirement]) PyName
                     -> TcTypeDecl
                     -> MgMonad (PyName, PyExpr)
mgTyDeclToPyClassDef extObjectsMap (Ann (conDeclO, absDeclOs)
                                        (Type targetTyName)) = do
  let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
      extStructName = externalDeclModuleName extDeclDetails
      extTyName = externalDeclElementName extDeclDetails
      extOrderedRequirements = orderedRequirements extDeclDetails

  pyExtObjectName <-
    case M.lookup (extStructName, extOrderedRequirements) extObjectsMap of
      Nothing -> throwLocatedE CompilerErr (srcCtx $ NE.head absDeclOs) $
        "could not find matching external object definition for " <>
        pshow extStructName <> " with requirements " <>
        pshow (map _parameterDecl extOrderedRequirements)
      Just pyObjectName -> pure pyObjectName

  checkPyBackend (srcCtx $ NE.head absDeclOs) "type" extDeclDetails

  pyTargetTyName <- mkPyName targetTyName
  pyExtTyName <- mkPyName extTyName

  pure ( pyTargetTyName
       , PyVarRef $ mkPyObjectMemberAccess pyExtObjectName pyExtTyName
       )


mgCallableDeclToPyDecoratedFunctionDef
  :: BoundNames
  -> M.Map (Name, [Requirement]) PyName
  -> TcCallableDecl
  -> MgMonad (PyFnCall, PyFunctionDef)
mgCallableDeclToPyDecoratedFunctionDef _ extObjectsMap
  (Ann (conDeclO, absDeclOs)
       (Callable cty name args retTy _ (ExternalBody _))) = do
    -- This case is hit when one of the API functions we want to expose
    -- is declared externally, and potentially renamed.
    -- In this case, to generate the API function, we need to perform
    -- an inline call to the external function.
    -- We do not simply generate something like 'f = e.g' with 'f' our function
    -- name and 'e.g' the function 'g' in the external 'e', because the
    -- '__name__' property of 'f' would still equal 'g', which would cause
    -- trouble when resolving overloaded calls to overloaded functions.
    let ~(Just (ConcreteExternalDecl _ extDeclDetails)) = conDeclO
        extStructName = externalDeclModuleName extDeclDetails
        extCallableName = externalDeclElementName extDeclDetails
        extOrderedRequirements = orderedRequirements extDeclDetails

    pyExtObjectName <-
      case M.lookup (extStructName, extOrderedRequirements) extObjectsMap of
        Nothing -> throwLocatedE CompilerErr (srcCtx $ NE.head absDeclOs) $
          "could not find matching external object definition for " <>
          pshow extStructName <> " with requirements " <>
          pshow (map _parameterDecl extOrderedRequirements)
        Just pyObjectName -> pure pyObjectName

    checkPyBackend (srcCtx $ NE.head absDeclOs) "function" extDeclDetails

    pyFnName <- mkPyName name
    pyRetTy <- case cty of
      Function -> mkPyType retTy
      Predicate -> mkPyType Pred
      _ -> mkPyType Unit -- Axiom and procedures
    pyParams <- mapM mgTypedVarToPy args
    pyExtFnName <- mkPyObjectMemberAccess pyExtObjectName <$>
      mkPyName extCallableName

    pyArgExprs <- mapM ((PyVarRef <$>) . mkPyName . nodeName) args
    pyArgTypeNames <- mapM (mkPyName . _varType . _elem) args

    let pyBody = PyStmtBlock
          (PyReturn (Just (PyCall $ PyFnCall pyExtFnName pyArgExprs [])) :| [])
        pyFnDef = PyFunction pyFnName pyParams pyRetTy pyBody
        pyReturnTypeName = pyConstructorFromType pyRetTy
        pyDecorator = overloadWith pyArgTypeNames pyReturnTypeName

    pure (pyDecorator, pyFnDef)

mgCallableDeclToPyDecoratedFunctionDef
  boundNames _ mgFn@(Ann _ (Callable cty name args retTy mguard _)) = do
  pyFnName <- mkPyName name
  pyBody <- mgFnBodyToPyStmtBlock boundNames mgFn
  pyRetTy <- case cty of
    Function -> mkPyType retTy
    Predicate -> mkPyType Pred
    _ -> mkPyType Unit -- Axiom and procedures
  pyParams <- mapM mgTypedVarToPy args
  -- TODO: what do we do with guard? Carry it in and use it as a
  -- precondition test?

  pyArgTypeNames <- mapM (mkPyName . _varType . _elem) args

  let pyFnDef = PyFunction pyFnName pyParams pyRetTy pyBody
      pyReturnTypeName = pyConstructorFromType pyRetTy
      pyDecorator = overloadWith pyArgTypeNames pyReturnTypeName

  pure (pyDecorator, pyFnDef)


mgTypedVarToPy :: TcTypedVar -> MgMonad PyVar
mgTypedVarToPy (Ann _ v) =
  PyVar <$> mkPyName (_varName v) <*> mkPyType (_varType v)

mgFnBodyToPyStmtBlock :: BoundNames -> TcCallableDecl -> MgMonad PyStmtBlock
mgFnBodyToPyStmtBlock boundNames
                      (Ann _ (Callable _ name args _ _ inBody)) =
  go inBody
    where
      argScope :: ArgScope
      argScope = S.fromList $ map nodeName args

      go :: CBody PhCheck -> MgMonad PyStmtBlock
      go body = case body of
        EmptyBody -> throwNonLocatedE CompilerErr $
          "attempted to generate implementation code for unimplemented " <>
          "callable " <> pshow name
        ExternalBody () -> throwNonLocatedE CompilerErr $
          "attempted to generate implementation code for external callable " <>
          pshow name
        BuiltinBody -> throwNonLocatedE CompilerErr $
          "attempted to generate implementation code for builtin callable " <>
          pshow name
        MagnoliaBody (Ann _ (MBlockExpr _ exprs)) -> do
          let localBoundNames = boundNames `S.union`
                foldl S.union (S.fromList (map boundNameFromVar args))
                              (NE.map gatherLocallyBoundNames exprs)
              (exprs', closureMaps) = evalState (NE.unzip <$>
                  mapM mgReplaceValueBlocksWithClosureCalls exprs
                ) localBoundNames
              aggregateClosureMap = foldl M.union M.empty closureMaps
              -- TODO: define closures
          pyClosures <- mapM (\(closureName, closureInfo) -> do
              pyClosureName <- mkPyName closureName
              mgValueBlockExprToPyClosure pyClosureName closureInfo argScope)
            (M.toList aggregateClosureMap)
          pyStmts <- mapM (mgExprToPyStmt argScope) exprs'
          pure $ PyStmtBlock (NE.fromList (pyClosures <> NE.toList pyStmts))
        MagnoliaBody tcExpr@(Ann ty _) ->
          case ty of
            Unit -> go (MagnoliaBody
              (Ann ty (MBlockExpr MEffectfulBlock (tcExpr :| []))))
            _ -> go (MagnoliaBody
              (Ann ty (MBlockExpr MValueBlock (Ann ty (MValue tcExpr) :| []))))

      gatherLocallyBoundNames :: TcExpr -> BoundNames
      gatherLocallyBoundNames (Ann _ expr) = case expr of
        MVar {} -> S.empty
        MCall _ args' _ ->
          foldl S.union S.empty (map gatherLocallyBoundNames args')
        MBlockExpr _ blockExprs ->
          foldl S.union S.empty (NE.map gatherLocallyBoundNames blockExprs)
        MValue expr' -> gatherLocallyBoundNames expr'
        MLet var massignmentExpr -> S.insert (boundNameFromVar var) $
          maybe S.empty gatherLocallyBoundNames massignmentExpr
        MIf cond trueExpr falseExpr ->
          gatherLocallyBoundNames cond `S.union`
          gatherLocallyBoundNames trueExpr `S.union`
          gatherLocallyBoundNames falseExpr
        MAssert expr' -> gatherLocallyBoundNames expr'
        MSkip -> S.empty

      boundNameFromVar = _name . _varName . _elem


-- | Replaces all the value blocks in an expression with calls to closures, and
-- returns the modified expression along with a map from the name of the
-- generated closures to their expected body and type.
mgReplaceValueBlocksWithClosureCalls
  :: Monad m
  => TcExpr
  -> StateT BoundNames m (TcExpr, M.Map Name (NE.NonEmpty TcExpr, MType))
mgReplaceValueBlocksWithClosureCalls tcExpr@(Ann ty _) = case _elem tcExpr of
  MVar _ -> unmodified
  MCall name args mreturnType -> do
    (newArgs, nameMaps) <- unzip <$>
      mapM mgReplaceValueBlocksWithClosureCalls args
    let aggregateMap = foldl M.union M.empty nameMaps
    pure (Ann ty (MCall name newArgs mreturnType), aggregateMap)
  MBlockExpr MValueBlock blockExprs -> do
    (newBlockExprs, nameMaps) <- NE.unzip <$>
      mapM mgReplaceValueBlocksWithClosureCalls blockExprs
    closureName <- freshNameM (GenName "__generated_closure")
    let aggregateMap = foldl M.union M.empty nameMaps
        closureCall = Ann ty (MCall closureName [] (Just ty))
    pure (closureCall, M.insert closureName (newBlockExprs, ty) aggregateMap)
  MBlockExpr MEffectfulBlock _ -> unmodified
  MValue expr -> do
    (expr', closureNameMap) <- mgReplaceValueBlocksWithClosureCalls expr
    pure (Ann ty (MValue expr'), closureNameMap)
  MLet var massignmentExpr -> do
    case massignmentExpr of
      Nothing -> pure (tcExpr, M.empty)
      Just assignmentExpr -> do
        (assignmentExpr', closureNameMap) <-
          mgReplaceValueBlocksWithClosureCalls assignmentExpr
        pure (Ann ty (MLet var (Just assignmentExpr')), closureNameMap)
  -- TODO: trueExpr and falseExpr, if they are value blocks, do not need
  --       need to be transformed into a closure. With that said, for the sake
  --       of simplicity (and to avoid defining another IR), we still convert
  --       them to closures at the moment. This allows to just error when
  --       we encounter value blocks in further functions.
  MIf cond trueExpr falseExpr -> do
    (cond', m1) <- mgReplaceValueBlocksWithClosureCalls cond
    (trueExpr', m2) <- mgReplaceValueBlocksWithClosureCalls trueExpr
    (falseExpr', m3) <- mgReplaceValueBlocksWithClosureCalls falseExpr
    pure (Ann ty (MIf cond' trueExpr' falseExpr'), M.union m1 (M.union m2 m3))
  MAssert assertExpr -> do
    (assertExpr', closureNameMap) <-
      mgReplaceValueBlocksWithClosureCalls assertExpr
    pure (Ann ty (MAssert assertExpr'), closureNameMap)
  MSkip -> unmodified
  where
    unmodified = pure (tcExpr, M.empty)


mgValueBlockExprToPyClosure :: PyName
                            -> (NE.NonEmpty TcExpr, MType)
                            -> ArgScope
                            -> MgMonad PyStmt
mgValueBlockExprToPyClosure pyClosureName (blockExprs, returnType) argScope = do
  pyStmts <- PyStmtBlock <$> mapM (mgExprToPyStmt argScope) blockExprs
  pyReturnType <- mkPyType returnType
  pure $ PyLocalDef $
    PyFunctionDef (PyFunction pyClosureName [] pyReturnType pyStmts)

-- | Transforms a Magnolia expression into a block statement.
mgExprToPyStmt :: ArgScope
               -> TcExpr
               -> MgMonad PyStmt
mgExprToPyStmt argScope annInExpr@(Ann _ inExpr) = case inExpr of
  MVar _ -> PyExpr <$> goExpr annInExpr
  MCall (ProcName "_=_") [Ann tyLhs (MVar v), rhs@(Ann tyRhs _)] _ ->
    if tyLhs == tyRhs
    then let varName = _varName $ _elem v
         in do
           pyVarName <- mkPyName varName
           pyRhs <- goExpr rhs
           if varName `S.member` argScope
           then pure $ PyMutate pyVarName pyRhs
           else pure $ PyAssign pyVarName pyRhs
    else PyExpr <$> goExpr annInExpr
  MCall {} -> PyExpr <$> goExpr annInExpr
  -- TODO: effectful blocks can never return a usable value. In essence, we
  --       should be able to rely on the fact that they can not occur nested
  --       in confusing situations, i.e. it is always valid to emit a Python
  --       nested block for each effectful block encountered. We achieve that
  --       by simply building a 'if True: block else: pass' construction. This
  --       is not pretty and deadline-driven. Will be fixed later. In case the
  --       block is empty, we just remove
  MBlockExpr MEffectfulBlock stmts -> do
    case stmts of
      Ann _ MSkip :| [] -> pure PySkip
      _ -> do pyStmts <- PyStmtBlock <$> mapM goStmt stmts
              pure $ PyIf PyTrue pyStmts Nothing
  MBlockExpr MValueBlock _ -> throwNonLocatedE CompilerErr $
    "value block expression was not converted to call to closure in Python " <>
    "code generation"
  MValue expr -> PyReturn . Just <$> goExpr expr
  MLet (Ann ty var) mrhsExpr -> case mrhsExpr of
    Nothing -> do
      pyVar <- PyVar <$> mkPyName (_varName var) <*> mkPyType ty
      pyConstructorName <- mkPyName ty
      let pyRhsExpr = PyCall $ PyFnCall pyConstructorName [] []
      pure $ PyVarDecl pyVar pyRhsExpr
    Just rhsExpr -> do
      pyVar <- PyVar <$> mkPyName (_varName var) <*> mkPyType ty
      pyRhsExpr <- goExpr rhsExpr
      pure $ PyVarDecl pyVar pyRhsExpr
  MIf cond trueStmt falseStmt ->
    PyIf <$> goExpr cond <*> goStmtBlock trueStmt <*>
      (Just <$> goStmtBlock falseStmt)
  MAssert expr -> PyAssert <$> goExpr expr
  MSkip -> pure PySkip
  where
    goStmtBlock = (PyStmtBlock . (:| []) <$>) . goStmt
    goStmt = mgExprToPyStmt argScope
    goExpr = mgExprToPyExpr

mgExprToPyExpr :: TcExpr -> MgMonad PyExpr
mgExprToPyExpr (Ann returnType inExpr) = case inExpr of
  MVar (Ann _ v) -> PyVarRef <$> mkPyName (_varName v)
  MCall name args _ -> do
    mpySpecialOpExpr <- tryMgCallToPySpecialOpExpr name args returnType
    case mpySpecialOpExpr of
      Just pySpecialOpExpr -> pure pySpecialOpExpr
      Nothing -> do
        pyFnName <- mkPyName name
        pyArgs <- mapM goExpr args
        pyKwarg <- returnTypeKwarg . PyVarRef . pyConstructorFromType <$>
          mkPyType returnType
        pure $ PyCall $ PyFnCall pyFnName pyArgs [pyKwarg]
  MBlockExpr {} -> throwNonLocatedE CompilerErr $
    "block expression was not converted to call to closure in Python " <>
    "code generation, and was in a setting where an expression was " <>
    "expected"
  MValue expr -> goExpr expr
  MLet {} -> throwNonLocatedE CompilerErr
    "let statement found in a setting where an expression was expected"
  MIf cond trueExpr falseExpr ->
    PyIfExpr <$> goExpr cond <*> goExpr trueExpr <*> goExpr falseExpr
  MAssert _ -> throwNonLocatedE CompilerErr
    "assertion found in a setting where an expression was expected"
  MSkip -> throwNonLocatedE CompilerErr
    "skip statement found in a setting where an expression was expected"
  where
    goExpr = mgExprToPyExpr


-- | Takes the name of a function, its arguments and its return type, and
-- produces a unop or binop expression node if it can be expressed as such in
-- Python. For the moment, functions that can be expressed like that include the
-- equality predicate between two elements of the same type, predicate
-- combinators (such as '_&&_' and '_||_'), and the boolean constants 'TRUE',
-- and 'FALSE'.
tryMgCallToPySpecialOpExpr :: Name -> [TcExpr] -> MType
                           -> MgMonad (Maybe PyExpr)
tryMgCallToPySpecialOpExpr name args retTy = do
  pyArgs <- mapM mgExprToPyExpr args
  case (pyArgs, retTy : map _ann args ) of
    ([pyExpr], [Pred, Pred]) -> pure $ unPredCombinator pyExpr
    ([pyLhsExpr, pyRhsExpr], [Pred, Pred, Pred]) ->
      pure $ binPredCombinator pyLhsExpr pyRhsExpr
    ([pyLhsExpr, pyRhsExpr], [Pred, a, b]) -> return $
      if a == b && name == FuncName "_==_"
      then pure $ PyBinOp PyEqual pyLhsExpr pyRhsExpr
      else Nothing
    ([], [Pred]) -> pure constPred
    _ -> return Nothing
  where
    constPred = case name of FuncName "FALSE" -> Just PyFalse
                             FuncName "TRUE"  -> Just PyTrue
                             _      -> Nothing

    unPredCombinator pyExpr =
      let mPyUnOp = case name of
            FuncName "!_" -> Just PyLogicalNot
            _ -> Nothing
      in mPyUnOp >>= \op -> Just $ PyUnOp op pyExpr

    binPredCombinator pyLhsExpr pyRhsExpr =
      let mPyBinOp = case name of
            FuncName "_&&_" -> Just PyLogicalAnd
            FuncName "_||_" -> Just PyLogicalOr
            _ -> Nothing
      in mPyBinOp >>= \op -> Just $ PyBinOp op pyLhsExpr pyRhsExpr