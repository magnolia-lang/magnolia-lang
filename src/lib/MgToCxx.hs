{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module MgToCxx (
  mgToCxx) where

import Control.Monad (foldM, unless)
import Control.Monad.IO.Class (liftIO)
import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isJust, isNothing)
import qualified Data.Set as S
import Data.Void (absurd)

import Cxx.Syntax
import Env
import Magnolia.PPrint -- TODO: move global pprint utils out of this
import Magnolia.Syntax
import Magnolia.Util

-- TODO: should we use something like BoundName?
-- | Wraps a name
-- newtype BoundName = BoundName Name

-- instance Eq BoundName where
--   BoundName (Name _ s1) == BoundName (Name _ s2) = s1 == s2

--mgPackageToCxx
--  :: M.Map FullyQualifiedName (CxxModule, [(TcTypeDecl, CxxName)])
--  -> MgMonad (CxxModule, [(TcTypeDecl, CxxName)])

mgToCxx
  :: (FullyQualifiedName, MModule PhCheck)
  -> M.Map FullyQualifiedName (CxxModule, [(TcTypeDecl, CxxName)])
  -> MgMonad (CxxModule, [(TcTypeDecl, CxxName)])
mgToCxx (_, Ann _ (RefModule _ _ v)) _ = absurd v
mgToCxx (moduleFqName, Ann declO (MModule _ name decls deps)) genModules = do
  unless (_targetName moduleFqName == name) $
    throwLocatedE CompilerErr (srcCtx declO) $ "mismatched fully qualified " <>
      "name and module name in code generation: expected " <> pshow name <>
      " but got " <> pshow moduleFqName
  moduleNamespaces <- mkCxxNamespaces moduleFqName
  moduleName <- mkCxxName name
  let boundNames = S.fromList . map _name $
        name : M.keys decls <> map nodeName deps

      declsToGen = M.foldr (\vs acc -> acc <> filter declFilter vs) [] decls
      typeDecls = getTypeDecls declsToGen
      callableDecls = getCallableDecls declsToGen

      (boundNames', typeNameAliases) =
        L.mapAccumL registerFreshName boundNames (map nodeName typeDecls)

      (boundNames'', returnTypeOverloadsNameAliasMap) =
        foldl (\(boundNs, nameMap) rTyO ->
              let (boundNs', newName) = registerFreshName boundNs (fst rTyO)
              in (boundNs', M.insert rTyO newName nameMap))
          (boundNames', M.empty) (S.toList returnTypeOverloads)

  declsAndCxxTemplateNames <- L.sortOn (_name . nodeName . fst) <$>
    (mapM mkCxxName typeNameAliases >>= \tna -> return $ zip typeDecls tna)

  let (innerTemplateParameters, outerTemplateParameters) =
        L.partition (_typeIsRequired . _elem . fst) declsAndCxxTemplateNames
      -- TODO: make types always sorted to set types easily?

  moduleRefSpecializations <-
    mapM (mgDepToCxxRefSpec genModules (M.fromList declsAndCxxTemplateNames))
         deps

  cxxTyDefs <- mapM mgTyNameToCxxTypeDef declsAndCxxTemplateNames
  -- TODO: emit overloaded functions properly
  cxxFnDefs <- mapM (mgCallableDeclToCxx returnTypeOverloadsNameAliasMap)
                    callableDecls

  -- TODO: properly handle defs, this is a test
  defs <- map (CxxPublic,) <$> (((cxxTyDefs <> cxxFnDefs) <>) <$>
      mgReturnTypeOverloadsToCxxTemplatedDefs boundNames'' returnTypeOverloadsNameAliasMap)
  let x = CxxModule moduleNamespaces
                     moduleName
                     moduleRefSpecializations
                     (map snd outerTemplateParameters)
                     (map snd innerTemplateParameters)
                     defs
  liftIO $ pprint x
  return ( CxxModule moduleNamespaces
                     moduleName
                     moduleRefSpecializations
                     (map snd outerTemplateParameters)
                     (map snd innerTemplateParameters)
                     defs
          , declsAndCxxTemplateNames
          )
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


-- | Produces a specialization of a templated C++ module based on a Magnolia
-- module dependency and (TODO fix) the mapping between the type declarations
-- of the corresponding Magnolia module and their corresponding template type
-- names within the corresponding C++ module.
mgDepToCxxRefSpec
  :: M.Map FullyQualifiedName (CxxModule, [(TcTypeDecl, CxxName)])
  -> M.Map TcTypeDecl CxxName
  -> TcModuleDep
  -> MgMonad CxxModuleRefSpecialization
mgDepToCxxRefSpec genModules declsToCxxTemplateTyNames (Ann _ dep) = do
  let ~(Just (cxxModule, tyDeclToCxxName)) = M.lookup (_depName dep) genModules
      tyDecls = map fst tyDeclToCxxName
  renamedTys <- foldM (\tys rblock -> mapM (rename rblock) tys) tyDecls
                      (_depRenamingBlocks dep)
  cxxTemplateTyNames <- mapM lookupTyCxxName renamedTys
  return $ CxxModuleRefSpecialization (_cxxModuleNamespaces cxxModule)
    (_cxxModuleName cxxModule) cxxTemplateTyNames
  where
    rename :: TcRenamingBlock -> TcTypeDecl -> MgMonad TcTypeDecl
    rename (Ann _ (MRenamingBlock renamings)) tyDecl@(Ann _ ty@(Type name _)) =
      case filter (renamingMatchesTyName name) renamings of
        [] -> return tyDecl
        [Ann _ (InlineRenaming (_, Name _ newStr))] -> return $
          tyDecl {_elem = ty { _typeName = Name (_namespace name) newStr } }
        _ -> throwNonLocatedE CompilerErr "TODO: error message"

    renamingMatchesTyName :: Name -> TcRenaming -> Bool
    renamingMatchesTyName (Name _ tyStr) renaming = case _elem renaming of
      InlineRenaming (Name _ srcStr, _) -> tyStr == srcStr
      _ -> False

    lookupTyCxxName :: TcTypeDecl -> MgMonad CxxName
    lookupTyCxxName tyDecl = case M.lookup tyDecl declsToCxxTemplateTyNames of
      Nothing -> throwNonLocatedE CompilerErr $
        "attempted to lookup a C++ name for " <> pshow tyDecl <> " but " <>
        "could not find a match while building a C++ module specialization"
      Just cxxName -> return cxxName


-- TODO: move fresh names to utils?
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

mgTyNameToCxxTypeDef :: (TcTypeDecl, CxxName) -> MgMonad CxxDef
mgTyNameToCxxTypeDef (Ann _ (Type targetTyName _), cxxSourceTyName) = do
  -- TODO: deal with outside required types names
  cxxTargetTyName <- mkCxxName targetTyName
  return $ CxxTypeDef cxxSourceTyName cxxTargetTyName

-- | mgCallableDeclToCxx assumes the function is not templated on return type.
-- TODO: clean up?
mgCallableDeclToCxx
  :: M.Map (Name, [MType]) Name
  -> TcCallableDecl
  -> MgMonad CxxDef
mgCallableDeclToCxx returnTypeOverloadsNameAliasMap
                    mgFn@(Ann _ (Callable cty name args retTy mguard cbody))
  | isOverloadedOnReturnType = do
      mgFn' <- mutify mgFn
      mgCallableDeclToCxx returnTypeOverloadsNameAliasMap mgFn'
  | otherwise = do
      cxxFnName <- mkCxxName name
      cxxBody <- mgFnBodyToCxxStmtBlock returnTypeOverloadsNameAliasMap mgFn
      let isVirtual = isNothing cxxBody
      (hasSideEffects, cxxRetTy) <- case cty of
          Function -> (False,) <$> mkCxxType retTy
          Predicate -> (False,) <$> mkCxxType Pred
          _ -> (True,) <$> mkCxxType Unit -- Axiom and procedures
      cxxParams <- mapM mgTypedVarToCxx args
      -- TODO: deal with overloaded functions, also in AST
      -- TODO: what do we do with guard? Carry it in and use it as a
      -- precondition test?
      return . CxxFunctionDef $
        CxxFunction isVirtual hasSideEffects cxxFnName [] cxxParams cxxRetTy
                    cxxBody
  where
    mgTypedVarToCxx :: TypedVar PhCheck -> MgMonad CxxVar
    mgTypedVarToCxx (Ann _ v) = do
      cxxVarName <- mkCxxName $ _varName v
      cxxVarType <- mkCxxType $ _varType v
      return $ CxxVar (_varMode v == MObs) True cxxVarName cxxVarType

    isOverloadedOnReturnType :: Bool
    isOverloadedOnReturnType = isJust $
      M.lookup (name, map (_varType . _elem) args)
               returnTypeOverloadsNameAliasMap

    mutify :: TcCallableDecl -> MgMonad TcCallableDecl
    mutify (Ann ann c) = mutifyBody >>= \mutifiedBody -> return $ Ann ann
      c { _callableType = Procedure
        , _callableName = ProcName (_name name)
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
      _ -> return cbody --throwNonLocatedE CompilerErr $ "what " <> pshow mgFn

-- TODO: what happens with procedures that return nothing re. overloading?
-- In Magnolia, whichever one we call is unambiguous. In C++, that might
-- be different. Let's see.

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
      cxxArgTypes <- mapM mkCxxType argTypes
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
              -- mutifiedFnName<T>(a0, a1, …, an, &o);
            , CxxExpr (CxxCall mutifiedFnCxxName [cxxTemplateTyName]
                               cxxCallArgExprs)
              -- return o;
            , CxxReturn (Just (CxxVarRef outVarCxxName))
            ]
      return . CxxFunctionDef $
        CxxFunction { _cxxFnIsVirtual = False
                    , _cxxFnHasSideEffects = True
                    , _cxxFnName = cxxFnName
                    , _cxxFnTemplateParameters = [cxxTemplateTyName]
                    , _cxxFnParams = cxxFnParams
                    , _cxxFnReturnType = cxxTemplateTy
                    , _cxxFnBody = Just cxxFnBody
                    }


-- mgExprToCxxStmtBlockBody
mgFnBodyToCxxStmtBlock :: M.Map (Name, [MType]) Name -> TcCallableDecl
                       -> MgMonad (Maybe CxxStmtBlock)
mgFnBodyToCxxStmtBlock returnTypeOverloadsNameAliasMap
                       (Ann declO (Callable _ name _ _ _ body)) = case body of
  EmptyBody -> return Nothing
  -- TODO: remake into an error.
  ExternalBody -> return Nothing --throwLocatedE CompilerErr (srcCtx $ NE.head declO) $
  --    "attempted to generate code for external callable " <> pshow name
  -- TODO: this is not extremely pretty. Can we easily do something better?
  MagnoliaBody (Ann _ (MBlockExpr _ exprs)) ->
    Just <$> mapM (mgExprToCxxStmt returnTypeOverloadsNameAliasMap)
                  (NE.toList exprs)
  MagnoliaBody expr -> Just . (:[]) <$>
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
        cxxVarType <- mkCxxType vty
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
        cxxName <- mkCxxName name
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