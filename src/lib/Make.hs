{-# LANGUAGE OverloadedStrings #-}

module Make (loadDependencyGraph, upsweep) where

import Control.Monad (foldM, unless, when)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Text.Lazy as T

import Check
import Env
import Parser
import PPrint
import Syntax
import Util

-- TODO: recover when tcing?
upsweep :: [G.SCC PackageHead] -> MgMonad (GlobalEnv PhCheck)
upsweep = foldMAccumErrorsAndFail go M.empty
  where
    -- TODO: keep going?
    go
      :: GlobalEnv PhCheck
      -> G.SCC PackageHead
      -> MgMonad (GlobalEnv PhCheck)
    go _ (G.CyclicSCC pkgHeads) =
      let pCycle = T.intercalate ", " $ map (pshow . _packageHeadName) pkgHeads
      in throwNonLocatedE CyclicPackageErr pCycle

    go globalEnv (G.AcyclicSCC pkgHead) = do
      Ann ann (MPackage name decls deps) <-
        parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
      -- TODO: add local/external ann for nodes
      importedEnv <-
        foldMAccumErrors (loadPackageDependency globalEnv) M.empty deps
      -- 1. Renamings
      envWithRenamings <- upsweepRenamings importedEnv $
        topSortRenamingDecls name (getNamedRenamings decls)
      -- TODO: deal with renamings first, then modules, then satisfactions
      -- 2. Modules
      checkedModulesNOTFINISHED <- upsweepModules envWithRenamings $
        topSortModules name (getModules decls)
      -- 3. Satisfactions
      -- TODO: ^
      -- TODO: deal with deps and other tld types
      let checkedPackage = Ann ann (MPackage name checkedModulesNOTFINISHED [])
      return $ M.insert name checkedPackage globalEnv

topSortModules :: Name -> [MModule PhParse] -> [G.SCC (MModule PhParse)]
topSortModules pkgName modules = G.stronglyConnComp
    [ ( modul
      , nodeName modul
      , map _targetName $
          filter (isLocalFQName pkgName) $
            dependencies modul
      )
    | modul <- modules
    ]

-- TODO: does that work for imported nodes? Non-existing dependencies?
topSortRenamingDecls
  :: Name -> [MNamedRenaming PhParse] -> [G.SCC (MNamedRenaming PhParse)]
topSortRenamingDecls pkgName namedRenamings = G.stronglyConnComp
    [ ( node
      , name
      , map _targetName $
          filter (isLocalFQName pkgName) $
            getRenamingDependencies renamingBlocks
      )
    | node@(Ann _ (MNamedRenaming name renamingBlocks)) <- namedRenamings
    ]

isLocalFQName :: Name -> FullyQualifiedName -> Bool
isLocalFQName pkgName (FullyQualifiedName mscopeName _) =
  case mscopeName of Nothing -> True
                     Just scopeName -> scopeName == pkgName

getRenamingDependencies :: MRenamingBlock PhParse -> [FullyQualifiedName]
getRenamingDependencies (Ann _ (MRenamingBlock renamings)) =
    foldl extractRenamingRef [] renamings

extractRenamingRef
  :: [FullyQualifiedName] -> MRenaming PhParse -> [FullyQualifiedName]
extractRenamingRef acc (Ann _ r) = case r of
  InlineRenaming _ -> acc
  RefRenaming name -> name:acc

loadPackageDependency
  :: GlobalEnv PhCheck
  -> Env [TCTopLevelDecl]
  -> MPackageDep PhParse
  -> MgMonad (Env [TCTopLevelDecl])
loadPackageDependency globalEnv localEnv (Ann src dep) =
  case M.lookup (nodeName dep) globalEnv of
    Nothing -> throwLocatedE MiscErr src $ "attempted to load package " <>
      pshow (nodeName dep) <> " but package couldn't be found"
    Just (Ann _ pkg) -> do
      let importedLocalDecls =
            M.map (foldl (importLocal (nodeName dep)) [])
                  (_packageDecls pkg)
      return $ M.unionWith (<>) importedLocalDecls localEnv

importLocal
  :: Name -- ^ Name of the package
  -> [TCTopLevelDecl]
  -> TCTopLevelDecl
  -> [TCTopLevelDecl]
importLocal name acc decl = case decl of
  MNamedRenamingDecl (Ann (LocalDecl dty src) node) ->
    MNamedRenamingDecl (Ann (mkImportedDecl dty src node) node):acc
  MModuleDecl (Ann (LocalDecl dty src) node) ->
    MModuleDecl (Ann (mkImportedDecl dty src node) node):acc
  MSatisfactionDecl (Ann (LocalDecl dty src) node) ->
    MSatisfactionDecl (Ann (mkImportedDecl dty src node) node):acc
  -- We do not import non-local decls from other packages.
  _ -> acc -- Ann (src, ImportedDecl name (nodeName modul)) modul:acc
  where
    mkImportedDecl dty src node =
      ImportedDecl (FullyQualifiedName (Just name) (nodeName node)) dty src


-- TODO: optimize as needed, could be more elegant.
-- Checks and expands renamings.
upsweepRenamings
  :: Env [TCTopLevelDecl]
  -> [G.SCC (MNamedRenaming PhParse)]
  -> MgMonad (Env [TCTopLevelDecl])
upsweepRenamings = foldMAccumErrors go
  where
    go
      :: Env [TCTopLevelDecl]
      -> G.SCC (MNamedRenaming PhParse)
      -> MgMonad (Env [TCTopLevelDecl])
    go _ (G.CyclicSCC namedBlock) =
      let rCycle = T.intercalate ", " $ map (pshow . nodeName) namedBlock in
      throwNonLocatedE CyclicNamedRenamingErr rCycle

    go env (G.AcyclicSCC (Ann src (MNamedRenaming name renamingBlock))) = do
      expandedBlock <-
        expandRenamingBlock (M.map getNamedRenamings env) renamingBlock
        -- TODO: expand named renamings
      let tcNamedRenaming = Ann (LocalDecl ConcreteDecl src)
                                (MNamedRenaming name expandedBlock)
      return $ M.insertWith (<>) name [MNamedRenamingDecl tcNamedRenaming]
                            env


upsweepModules
  :: Env [TCTopLevelDecl]
  -> [G.SCC (MModule PhParse)]
  -> MgMonad (Env [TCTopLevelDecl])
upsweepModules = foldMAccumErrors go
  where
    go
      :: Env [TCTopLevelDecl]
      -> G.SCC (MModule PhParse)
      -> MgMonad (Env [TCTopLevelDecl])
    go _ (G.CyclicSCC modules) =
      let mCycle = T.intercalate ", " $ map (pshow . nodeName) modules in
      throwNonLocatedE CyclicModuleErr mCycle

    -- TODO: error catching & recovery
    go env (G.AcyclicSCC modul) = checkModule env modul

-- TODO: cache and choose what to reload with granularity
loadDependencyGraph :: FilePath -> MgMonad [G.SCC PackageHead]
loadDependencyGraph = recover ((topSortPackages <$>) . go M.empty)
  where
    go :: M.Map String PackageHead -> FilePath -> MgMonad (M.Map String PackageHead)
    go loadedHeads filePath = case M.lookup filePath loadedHeads of
      Just _ -> return loadedHeads
      Nothing -> do unless (".mg" `L.isSuffixOf` filePath) $
                      throwNonLocatedE MiscErr $ "Magnolia source code " <>
                        "files must have the \".mg\" extension"
                    input <- liftIO $ readFile filePath
                    packageHead <- parsePackageHead filePath input
                    let pkgStr = _name $ _packageHeadName packageHead
                        expectedPkgStr = _name $ mkPkgNameFromPath filePath
                        newHeads = M.insert filePath packageHead loadedHeads
                        -- TODO: make instance of "HasDependencies" for
                        --       PackageHead
                        imports = map (mkPkgPathFromName . _fromSrc)
                            (_packageHeadImports packageHead)
                    when (expectedPkgStr /= pkgStr) $
                      throwNonLocatedE MiscErr $ "expected package to have " <>
                        "name " <> pshow expectedPkgStr <> " but got " <>
                        pshow pkgStr
                    --lift $ pprint imports -- debug
                    foldM go newHeads imports
    topSortPackages pkgHeads = G.stronglyConnComp
        [ (ph, pk, map (_name . _fromSrc) $ _packageHeadImports ph)
        | (pk, ph) <- M.toList pkgHeads
        ]