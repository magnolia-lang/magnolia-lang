{-# LANGUAGE OverloadedStrings #-}

module Make (
  load, upsweep) where

import Control.Monad.Except
import qualified Data.Graph as G
import qualified Data.Map as M
import qualified Data.Text.Lazy as T

import Check
import Env
import Parser
import PPrint
import Syntax

upsweep :: [G.SCC PackageHead] -> ExceptT Err IO (GlobalEnv PhCheck)
upsweep = foldM go M.empty
  where
    -- TODO: keep going?
    go
      :: GlobalEnv PhCheck
      -> G.SCC PackageHead
      -> ExceptT Err IO (GlobalEnv PhCheck)
    go _ (G.CyclicSCC pkgHeads) =
      let pCycle = T.intercalate ", " $ map (pshow . _packageHeadName) pkgHeads in
      throwError $ NoCtx $ "Found cyclic dependency between the following " <>
            "packages: [" <> pCycle <> "]."

    go globalEnv (G.AcyclicSCC pkgHead) = do
      Ann ann (UPackage name decls deps) <-
        parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
      -- TODO: add local/external ann for nodes
      importedEnv <- foldM (loadPackageDependency globalEnv) M.empty deps
      -- TODO: deal with renamings first, then modules, then satisfactions
      -- TODO: make UTopLevelDecl map instead of TCModule
      checkedModulesNOTFINISHED <-
        upsweepModules importedEnv $ topSortModules (getModules decls)

      -- TODO: deal with deps and other tld types
      let checkedPackage = Ann ann (UPackage name checkedModulesNOTFINISHED [])
      return $ M.insert name checkedPackage globalEnv

    topSortModules :: [UModule PhParse] -> [G.SCC (UModule PhParse)]
    topSortModules modules = G.stronglyConnComp
        [ ( modul
          , nodeName modul
          , map nodeName (_moduleDeps (_elem modul))
          )
        | modul <- modules
        ]

    getModules :: [UTopLevelDecl p] -> [UModule p]
    getModules decls = foldl extractModule [] decls

    extractModule :: [UModule p] -> UTopLevelDecl p -> [UModule p]
    extractModule acc topLevelDecl
      | UModuleDecl m <- topLevelDecl = m:acc
      | otherwise = acc

    loadPackageDependency
      :: GlobalEnv PhCheck
      -> Env [TCTopLevelDecl]
      -> UPackageDep PhParse
      -> ExceptT Err IO (Env [TCTopLevelDecl])
    loadPackageDependency globalEnv localEnv (Ann src dep) =
      case M.lookup (nodeName dep) globalEnv of
        Nothing -> throwError $ WithSrc src ("Attempted to load package " <>
          pshow (nodeName dep) <> " but package couldn't be found.")
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
      UNamedRenamingDecl (Ann (src, LocalDecl) node) ->
        UNamedRenamingDecl (Ann (src, ImportedDecl name (nodeName node)) node):acc
      UModuleDecl (Ann (src, LocalDecl) node) ->
        UModuleDecl (Ann (src, ImportedDecl name (nodeName node)) node):acc
      USatisfactionDecl (Ann (src, LocalDecl) node) ->
        USatisfactionDecl (Ann (src, ImportedDecl name (nodeName node)) node):acc
      _ -> acc -- Ann (src, ImportedDecl name (nodeName modul)) modul:acc


upsweepModules
  :: Env [TCTopLevelDecl]
  -> [G.SCC (UModule PhParse)]
  -> ExceptT Err IO (Env [TCTopLevelDecl])
upsweepModules = foldM go
  where
    go
      :: Env [TCTopLevelDecl]
      -> G.SCC (UModule PhParse)
      -> ExceptT Err IO (Env [TCTopLevelDecl])
    go _ (G.CyclicSCC modules) =
      let mCycle = T.intercalate ", " $ map (pshow . _moduleName . _elem) modules in
      throwError $ NoCtx $ "Found cyclic dependency between the following " <>
          "modules: [" <> mCycle <> "]."

    -- TODO: error catching
    go env (G.AcyclicSCC modul) =
      let eitherNewEnv = runExcept $ checkModule env modul in
      case eitherNewEnv of
        Left e -> lift (pprint e) >> throwError e
        Right newEnv -> return newEnv

-- TODO: cache and choose what to reload with granularity
load :: FilePath -> ExceptT Err IO [G.SCC PackageHead]
load = (topSortPackages <$>) . go M.empty
  where
    go loadedHeads filePath = case M.lookup filePath loadedHeads of
      Just _ -> return loadedHeads
      Nothing -> do input <- lift $ readFile filePath
                    packageHead <- parsePackageHead filePath input
                    let newHeads = M.insert filePath packageHead loadedHeads
                        imports = map _fromSrc (_packageHeadImports packageHead)
                    lift $ pprint imports
                    foldM (\state imp -> go state (_name imp)) newHeads imports
    topSortPackages pkgHeads = G.stronglyConnComp
        [ (ph, pk, map (_name . _fromSrc) $ _packageHeadImports ph)
        | (pk, ph) <- M.toList pkgHeads
        ]
