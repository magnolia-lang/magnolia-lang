{-# LANGUAGE OverloadedStrings #-}

module Make (
  load, upsweep) where

import Control.Monad (foldM)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except
import qualified Data.Graph as G
import qualified Data.Map as M
import qualified Data.Text.Lazy as T

import Check
import Env
import Parser
import PPrint
import Syntax
import Util

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
      throwNonLocatedE $ "Found cyclic dependency between the following " <>
        "packages: [" <> pCycle <> "]."

    go globalEnv (G.AcyclicSCC pkgHead) = do
      Ann ann (UPackage name decls deps) <-
        parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
      -- TODO: add local/external ann for nodes
      importedEnv <- foldM (loadPackageDependency globalEnv) M.empty deps
      -- 1. Renamings
      envWithRenamings <-
        upsweepRenamings importedEnv $ topSortRenamingDecls (getNamedRenamings decls)
      -- TODO: deal with renamings first, then modules, then satisfactions
      -- 2. Modules
      checkedModulesNOTFINISHED <-
        upsweepModules envWithRenamings $ topSortModules (getModules decls)
      -- 3. Satisfactions
      -- TODO: ^
      -- TODO: deal with deps and other tld types
      let checkedPackage = Ann ann (UPackage name checkedModulesNOTFINISHED [])
      return $ M.insert name checkedPackage globalEnv

topSortModules :: [UModule PhParse] -> [G.SCC (UModule PhParse)]
topSortModules modules = G.stronglyConnComp
    [ ( modul
      , nodeName modul
      , moduleDepNames modul
      )
    | modul <- modules
    ]

-- TODO: does that work for imported nodes? Non-existing dependencies?
topSortRenamingDecls
  :: [UNamedRenaming PhParse] -> [G.SCC (UNamedRenaming PhParse)]
topSortRenamingDecls namedRenamings = G.stronglyConnComp
    [ ( node
      , name
      , getRenamingDependencies renamingBlocks
      )
    | node@(Ann _ (UNamedRenaming name renamingBlocks)) <- namedRenamings
    ]

getRenamingDependencies :: URenamingBlock PhParse -> [Name]
getRenamingDependencies (Ann _ (URenamingBlock renamings)) =
    foldl extractRenamingRef [] renamings

extractRenamingRef :: [Name] -> URenaming PhParse -> [Name]
extractRenamingRef acc (Ann _ r) = case r of
  InlineRenaming _ -> acc
  RefRenaming name -> name:acc

loadPackageDependency
  :: GlobalEnv PhCheck
  -> Env [TCTopLevelDecl]
  -> UPackageDep PhParse
  -> ExceptT Err IO (Env [TCTopLevelDecl])
loadPackageDependency globalEnv localEnv (Ann src dep) =
  case M.lookup (nodeName dep) globalEnv of
    Nothing -> throwLocatedE src $ "Attempted to load package " <>
      pshow (nodeName dep) <> " but package couldn't be found."
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


-- TODO: optimize as needed, could be more elegant.
-- Checks and expands renamings.
upsweepRenamings
  :: Env [TCTopLevelDecl]
  -> [G.SCC (UNamedRenaming PhParse)]
  -> ExceptT Err IO (Env [TCTopLevelDecl])
upsweepRenamings = foldM go
  where
    go
      :: Env [TCTopLevelDecl]
      -> G.SCC (UNamedRenaming PhParse)
      -> ExceptT Err IO (Env [TCTopLevelDecl])
    go _ (G.CyclicSCC namedBlock) =
      let rCycle = T.intercalate ", " $ map (pshow . nodeName) namedBlock in
      throwNonLocatedE $ "Found cyclic dependency between the following " <>
        "renamings: [" <> rCycle <> "]."
    
    go env (G.AcyclicSCC (Ann src (UNamedRenaming name renamingBlock))) =
      let eitherExpandedBlock = runExcept $
            expandRenamingBlock (M.map getNamedRenamings env) renamingBlock in -- TODO: expand named renamings
      case eitherExpandedBlock of
        Left e -> lift (pprint e) >> throwE e
        Right expandedBlock ->
          let tcNamedRenaming = Ann (src, LocalDecl)
                                    (UNamedRenaming name expandedBlock) in
          return $ M.insertWith (<>) name [UNamedRenamingDecl tcNamedRenaming]
                                env


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
      let mCycle = T.intercalate ", " $ map (pshow . nodeName) modules in
      throwNonLocatedE $ "Found cyclic dependency between the following " <>
        "modules: [" <> mCycle <> "]."

    -- TODO: error catching
    go env (G.AcyclicSCC modul) =
      let eitherNewEnv = runExcept $ checkModule env modul in
      case eitherNewEnv of
        Left e -> lift (pprint e) >> throwE e
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
