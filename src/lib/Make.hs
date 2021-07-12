{-# LANGUAGE OverloadedStrings #-}

module Make (loadDependencyGraph, upsweep, TcGlobalEnv) where

import Control.Monad (foldM, unless)
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

type TcGlobalEnv = Env (MPackage PhCheck)

-- TODO: recover when tcing?
upsweep :: [G.SCC PackageHead] -> MgMonad TcGlobalEnv
upsweep = foldMAccumErrorsAndFail go M.empty
  where
    -- TODO: keep going?
    go :: TcGlobalEnv
       -> G.SCC PackageHead
       -> MgMonad TcGlobalEnv
    go _ (G.CyclicSCC pkgHeads) =
      let pCycle = T.intercalate ", " $ map (pshow . _packageHeadName) pkgHeads
      in throwNonLocatedE CyclicPackageErr pCycle

    go globalEnv (G.AcyclicSCC pkgHead) =
      enter (fromFullyQualifiedName (_packageHeadName pkgHead)) $ do
        Ann pkgAnn (MPackage name decls deps) <-
          parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
        let allModules = getModules decls
            allNamedRenamings = getNamedRenamings decls
        -- TODO: add local/external ann for nodes
        importedEnv <- foldMAccumErrors (loadDependency globalEnv) M.empty deps
        -- 1. Renamings
        envWithRenamings <- upsweepNamedRenamings importedEnv $
          topSortTopLevelE name allNamedRenamings
        -- TODO: deal with renamings first, then modules, then satisfactions
        -- 2. Modules
        tcModulesNOTFINISHED <- upsweepModules envWithRenamings $
          topSortTopLevelE name allModules
        -- 3. Satisfactions
        -- TODO: ^
        -- TODO: deal with deps and other tld types
        let tcPackage = Ann pkgAnn (MPackage name tcModulesNOTFINISHED [])
        return $ M.insert name tcPackage globalEnv

loadDependency :: TcGlobalEnv -> Env [TcTopLevelDecl]
               -> MPackageDep PhParse -> MgMonad (Env [TcTopLevelDecl])
loadDependency globalEnv localEnv (Ann src dep) =
  case M.lookup (nodeName dep) globalEnv of
    Nothing -> throwLocatedE MiscErr src $ "attempted to load package " <>
      pshow (nodeName dep) <> " but package couldn't be found"
    Just (Ann _ pkg) -> do
      let importedLocalDecls =
            M.map (foldl (importLocal (nodeName dep)) [])
                  (_packageDecls pkg)
      return $ M.unionWith (<>) importedLocalDecls localEnv

importLocal :: Name             -- ^ Name of the package to import from
            -> [TcTopLevelDecl]
            -> TcTopLevelDecl
            -> [TcTopLevelDecl]
importLocal name acc decl = case decl of
  MNamedRenamingDecl (Ann (LocalDecl src) node) ->
    MNamedRenamingDecl (Ann (mkImportedDecl src node) node):acc
  MModuleDecl (Ann (LocalDecl src) node) ->
    MModuleDecl (Ann (mkImportedDecl src node) node):acc
  MSatisfactionDecl (Ann (LocalDecl src) node) ->
    MSatisfactionDecl (Ann (mkImportedDecl src node) node):acc
  -- We do not import non-local decls from other packages.
  _ -> acc
  where
    mkImportedDecl src node =
      ImportedDecl (FullyQualifiedName (Just name) (nodeName node)) src

-- | Checks and expands a list of topologically sorted named renamings, and then
-- adds them to an initial top level environment.
-- TODO: optimize as needed, could be more elegant.
upsweepNamedRenamings
  :: Env [TcTopLevelDecl]             -- ^ The starting top level environment
  -> [G.SCC (MNamedRenaming PhParse)] -- ^ The named renamings to check
  -> MgMonad (Env [TcTopLevelDecl])
upsweepNamedRenamings = foldMAccumErrors go
  where
    go
      :: Env [TcTopLevelDecl]
      -> G.SCC (MNamedRenaming PhParse)
      -> MgMonad (Env [TcTopLevelDecl])
    go _ (G.CyclicSCC namedBlock) =
      let rCycle = T.intercalate ", " $ map (pshow . nodeName) namedBlock in
      throwNonLocatedE CyclicNamedRenamingErr rCycle

    go env (G.AcyclicSCC (Ann src (MNamedRenaming name renamingBlock))) = do
      expandedBlock <-
        expandRenamingBlock (M.map getNamedRenamings env) renamingBlock
        -- TODO: expand named renamings
      let tcNamedRenaming = Ann (LocalDecl src)
                                (MNamedRenaming name expandedBlock)
      return $ M.insertWith (<>) name [MNamedRenamingDecl tcNamedRenaming]
                            env

-- | Checks a list of topologically sorted modules, and then adds them to an
-- initial top level environment.
upsweepModules
  :: Env [TcTopLevelDecl]      -- ^ The starting top level environment
  -> [G.SCC (MModule PhParse)] -- ^ The modules to check
  -> MgMonad (Env [TcTopLevelDecl])
upsweepModules = foldMAccumErrors go
  where
    go
      :: Env [TcTopLevelDecl]
      -> G.SCC (MModule PhParse)
      -> MgMonad (Env [TcTopLevelDecl])
    go _ (G.CyclicSCC modules) =
      let mCycle = T.intercalate ", " $ map (pshow . nodeName) modules in
      throwNonLocatedE CyclicModuleErr mCycle

    go env (G.AcyclicSCC modul) = checkModule env modul

-- TODO: cache and choose what to reload with granularity
-- | Takes the path towards a Magnolia package and constructs its dependency
-- graph, which is returned as a topologically sorted list of strongly
-- connected components (of package heads).
--
-- For example, assuming a Magnolia package "A" depends on a Magnolia package
-- "B", the result will be sorted like ["B", "A"].
loadDependencyGraph :: FilePath -> MgMonad [G.SCC PackageHead]
loadDependencyGraph = (topSortPackages <$>) . recover (go M.empty)
  where
    go :: M.Map Name PackageHead -> FilePath -> MgMonad (M.Map Name PackageHead)
    go loadedHeads filePath = do
      unless (".mg" `L.isSuffixOf` filePath) $
        throwNonLocatedE MiscErr
          "Magnolia source code files must have the \".mg\" extension"
      let expectedPkgName = mkPkgNameFromPath filePath
      case M.lookup expectedPkgName loadedHeads of
        Just _ -> return loadedHeads
        Nothing -> do
          input <- liftIO $ readFile filePath
          packageHead <- parsePackageHead filePath input
          let pkgName = fromFullyQualifiedName $ _packageHeadName packageHead
              imports =  map (mkPkgPathFromName . fromFullyQualifiedName)
                             (_packageHeadImports packageHead)
          unless (expectedPkgName == pkgName) $
            throwNonLocatedE MiscErr $ "expected package to have " <>
              "name " <> pshow expectedPkgName <> " but got " <>
              pshow pkgName
          foldM go (M.insert pkgName packageHead loadedHeads) imports
    topSortPackages pkgHeads = G.stronglyConnComp
        [ ( pkgHead
          , _packageHeadName pkgHead
          , _packageHeadImports pkgHead
          )
        | (_, pkgHead) <- M.toList pkgHeads
        ]