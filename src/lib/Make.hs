{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}

module Make (
    -- * Passes
    checkPass
  , depAnalPass
  , parsePass
  )
  where

import Control.Monad (foldM, unless)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Text.Lazy as T

--import Cxx.Syntax
import Env
import Magnolia.Check
import Magnolia.Parser
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
--import MgToCxx

type TcGlobalEnv = Env (MPackage PhCheck)

data CxxPackage

-- -- === passes ===

depAnalPass :: FilePath -> MgMonad [PackageHead]
depAnalPass filepath = loadDependencyGraph filepath >>= detectCycle

--type CxxCodegenEnv = M.Map FullyQualifiedName (CxxModule, [(TcTypeDecl, CxxName)])

-- upsweepAndCodegen :: [G.SCC PackageHead] -> MgMonad TcGlobalEnv
-- upsweepAndCodegen = (fst <$>) . foldMAccumErrorsAndFail go (M.empty, M.empty)
--   where
--     -- TODO: keep going?
--     go
--       :: (TcGlobalEnv, CxxCodegenEnv)
--       -> G.SCC PackageHead
--       -> MgMonad (TcGlobalEnv, CxxCodegenEnv)
--     go _ (G.CyclicSCC pkgHead) =
--       let pCycle = T.intercalate ", " $ map (pshow . _packageHeadName) pkgHead
--       in throwNonLocatedE CyclicErr pCycle

--     go (globalEnv, inCxxModules) (G.AcyclicSCC pkgHead) =
--       let pkgName = fromFullyQualifiedName (_packageHeadName pkgHead)
--       in enter pkgName $ do
--         Ann ann (MPackage name decls deps) <-
--           parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
--         -- TODO: add local/external ann for nodes
--         importedEnv <- foldM (loadDependency globalEnv) M.empty deps
--         -- 1. Renamings BROKEN
--         envWithRenamings <- upsweepNamedRenamings importedEnv $
--           topSortTopLevelE name (getNamedRenamings decls)
--         -- TODO: deal with renamings first, then modules, then satisfactions
--         -- 2. Modules
--         (tcModulesNOTFINISHED, outCxxModules) <- upsweepAndCodegenModules
--             pkgName (envWithRenamings, inCxxModules) $
--             topSortTopLevelE name (getModules decls)
--         -- 3. Satisfactions
--         -- TODO: ^
--         -- TODO: deal with deps and other tld types
--         let tcPackage = Ann ann (MPackage name tcModulesNOTFINISHED [])
--         return (M.insert name tcPackage globalEnv, outCxxModules)

parsePass :: [PackageHead] -> MgMonad [ParsedPackage]
parsePass pkgHeads =
  let parseFn pkgHead = parsePackage (_packageHeadPath pkgHead)
                                     (_packageHeadFileContent pkgHead)
  in mapM parseFn pkgHeads

-- | This function performs type and consistency checkings of a list of
-- parsed packages. It is assumed that the packages passed as a parameter to
-- checkPass are topologically sorted, so that each package is checked after
-- its dependencies.
checkPass :: [ParsedPackage] -> MgMonad TcGlobalEnv
checkPass = foldMAccumErrorsAndFail go M.empty
  where
    go :: TcGlobalEnv -> ParsedPackage -> MgMonad TcGlobalEnv
    go env parsedPkg = do
      tcPkg <- checkPackage env parsedPkg
      return $ M.insert (nodeName parsedPkg) tcPkg env

codegenCxxPass :: TcGlobalEnv -> MgMonad (M.Map Name CxxPackage)
codegenCxxPass tcPkgs = do
  sortedPkgs <- detectCycle tcPkgsSccs
  foldMAccumErrors go M.empty sortedPkgs
  where
    tcPkgsSccs = G.stronglyConnComp
      [ ( tcPkg
        , nodeName tcPkg
        , map _targetName $ dependencies tcPkg
        )
        | tcPkg <- map snd $ M.toList tcPkgs
      ]

    go :: M.Map Name CxxPackage
       -> TcPackage
       -> MgMonad (M.Map Name CxxPackage)
    go = undefined

-- upsweepAndCodegenModules
--   :: Name
--   -> (Env [TcTopLevelDecl], CxxCodegenEnv)
--   -> [G.SCC (MModule PhParse)]
--   -> MgMonad (Env [TcTopLevelDecl], CxxCodegenEnv)
-- upsweepAndCodegenModules pkgName = foldMAccumErrors go
--   where
--     go
--       :: (Env [TcTopLevelDecl], CxxCodegenEnv)
--       -> G.SCC (MModule PhParse)
--       -> MgMonad (Env [TcTopLevelDecl], CxxCodegenEnv)
--     go _ (G.CyclicSCC modules) =
--       let mCycle = T.intercalate ", " $ map (pshow . nodeName) modules in
--       throwNonLocatedE CyclicErr mCycle

--     -- TODO: error catching & recovery
--     go (env, cxxModules) (G.AcyclicSCC modul) = do
--       tcModule <- checkModule env modul
--       let fqModuleName = FullyQualifiedName (Just pkgName) (nodeName modul)
--       moduleCxx <- mgToCxx (fqModuleName, tcModule) cxxModules
--       return ( M.insertWith (<>) (nodeName modul) [MModuleDecl tcModule] env
--              , M.insert fqModuleName moduleCxx cxxModules)


-- | Checks if a list of strongly connected components contains cycles.
-- Does not fail, but logs an error for each detected cycle. The resulting
-- list contains exactly all the elements that were part of an acyclic strongly
-- connected component.
detectCycle :: (HasSrcCtx a, HasName a) => [G.SCC a] -> MgMonad [a]
detectCycle = (L.reverse <$>) . foldMAccumErrors
  (\acc c -> (:acc) <$> checkNoCycle c) []

-- TODO: cache and choose what to reload with granularity
-- | Takes the path towards a Magnolia package and constructs its dependency
-- graph, which is returned as a topologically sorted list of strongly
-- connected components (of package heads).
--
-- For example, assuming a Magnolia package "A" depends on a Magnolia package
-- "B", the result will be sorted like ["B", "A"].
loadDependencyGraph :: FilePath -> MgMonad [G.SCC PackageHead]
loadDependencyGraph = (topSortPackageHeads <$>) . recover (go M.empty)
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
    topSortPackageHeads pkgHeads = G.stronglyConnComp
        [ ( pkgHead
          , _packageHeadName pkgHead
          , _packageHeadImports pkgHead
          )
        | (_, pkgHead) <- M.toList pkgHeads
        ]