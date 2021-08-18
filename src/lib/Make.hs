{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}

module Make (
    -- * Passes
    checkPass
  , depAnalPass
  , parsePass
  , programCodegenCxxPass
  )
  where

import Control.Monad (foldM, unless)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M

import Cxx.Syntax
import Env
import Magnolia.Check
import Magnolia.Parser
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import MgToCxx

-- -- === passes ===

depAnalPass :: FilePath -> MgMonad [PackageHead]
depAnalPass filepath = loadDependencyGraph filepath >>= detectCycle

parsePass :: [PackageHead] -> MgMonad [ParsedPackage]
parsePass pkgHeads =
  let parseFn pkgHead = parsePackage (_packageHeadPath pkgHead)
                                     (_packageHeadFileContent pkgHead)
  in mapM parseFn pkgHeads

-- | This function performs type and consistency checkings of a list of
-- parsed packages. It is assumed that the packages passed as a parameter to
-- checkPass are topologically sorted, so that each package is checked after
-- its dependencies.
checkPass :: [ParsedPackage] -> MgMonad (Env TcPackage)
checkPass = foldMAccumErrorsAndFail go M.empty
  where
    go :: Env TcPackage -> ParsedPackage -> MgMonad (Env TcPackage)
    go env parsedPkg = do
      tcPkg <- checkPackage env parsedPkg
      return $ M.insert (getName parsedPkg) tcPkg env

-- | This function takes in a Map of typechecked packages, and produces, for
-- each of them, a C++ package containing all the programs defined in
programCodegenCxxPass :: Env TcPackage -> MgMonad [CxxPackage]
programCodegenCxxPass = foldMAccumErrorsAndFail
  (\acc -> ((:acc) <$>) . mgPackageToCxxSelfContainedProgramPackage) []

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