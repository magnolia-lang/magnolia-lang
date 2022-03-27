{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}

module Make (
    -- * Utils
    RewritingSystemConfig (..)
  , RewritingSystemMode (..)
    -- * Passes
  , checkPass
  , depAnalPass
  , parsePass
  , programCodegenCxxPass
  , programCodegenPyPass
  , rewritePass
  )
  where

import Control.Monad (foldM, join, unless)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M

import Cxx.Syntax
import Env
import Magnolia.Check
import Magnolia.EquationalRewriting
import Magnolia.Parser
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import MgToCxx
import MgToPython
import Monad
import Python.Syntax

-- === utils ===

data RewritingSystemConfig =
  RewritingSystemConfig { -- | The name of the concept containing the rewriting
                          -- system.
                          _rewritingSystemName :: FullyQualifiedName
                          -- | The maximum number of rewriting steps the
                          -- compiler is allowed to perform per expression with
                          -- the given rewriting system.
                        , _rewritingSystemMaxRewritingSteps :: Int
                          -- | What type of operation to use the rewriting
                          -- system for.
                        , _rewritingSystemMode :: RewritingSystemMode
                        }

data RewritingSystemMode = RewritingSystemMode'Optimize
                         | RewritingSystemMode'GenerateCallable

-- === passes ===

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
      return $ M.insert (nodeName parsedPkg) tcPkg env

-- | This function takes in a Map of typechecked packages, and produces, for
-- each of them, a C++ package containing all the programs defined in the
-- input package.
programCodegenCxxPass :: Env TcPackage -> MgMonad [CxxPackage]
programCodegenCxxPass = foldMAccumErrorsAndFail
  (\acc -> ((:acc) <$>) . mgPackageToCxxSelfContainedProgramPackage) []

-- | This function takes in a Map of typechecked packages, and produces, for
-- each of them, a Python package containing all the programs defined in the
-- input package.
programCodegenPyPass :: Env TcPackage -> MgMonad [PyPackage]
programCodegenPyPass = foldMAccumErrorsAndFail
  (\acc -> ((:acc) <$>) . mgPackageToPySelfContainedProgramPackage) []

-- | This function takes in a list of concept names containing rewriting rules
-- (in the forms of equational assertions, in axioms), a list of program
-- names on which to apply them, TODO: complete
rewritePass :: [RewritingSystemConfig]
            -> [FullyQualifiedName]
            -> Env TcPackage
            -> MgMonad (Env TcPackage)
rewritePass rewritingSystemConfigs programsToRewriteLocations env = do
  --liftIO $ pprint rewritingSystemConfigs
  rewritingModules <- mapM (findModule . _rewritingSystemName)
    rewritingSystemConfigs
  programsToRewrite <- mapM findModule programsToRewriteLocations
  let rewritingModulesWithModeAndMaxSteps = zip rewritingModules $
          zip (map _rewritingSystemMode rewritingSystemConfigs)
              (map _rewritingSystemMaxRewritingSteps rewritingSystemConfigs)
      rewriteAll tgts (rewModule, (mode, maxSteps)) =
        mapM (applyRewritingModule rewModule mode maxSteps) tgts
  rewrittenProgramsWithNames <- zip programsToRewriteLocations <$>
    foldM rewriteAll programsToRewrite rewritingModulesWithModeAndMaxSteps
  foldM (\env' (fqn, program) -> insertModule program env' fqn) env
        rewrittenProgramsWithNames
  where
    applyRewritingModule :: TcModule -> RewritingSystemMode -> Int -> TcModule
                         -> MgMonad TcModule
    applyRewritingModule rewModule mode = case mode of
      RewritingSystemMode'Optimize -> runOptimizer rewModule
      RewritingSystemMode'GenerateCallable -> \_ -> runGenerator rewModule

    findModule :: FullyQualifiedName -> MgMonad TcModule
    findModule fqn = case fqn of
      FullyQualifiedName Nothing n -> invalidFullyQualifiedName n
      FullyQualifiedName (Just packageName) _ -> enter packageName $
        case M.lookup packageName env of
          Nothing -> throwNonLocatedE MiscErr $ "module " <> pshow fqn <>
            " does not exist, because package name does not exist"
          Just (Ann src package) -> lookupTopLevelRef src
            (M.map getModules (_packageDecls package)) fqn

    insertModule :: TcModule
                 -> Env TcPackage
                 -> FullyQualifiedName
                 -> MgMonad (Env TcPackage)
    insertModule newModule env' fqn = do
      oldModule <- findModule fqn
      -- 'findModule fqn' ensures that the fully qualified name contains a
      -- package name as expected. We can hence pattern match lazily below.
      let ~(FullyQualifiedName (Just packageName) moduleName) = fqn
      (Ann src oldPackage) <- case M.lookup packageName env' of
        Nothing -> throwNonLocatedE MiscErr $ "module " <> pshow fqn <>
          " does not exist, because package name does not exist"
        Just tcPackage -> pure tcPackage
      let oldDecls = _packageDecls oldPackage
          edit new old = new <> filter (MModuleDecl oldModule /=) old
          newDecls = M.insertWith edit moduleName
            [MModuleDecl newModule] oldDecls
          newPackage = Ann src $ oldPackage { _packageDecls = newDecls }
      pure $ M.insert packageName newPackage env'

    invalidFullyQualifiedName :: Name -> MgMonad a
    invalidFullyQualifiedName moduleName =
      throwNonLocatedE CompilerErr $ "expected fully qualified name in " <>
        "rewrite pass but got module name " <> pshow moduleName

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