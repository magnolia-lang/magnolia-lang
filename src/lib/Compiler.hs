module Compiler (
    -- * Compiler configuration-related data structures
    CompilerMode (..)
  , Config (..)
  , WriteToFsBehavior (..)
  , Pass (..)
    -- * Compiler mode-related utils
    -- ** Test mode
  , runTestWith
  , logErrs
    -- ** Build mode
  , runCompileWith
  )
  where

import Control.Applicative
import Control.Monad (when)
import qualified Data.List as L
import Data.Maybe (fromMaybe)
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import System.Directory (createDirectoryIfMissing, doesFileExist, doesPathExist)
import System.FilePath (takeDirectory)
import System.IO.Error (alreadyExistsErrorType, ioeSetErrorType)

import Env
import Cxx.Syntax
import Magnolia.PPrint
import Magnolia.Syntax
import Python.Syntax
import Make
import Monad

data CompilerMode = ReplMode
                  | BuildMode Config FilePath
                  | TestMode Config FilePath

-- | Configuration for the compiler.
data Config = Config { -- | Up to (and including) what pass the compiler should
                       -- run.
                       _configPass :: Pass
                       -- | What backend the compiler should emit code for if
                       -- relevant.
                     , _configBackend :: Backend
                       -- | What the output directory of the generated code
                       -- should be (if relevant). Must be set for code
                       -- generation.
                     , _configOutputDirectory :: Maybe FilePath
                       -- | What the base directory for imports of the
                       -- generated code should be (if relevant). If the field
                       -- is needed but has been left unset, defaults to
                       -- '_configOutputDirectory'.
                     , _configImportBaseDirectory :: Maybe FilePath
                       -- | What the behavior of the compiler should be when
                       -- attempting to write to the file system.
                     , _configWriteToFsBehavior :: WriteToFsBehavior
                       -- | What concepts contain the rewriting rules to use
                       -- for the equational rewriting pass.
                     , _configEquationsLocation :: [FullyQualifiedName ]
                       -- | What programs should be rewritten using the
                       -- rewriting rules.
                     , _configProgramsToRewrite :: [FullyQualifiedName]
                     }

-- | Compiler passes
data Pass = CheckPass
          | DepAnalPass
          | EquationalRewritingPass
          | ParsePass
          | SelfContainedProgramCodegenPass
          | StructurePreservingCodegenPass

instance Show Pass where
  show pass = case pass of
    CheckPass -> "check"
    DepAnalPass -> "dependency analysis"
    EquationalRewritingPass -> "equational rewriting"
    ParsePass -> "parse"
    SelfContainedProgramCodegenPass -> "self-contained code generation"
    StructurePreservingCodegenPass -> "structure preserving code generation"

data WriteToFsBehavior = OverwriteTargetFiles | WriteIfDoesNotExist

-- === test mode-related utils ===

-- | Runs the compiler up to the pass specified in the test configuration, and
-- logs the errors encountered (or the generated code if relevant) to standard
-- output.
runTestWith :: FilePath -> Config -> IO ()
runTestWith filePath config = case _configPass config of
  DepAnalPass -> runAndLogErrs $ depAnalPass filePath
  ParsePass -> runAndLogErrs $ depAnalPass filePath >>= parsePass
  CheckPass -> runAndLogErrs $ depAnalPass filePath >>= parsePass >>= checkPass
  EquationalRewritingPass ->
    let equationsLocation = _configEquationsLocation config
        programsToRewrite = _configProgramsToRewrite config
    in runAndLogErrs $ depAnalPass filePath >>= parsePass >>= checkPass
        >>= rewritePass equationsLocation programsToRewrite
  -- TODO(bchetioui): add rewrite pass in codegen?
  SelfContainedProgramCodegenPass -> case _configBackend config of
    Cxx -> do
      (ecxxPkgs, errs) <- runMgMonad $ compileToCxxPackages filePath
      case ecxxPkgs of
        Left () -> logErrs errs
        Right cxxPkgs -> do
          let cxxPkgsWithModules =
                filter (not . null . _cxxPackageModules) cxxPkgs
              sortedPkgs = L.sortOn _cxxPackageName cxxPkgsWithModules
              outDir = _configImportBaseDirectory config <|>
                       _configOutputDirectory config
          mapM_ (pprintCxxPackage outDir CxxHeader) sortedPkgs
          mapM_ (pprintCxxPackage outDir CxxImplementation) sortedPkgs
    Python -> do
      (epyPkgs, errs) <- runMgMonad $ compileToPyPackages filePath
      case epyPkgs of
        Left () -> logErrs errs
        Right pyPkgs -> do
          let pyPkgsWithModules =
                filter (not . null . _pyPackageModules) pyPkgs
              sortedPkgs = L.sortOn _pyPackageName pyPkgsWithModules
              outDir = _configImportBaseDirectory config <|>
                       _configOutputDirectory config
          mapM_ (pprintPyPackage outDir) sortedPkgs
    _ -> fail "codegen not yet implemented"
  StructurePreservingCodegenPass ->
    fail "structure preserving codegen not yet implemented"

logErrs :: S.Set Err -> IO ()
logErrs errs = pprintList (L.sort (S.toList errs))

runAndLogErrs :: MgMonad a -> IO ()
runAndLogErrs m = runMgMonad m >>= \(_, errs) -> logErrs errs

-- === build mode-related utils ===

runCompileWith :: FilePath -> Config -> IO ()
runCompileWith filePath config = case _configOutputDirectory config of
  Nothing -> fail "output directory must be set"
  Just outDir -> case (_configBackend config, _configPass config) of
    (Cxx, SelfContainedProgramCodegenPass) -> do
      (ecxxPkgs, errs) <- runMgMonad $ compileToCxxPackages filePath
      case ecxxPkgs of
        Left () -> logErrs errs >> fail "compilation failed"
        Right cxxPkgs -> do
          let cxxPkgsWithModules =
                filter (not . null . _cxxPackageModules) cxxPkgs
          createBasePath outDir
          mapM_ (writeCxxPackage outDir) cxxPkgsWithModules
    (Python, SelfContainedProgramCodegenPass) -> do
      (epyPkgs, errs) <- runMgMonad $ compileToPyPackages filePath
      case epyPkgs of
        Left () -> logErrs errs >> fail "compilation failed"
        Right pyPkgs -> do
          let pyPkgsWithModules =
                filter (not . null . _pyPackageModules) pyPkgs
          createBasePath outDir
          mapM_ (writePyPackage outDir) pyPkgsWithModules
    (_, StructurePreservingCodegenPass) ->
      fail $ show StructurePreservingCodegenPass <> " not yet implemented"
    (_, pass) -> fail $ "unexpected pass \"" <> show pass <> "\" in build mode"
  where
    createBasePath :: FilePath -> IO ()
    createBasePath basePath = do
      fileExists <- doesFileExist basePath
      when fileExists $ ioError $
        fileExistsError $ "output directory \"" <> basePath <>
          "\" can not be created. Delete the file manually and try again, " <>
          "or provide a different output directory"
      createDirectoryIfMissing True basePath

    fileExistsError s = ioeSetErrorType (userError s) alreadyExistsErrorType

    writeCxxPackage :: FilePath -> CxxPackage -> IO ()
    writeCxxPackage baseOutPath cxxPkg = do
      let pathToTargetNoExt = baseOutPath <> "/" <>
            map (\c -> if c == '.' then '/' else c)
                (_name $ _cxxPackageName cxxPkg)
      -- Here, we do not need to check for configuration any way. We only allow
      -- overwriting leave files, so this operation will fail if a file exists
      -- where we want to create a directory.
      createDirectoryIfMissing True (takeDirectory pathToTargetNoExt)
      let pathToHeaderFile = pathToTargetNoExt <> ".hpp"
          pathToImplementationFile = pathToTargetNoExt <> ".cpp"
          baseIncludePath =
            fromMaybe baseOutPath (_configImportBaseDirectory config)
          headerFileContent = T.unpack $
            pshowCxxPackage (Just baseIncludePath) CxxHeader cxxPkg
          implementationFileContent = T.unpack $
            pshowCxxPackage (Just baseIncludePath) CxxImplementation cxxPkg
          writeFn = case _configWriteToFsBehavior config of
            WriteIfDoesNotExist -> writeIfNotExists
            OverwriteTargetFiles -> writeFile
      writeFn pathToHeaderFile headerFileContent
      writeFn pathToImplementationFile implementationFileContent

    -- TODO: carefully test this frontend as well.
    writePyPackage :: FilePath -> PyPackage -> IO ()
    writePyPackage baseOutPath pyPkg = do
      let pathToTarget = baseOutPath <> "/" <>
            map (\c -> if c == '.' then '/' else c)
                (_name $ _pyPackageName pyPkg) <> ".py"
      -- Here, we do not need to check for configuration any way. We only allow
      -- overwriting leave files, so this operation will fail if a file exists
      -- where we want to create a directory.
      createDirectoryIfMissing True (takeDirectory pathToTarget)
      let baseIncludePath =
            fromMaybe baseOutPath (_configImportBaseDirectory config)
          fileContent = T.unpack $
            pshowPyPackage (Just baseIncludePath) pyPkg
          writeFn = case _configWriteToFsBehavior config of
            WriteIfDoesNotExist -> writeIfNotExists
            OverwriteTargetFiles -> writeFile
      writeFn pathToTarget fileContent

    writeIfNotExists :: FilePath -> String -> IO ()
    writeIfNotExists path content = do
      pathExists <- doesPathExist path
      when pathExists $ ioError $
        fileExistsError $ "could not write to \"" <> path <> "\""
      writeFile path content

-- === common utils ===

-- TODO(bchetioui): add rewriting step in compilation steps

compileToCxxPackages :: FilePath -> MgMonad [CxxPackage]
compileToCxxPackages filePath =
  depAnalPass filePath >>= parsePass >>= checkPass >>= programCodegenCxxPass

compileToPyPackages :: FilePath -> MgMonad [PyPackage]
compileToPyPackages filePath =
  depAnalPass filePath >>= parsePass >>= checkPass >>= programCodegenPyPass