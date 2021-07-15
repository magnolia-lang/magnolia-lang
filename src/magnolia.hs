import Control.Monad (join)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.State
import Data.Foldable (toList)
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Set as S
import Debug.Trace (trace)
import Options.Applicative hiding (Success, Failure)
import System.Console.Haskeline

import Env
import Make
import Magnolia.Parser
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util

type TopEnv = Env (MPackage PhCheck)
data CompilerMode = ReplMode | BuildMode FilePath | TestMode TestConfig FilePath

data TestConfig = TestConfig { _testPass :: Pass
                             , _testBackend :: Backend
                             , _outputDirectory :: Maybe FilePath
                             }

-- | Compiler passes
data Pass = CheckPass
          | DepAnalPass
          | ParsePass
          | ProgramCodegenPass
          | StructurePreservingCodegenPass

-- | Runs the compiler up to the pass specified in the test configuration, and
-- logs the errors encountered to standard output.
runTest :: TestConfig -> FilePath -> IO ()
runTest config filePath = case _testPass config of
  DepAnalPass -> runAndLogErrs $ depAnalPass filePath
  ParsePass -> runAndLogErrs $ depAnalPass filePath >>= parsePass
  CheckPass -> runAndLogErrs $ depAnalPass filePath >>= parsePass >>= checkPass
  ProgramCodegenPass -> case _testBackend config of
    Cxx -> undefined
    _ -> error "codegen not yet implemented"
  StructurePreservingCodegenPass ->
    error "structure preserving codegen not yet implemented"

-- === debugging utils ===

logErrs :: S.Set Err -> IO ()
logErrs errs = pprintList (L.sort (toList errs))

runAndLogErrs :: MgMonad a -> IO ()
runAndLogErrs m = runMgMonad m >>= \(_, errs) -> logErrs errs

-- === passes ===

mkInfo :: Parser a -> ParserInfo a
mkInfo p = info (p <**> helper) mempty

parseCompilerMode :: ParserInfo CompilerMode
parseCompilerMode = mkInfo compilerMode
  where
    compilerMode :: Parser CompilerMode
    compilerMode = subparser $ replCmd <> buildCmd <> testCmd

    replCmd = command "repl"
      (info (helper <*> pure ReplMode)
            (progDesc "Start Magnolia repl"))

    buildCmd = command "build"
      (info (helper <*> (BuildMode <$> target))
            (progDesc "Compile a package"))

    testCmd = command "test"
      (info (helper <*> (TestMode <$> testConfig <*> target))
            (progDesc "Test the compiler passes"))

    target = argument str (metavar "FILE" <> help "Source program")
    testConfig = TestConfig <$>
      option (oneOf passOpts)
             (long "pass" <> value CheckPass <>
              help ("Pass: " <> dispOpts passOpts)) <*>
      option (oneOf backendOpts)
             (long "backend" <> value Cxx <>
              help ("Backend: " <> dispOpts backendOpts)) <*>
      option (Just <$> str)
             (long "output-directory" <> value Nothing <>
              help "Output directory for code generation")

    passOpts = [ ("check", CheckPass)
               , ("codegen", ProgramCodegenPass)
               , ("depanal", DepAnalPass)
               , ("parse", ParsePass)
               , ("structured-codegen", StructurePreservingCodegenPass)
               ]
    backendOpts = [ ("cpp", Cxx)
                  , ("javascript", JavaScript)
                  , ("python", Python)]

    oneOf :: [(String, a)] -> ReadM a
    oneOf optPairs = eitherReader (\v -> case L.lookup v optPairs of
      Just someOpt -> Right someOpt
      _ -> Left $ "Option must be one of " <>
                  L.intercalate ", " (map (show . fst) optPairs))

    dispOpts opts = fst (head opts) <> " (default) | " <>
      L.intercalate " | " (map fst (tail opts))

main :: IO ()
main = do
  compilerMode <- customExecParser (prefs showHelpOnEmpty) parseCompilerMode
  case compilerMode of
    ReplMode -> repl `runStateT` M.empty >> return ()
    BuildMode filePath -> let config = TestConfig CheckPass Cxx Nothing
                          in runTest config filePath
    TestMode config filePath -> runTest config filePath


repl :: StateT TopEnv IO ()
repl = runInputT defaultSettings go
  where
    go :: InputT (StateT TopEnv IO) ()
    go = do
      minput <- getInputLine "mgn> "
      case minput of
        Nothing -> return ()
        Just "q" -> return ()
        Just input -> do
          (eitherCmd, errs) <-
            liftIO $ runMgMonad (parseReplCommand input)
          case eitherCmd of
            Left _ -> liftIO $ pprintList (L.sort (toList errs))
            Right cmd -> lift $ execCmd cmd
          go

execCmd :: Command -> StateT TopEnv IO ()
execCmd cmd = case cmd of
  LoadPackage pkgStr -> loadPackage False pkgStr
  ReloadPackage pkgStr -> loadPackage True pkgStr
  InspectModule modName -> inspectModule modName
  InspectPackage pkgName -> inspectPackage pkgName
  ListPackages -> listPackages
  ShowMenu -> showMenu
  where
    loadPackage :: Bool -> String -> StateT TopEnv IO ()
    loadPackage reload pkgStr = do
      env <- get
      let (pkgName, pkgPath) = if isPkgPath pkgStr
                               then (mkPkgNameFromPath pkgStr, pkgStr)
                               else (PkgName pkgStr, mkPkgPathFromStr pkgStr)
          pkgRef = if not reload then M.lookup pkgName env else Nothing
      case pkgRef of
        Just _ -> return ()
        Nothing -> do
          (eenv, errs) <- liftIO $ runMgMonad (depAnalPass pkgPath >>= parsePass >>= checkPass)
          case eenv of
            Left () -> liftIO (logErrs errs)
            Right newEnv -> case M.lookup pkgName newEnv of
              -- TODO: handle better package names that don't fit
              Nothing ->
                trace (show newEnv) (return ()) >> error "Compiler bug!"
              Just pkg -> modify (M.insert pkgName pkg) -- TODO: what to do with other packages?

    inspectModule :: Name -> StateT TopEnv IO ()
    inspectModule modName = do
      env <- get
      let extractPkgModules =
            getModules . join . M.elems . _packageDecls . _elem
          matches = filter (\m -> nodeName m == modName) $
            join . M.elems $ M.map extractPkgModules env
      case matches of
        []   -> liftIO $ pprint "No such module has been loaded"
        m:ms ->
          if not (null ms) then
            liftIO $ pprint "Several possible matches (TODO: allow specifying)"
          else
            liftIO $ pprint m

    inspectPackage :: Name -> StateT TopEnv IO ()
    inspectPackage pkgName = do
      env <- get
      case M.lookup pkgName env of
        Nothing -> liftIO $ pprint ("Package " <> _name pkgName <>
                                    " is not in scope")
        Just (Ann _ pkg) -> do
          let modules = getModules . join . M.elems $ _packageDecls pkg
          liftIO . putStrLn $
              L.intercalate "\n" (map (_name . nodeName) modules)

    listPackages :: StateT TopEnv IO ()
    listPackages = do
      env <- get
      liftIO $ putStrLn $ L.intercalate "\n" (map _name $ M.keys env)

    showMenu :: StateT TopEnv IO ()
    showMenu = do
      liftIO . pprint $ "Available commands are: " <>
          L.intercalate "\n\t" [ "help: show this help menu"
                               , "inspectm: inspect a module"
                               , "inspectp: inspect a package"
                               , "list: list the loaded packages"
                               , "load: load a package"
                               , "reload: reload a package"
                               ]
