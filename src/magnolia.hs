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

import Cxx.Syntax
import Compiler
import Env
import Make
import Magnolia.Parser
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import Monad

type TopEnv = Env (MPackage PhCheck)

-- === parsing utils ===

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
      (info (helper <*> (BuildMode <$> buildConfig <*> target))
            (progDesc "Compile a package"))

    testCmd = command "test"
      (info (helper <*> (TestMode <$> testConfig <*> target))
            (progDesc "Test the compiler passes"))

    target = argument str (metavar "FILE" <> help "Source program")

    buildConfig = Config SelfContainedProgramCodegenPass <$>
      optBackend <*>
      option (Just <$> str)
             (long "output-directory" <>
              short 'o' <>
              help "Output directory for code generation") <*>
      optImportDir <*>
      flag WriteIfDoesNotExist OverwriteTargetFiles
           (long "allow-overwrite" <>
            help "Allow overwriting target files in the output directory") <*>
      optEquationsLocation <*>
      optProgramsToRewrite

    testConfig = Config <$>
      option (oneOf passOpts)
             (long "pass" <> value CheckPass <>
              help ("Pass: " <> dispOpts passOpts)) <*>
      optBackend <*>
      option (Just <$> str)
             (long "output-directory" <> value Nothing <>
              short 'o' <>
              help "Output directory for code generation") <*>
      optImportDir <*>
      pure WriteIfDoesNotExist <*>
      optEquationsLocation <*>
      optProgramsToRewrite

    optBackend =
      option (oneOf backendOpts)
             (long "backend" <> value Cxx <>
              help ("Backend: " <> dispOpts backendOpts))

    optImportDir =
      option (Just <$> str)
             (long "base-import-directory" <> value Nothing <>
              short 'i' <>
              help ("Base import directory for the generated code " <>
                    "(defaults to the output directory)"))

    optEquationsLocation =
      option (map toFullyQualifiedModuleName . splitOn ',' <$> str)
             (long "equations-location" <> value [] <>
              short 'e' <>
              help ("Comma-separated list of fully qualified concept names " <>
                    "in which to look for rewriting rules"))

    optProgramsToRewrite =
      option (map toFullyQualifiedModuleName . splitOn ',' <$> str)
             (long "programs-to-rewrite" <> value [] <>
              short 'r' <>
              help ("Comma-separated list of fully qualified program names " <>
                    "on which to apply rewriting rules"))

    splitOn :: Eq a => a -> [a] -> [[a]]
    splitOn sep list = case list of
      [] -> [[]]
      (a:as) -> if a == sep
        then []:splitOn sep as
        else let (as':ass) = splitOn sep as in (a:as'):ass

    toFullyQualifiedModuleName :: String -> FullyQualifiedName
    toFullyQualifiedModuleName s = case splitOn '.' s of
      [] -> error "unreachable code" -- can not happen?
      [moduleString] -> FullyQualifiedName Nothing (ModName moduleString)
      fullyQualifiedString -> FullyQualifiedName
        (Just $ PkgName (L.intercalate "." (init fullyQualifiedString)))
        (ModName $ last fullyQualifiedString)

    passOpts = [ ("check", CheckPass)
               , ("self-contained-codegen", SelfContainedProgramCodegenPass)
               , ("depanal", DepAnalPass)
               , ("parse", ParsePass)
               , ("structured-codegen", StructurePreservingCodegenPass)
               , ("rewrite", EquationalRewritingPass)
               ]
    backendOpts = [ ("cpp", Cxx)
                  , ("javascript", JavaScript)
                  , ("python", Python)
                  ]

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
    ReplMode -> putStrLn ("WARNING: the repl is outdated, and likely going " <>
                          "towards deprecation")
                >> repl `runStateT` M.empty
                >> return ()
    BuildMode config filePath -> filePath `runCompileWith` config
    TestMode config filePath -> filePath `runTestWith` config

-- === repl-related utils ===

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
