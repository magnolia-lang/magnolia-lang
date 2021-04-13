import Control.Monad (join)
import Control.Monad.Except (runExceptT)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.State
import qualified Data.List as L
import qualified Data.Map as M
import Debug.Trace (trace)
import Options.Applicative hiding (Success, Failure)
import System.Console.Haskeline --(InputT, runInputT)

import EmitPy
import Env
import Make
import Parser
import PPrint
import Syntax
import Util

type TopEnv = GlobalEnv PhCheck
data Status = Failure | Success -- TODO: move and expand statuses
data CompilerMode = ReplMode | BuildMode String

mkInfo :: Parser a -> ParserInfo a
mkInfo p = info (p <**> helper) mempty

parseCompilerMode :: ParserInfo CompilerMode
parseCompilerMode = mkInfo compilerMode
  where
    compilerMode :: Parser CompilerMode
    compilerMode = subparser $
         command "repl" (info (pure ReplMode) (progDesc "Start Magnolia repl"))
      <> command "build" (mkInfo (BuildMode <$>
                                    argument str (metavar "FILE"
                                                    <> help "Source program")))

main :: IO ()
main = do
  compilerMode <- execParser parseCompilerMode
  case compilerMode of
    ReplMode -> repl `runStateT` M.empty >> return ()
    BuildMode filename -> build filename >>= pprint

codegen :: String -> TopEnv -> IO String -- TODO: return instead source code type or smth
codegen filename env = case M.lookup (mkPkgNameFromPath filename) env of
  Nothing -> error $ "Compiler bug! Package for file " <> filename <>
    " not found."
  -- TODO: handle dir paths better
  Just pkg -> return (emitPyPackage pkg)

-- TODO: add existing env, and move "compile" to Make module
compile :: String -> IO (Status, TopEnv)
compile filename = do
  -- TODO: replace "ExceptT" with (Status, TopEnv) to allow partial success
  eenv <- runExceptT (load filename >>= upsweep)
  case eenv of
    Left e -> pprint e >> return (Failure, M.empty)
    Right env -> return (Success, env)

build :: String -> IO String -- TODO: return instead source code type or smth
build filename = do
  (status, compiledPackages) <- compile filename
  case status of
    Failure -> error "Exiting"
    Success -> codegen filename compiledPackages

repl :: StateT TopEnv IO () --InputT (StateT s IO) ()
repl = runInputT defaultSettings go
  where
    go :: InputT (StateT TopEnv IO) ()
    go = do
      minput <- getInputLine "mgn> "
      case minput of
        Nothing -> return ()
        Just "q" -> return ()
        Just input -> do
          mcmd <- liftIO $ runExceptT (parseReplCommand input)
          case mcmd of Left e -> liftIO $ pprint e
                       Right cmd -> lift $ execCmd cmd
          go

execCmd :: Command -> StateT TopEnv IO ()
execCmd cmd = case cmd of
  LoadPackage pkgStr -> loadPackage False pkgStr
  ReloadPackage pkgStr -> loadPackage True pkgStr --undefined
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
          (success, newEnv) <- liftIO $ compile pkgPath
          case success of
            Failure -> return ()
            Success -> case M.lookup pkgName newEnv of
              -- TODO: handle better package names that don't fit
              Nothing -> trace (show newEnv) (return ()) >> error "Compiler bug!"
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