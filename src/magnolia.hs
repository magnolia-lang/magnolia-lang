import Control.Monad (foldM)
import Control.Monad.Except (runExceptT, runExcept)
import qualified Data.Map as M
import System.Environment (getArgs)

import Check
import EmitPy
import Env
import Make
import Parser
import PPrint
import Syntax

main :: IO ()
main = do
  srcFilename <- head <$> getArgs
  input <- readFile srcFilename
  runExceptT (load srcFilename >>= upsweep) >>= pprint
  --parsePackageHead srcFilename input
  error "end"
  {--
  eitherModules <- runExceptT $ parsePackage srcFilename input
  case eitherModules of
    Left e -> pprint e >> error "Shouldn't happen!!"
    Right modules -> foldM checker pkg modules >>= (putStrLn . emitPyPackage)
  where
    checker :: TCPackage -> UModule PhParse -> IO TCPackage
    checker pkg mod = do
      case runExcept $ checkModule pkg mod of
        Left e  -> putStr "Failed with: " >> pprint e >> return pkg
        Right pkg' -> return pkg'
    pkg :: TCPackage
    pkg = M.empty
  --putStrLn $ show $ checkModule scope (NoCtx obj)
  --}
