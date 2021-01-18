import Control.Monad (foldM)
import Control.Monad.Except (runExcept)
import qualified Data.Map as M
import System.Environment (getArgs)

import Check
import EmitPy
import Env
import Parser
import PPrint
import Syntax

main :: IO ()
main = do
  srcFilename <- head <$> getArgs
  input <- readFile srcFilename
  case runExcept $ parsePackage srcFilename input of
    Left e -> putStrLn e >> error "Shouldn't happen!!"
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
