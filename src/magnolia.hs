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
  eitherGlobalEnv <-runExceptT (load srcFilename >>= upsweep)
  case eitherGlobalEnv of
    Left e -> pprint e >> error "Shouldn't happen!!"
    Right globalEnv -> case M.lookup (PkgName srcFilename) globalEnv of
      Nothing  -> putStrLn "Compiler bug!" -- TODO: handle dir paths better
      Just pkg -> pprint pkg >> putStrLn (emitPyPackage pkg)
