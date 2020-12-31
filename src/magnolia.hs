import Control.Monad (foldM)
import Control.Monad.Except (runExcept)
import qualified Data.Map as M
import System.Environment (getArgs)

import Check
import Env
import Parser
import PPrint
import Syntax

obj = UImpl (UnspecName "ImA") [ NoCtx $ UType $ TypeName "T"
                               , NoCtx $ UType $ TypeName "T2"
                               , NoCtx $ UFunc (FuncName "f")
                                         [ NoCtx $ Var UObs (VarName "a")
                                                   (Just (TypeName "T4"))
                                         , NoCtx $ Var UObs (VarName "b")
                                                   (Just (TypeName "T3"))
                                         , NoCtx $ Var UOut (GenName "retval#1")
                                                   (Just (TypeName "T3"))
                                         ] (Just $ NoCtx $ UCall
                                            (UnspecName "f")
                                            [ NoCtx $ UVar $ NoCtx $
                                              Var UObs (VarName "a") Nothing
                                            , NoCtx $ UVar $ NoCtx $
                                              Var UObs (VarName "a") Nothing
                                            ] Nothing)
                               ]
                               [ NoCtx $ UModuleDep (ModName "SigB")
                                         [NoCtx
                                           [ (UnspecName "T1", UnspecName "T2")
                                           , (UnspecName "T2", UnspecName "T3")
                                           , (UnspecName "T3", UnspecName "T4")
                                           , (UnspecName "T4", UnspecName "T1")
                                           ]
                                         ]]

scope :: Package
scope = M.fromList [( UnspecName "SigB"
                    , M.fromList [ ( TypeName "T1"
                                   , [NoCtx $ UType $ TypeName "T1"]
                                   )
                                 , ( TypeName "T2"
                                   , [NoCtx $ UType $ TypeName "T2"]
                                   )
                                 , ( TypeName "T3"
                                   , [NoCtx $ UType $ TypeName "T3"]
                                   )
                                 , ( TypeName "T4"
                                   , [NoCtx $ UType $ TypeName "T4"]
                                 )
                                 , ( FuncName "f"
                                   , [NoCtx $ UFunc (FuncName "f")
                                            [ NoCtx $ Var UObs (VarName "a")
                                                      (Just (TypeName "T4"))
                                            , NoCtx $ Var UObs (VarName "b")
                                                      Nothing
                                            , NoCtx $ Var UOut (GenName "c")
                                                      Nothing
                                            ] Nothing]
                                 ) ]
                   )]

pkg :: Package
pkg = M.empty

main :: IO ()
main = do
  srcFilename <- head <$> getArgs
  input <- readFile srcFilename
  case runExcept $ parsePackage srcFilename input of
    Left e -> putStrLn e >> error "Shouldn't happen!!"
    Right modules -> foldM checker pkg modules >>= print
  where
    checker :: Package -> UModule -> IO Package
    checker pkg mod = do
      case runExcept $ checkModule pkg mod of
        Left e  -> putStr "Failed with: " >> pprint e >> return pkg
        Right pkg' -> return pkg'
  --putStrLn $ show $ checkModule scope (NoCtx obj)
