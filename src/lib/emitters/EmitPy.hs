{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}

module EmitPy (PythonSource (..)) where

import Data.List (intercalate)
import qualified Data.List.NonEmpty as NE

import Emit
import Env
import Syntax

--data PythonSource = PythonSourceFile String PythonSource
--                  | PythonSourceBlock [PythonSource]
--                  | PythonSourceUnit String

type PythonSource = String

instance Emitter PythonSource where
  emitGlobalEnv = emitPyGlobalEnv
  emitPackage = emitPyPackage
  emitModule = emitPyModule
  emitDecl = emitPyDecl
  emitExpr = emitPyExprNoIndent

emitPyGlobalEnv :: GlobalEnv -> PythonSource
emitPyGlobalEnv = undefined

emitPyPackage :: UPackage -> PythonSource
emitPyPackage = undefined

emitPyModule :: UModule -> PythonSource
emitPyModule = undefined

emitPyDecl :: UDecl -> PythonSource
emitPyDecl = undefined

emitPyExprNoIndent :: UExpr -> PythonSource
emitPyExprNoIndent inputExpr = emitPyExpr 0 inputExpr

emitPyExpr :: Int -> UExpr -> PythonSource
emitPyExpr ind (WithSrc _ inputExpr) = emitPyExpr' ind inputExpr
  where
    emitPyExpr' :: Int -> UExpr' -> PythonSource
    emitPyExpr' indent expr = let strIndent = mkIndent indent in case expr of
      UVar (WithSrc _ v) -> emitName (_varName v)
      UCall name args _ ->
        emitName name <> "(" <> (intercalate "," (map (emitPyExpr 0) args)) <>
        ")"
      UBlockExpr stmts ->
        intercalate ("\n" <> strIndent) $
                    map (emitPyExpr indent) (NE.toList stmts)
      ULet _ name _ maybeAssignmentExpr -> case maybeAssignmentExpr of
        Nothing -> emitName name <> " = None"
        Just assignmentExpr -> mkInlineExpr indent assignmentExpr
      UIf cond bTrue bFalse ->
        "if " <> mkInlineExpr indent cond <> ":\n" <>
        emitPyExpr (incIndent indent) bTrue <> "\n" <> strIndent <> "else:" <>
        emitPyExpr (incIndent indent) bFalse <> "\n"
      UAssert cond -> "assert " <> mkInlineExpr indent cond
      USkip -> "pass"
      UTypedExpr expr' -> emitPyExpr indent expr'

    mkInlineExpr :: Int -> UExpr -> PythonSource
    mkInlineExpr indent (WithSrc _ inlineExpr) = case inlineExpr of
      UBlockExpr _ ->
        "lambda: (\n" <> mkIndent indent <>
        emitPyExpr' (incIndent $ incIndent indent) inlineExpr <> ")()"
      _            -> emitPyExpr' indent inlineExpr


mkIndent :: Int -> PythonSource
mkIndent = flip replicate ' '

incIndent :: Int -> Int
incIndent = (+2)

emitName :: Name -> PythonSource
emitName (GenName s) = error $ "Should not happen; trying to emit: " ++
                                    show s
emitName (Name _ s) = mkPythonSource s

mkPythonSource :: String -> PythonSource
mkPythonSource = id
