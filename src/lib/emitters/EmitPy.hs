{-# LANGUAGE FlexibleInstances #-}

module EmitPy (PythonSource (..), emitPyPackage) where

import Data.List (intercalate)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M

import Env
import PPrint
import Syntax

--data PythonSource = PythonSourceFile String PythonSource
--                  | PythonSourceBlock [PythonSource]
--                  | PythonSourceUnit String

type PythonSource = String

type TCDecl' = UDecl' PhCheck
type TCExpr' = UExpr' PhCheck

emitPyGlobalEnv :: GlobalEnv PhCheck -> PythonSource
emitPyGlobalEnv = undefined

-- TODO: actually implement
emitPyPackage :: TCPackage -> PythonSource
emitPyPackage pkg = emitPyModule $ snd . head $ M.toList pkg

-- TODO: actually implement
emitPyModule :: TCModule -> PythonSource
emitPyModule modul = intercalate "\n\n" $ map (emitPyDecl 0) firstModuleDecls
  where firstModuleDecls = snd $ head $ M.toList modul

emitPyDecl :: Int -> TCDecl -> PythonSource
emitPyDecl indent (Ann _ decl) = case decl of
  -- TODO: add type annotations?
  UType _ -> noEmit
  UCallable _ _ _ _ Nothing -> noEmit -- Ignore prototypes
  -- In Python, it is not possible to have side-effects on all the types of
  -- arguments. Therefore, procedures are turned into functions returning
  -- tuples.
  UCallable callableType _ args _ (Just body) ->
      emitProto decl <> mkIndent bodyIndent <> case callableType of
        Axiom     -> emitPyExpr bodyIndent body <> "\n"
        Procedure -> emitPyExpr bodyIndent body <> mkIndent bodyIndent <>
                     emitProcReturn args <> "\n"
        -- Predicates are functions, so UCallable {Function,Predicate} requires
        -- one unique behavior.
        -- TODO: make sure the chosen return name is free.
        _         -> emitPrefixedPyExpr bodyIndent
                                        (mkIndent bodyIndent <> "return ") body
  where
    emitProcReturn :: [TCVar] -> PythonSource
    emitProcReturn args =
      case filter (\(Ann _ (Var mode _ _)) -> mode /= UObs) args of
        [] -> "return None"
        rets -> "return (" <> intercalate ", " (map emitVarName rets) <> ",)"

    bodyIndent = incIndent indent

emitPyExpr :: Int -> TCExpr -> PythonSource
emitPyExpr = flip emitPrefixedPyExpr ""

emitPrefixedPyExpr :: Int -> PythonSource -> TCExpr -> PythonSource
emitPrefixedPyExpr ind prefixPySrc (Ann _ inputExpr) =
  go ind prefixPySrc inputExpr
  where
    go :: Int -> PythonSource -> TCExpr' -> PythonSource
    go indent prefix expr = let strIndent = mkIndent indent in case expr of
      UVar (Ann _ v) -> prefix <> emitName (_varName v)
      UCall name args _ -> prefix <> emitName name <> "(" <>
          intercalate "," (map (emitPyExpr 0) args) <> ")"
      UBlockExpr stmts ->
        intercalate ("\n" <> strIndent)
                    (map (emitPyExpr indent) (NE.init stmts)) <> "\n" <>
        emitPrefixedPyExpr indent prefix (NE.last stmts)
      -- TODO: prefix == "" or error?
      ULet _ name _ maybeAssignmentExpr -> prefix <> case maybeAssignmentExpr of
        Nothing -> emitName name <> " = None"
        Just assignmentExpr ->
          emitPrefixedPyExpr indent (emitName name <> " = ") assignmentExpr
      UIf cond bTrue bFalse ->
        emitPrefixedPyExpr indent "if " cond <> ":\n" <>
        emitPrefixedPyExpr (incIndent indent) prefix bTrue <> "\n" <>
        strIndent <> "else:\n" <>
        emitPrefixedPyExpr (incIndent indent) prefix bFalse <> "\n"
      -- TODO: prefix == "" or error?
      UAssert cond -> prefix <> emitPrefixedPyExpr indent "assert " cond
      USkip -> "pass"
      --UTypedExpr expr' _ -> emitPyExpr indent expr'

-- === utils ===

mkIndent :: Int -> PythonSource
mkIndent = flip replicate ' '

incIndent :: Int -> Int
incIndent = (+2)

noEmit :: PythonSource
noEmit = ""

emitName :: Name -> PythonSource
emitName (Name _ s) = mkPythonSource s

emitProto :: TCDecl' -> PythonSource
emitProto ~(UCallable _ name args _ _) = "def " <> emitName name <> "(" <>
  intercalate ", " (map emitVarName args) <> "):\n"

emitVarName :: TCVar -> PythonSource
emitVarName (Ann _ v) = emitName $ _varName v

mkPythonSource :: String -> PythonSource
mkPythonSource = id
