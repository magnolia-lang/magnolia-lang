{-# LANGUAGE FlexibleInstances #-}

module EmitPy (PythonSource, emitPyPackage) where

import Data.List (intercalate)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M

import Env
import Syntax

--data PythonSource = PythonSourceFile String PythonSource
--                  | PythonSourceBlock [PythonSource]
--                  | PythonSourceUnit String

type PythonSource = String

type TCExpr' = MExpr' PhCheck

-- TODO: actually implement
emitPyPackage :: TCPackage -> PythonSource
emitPyPackage pkg =
  intercalate "\n\n# Next module\n\n" $ map emitPyModule $
    foldl extractModuleList [] (_packageDecls (_elem pkg))
  where
    extractModuleList moduleList pkgDeclList =
      moduleList <> foldl extractModule [] pkgDeclList
    extractModule moduleList pkgDecl = case pkgDecl of
      MModuleDecl modul -> modul:moduleList
      _ -> moduleList

-- TODO: actually implement
emitPyModule :: TCModule -> PythonSource
emitPyModule modul = intercalate "\n\n" $ map (emitPyDecl 0) firstModuleDecls
  where firstModuleDecls = snd $ head $ M.toList (moduleDecls modul)

emitPyDecl :: Int -> TCDecl -> PythonSource
emitPyDecl indent decl = case decl of
  -- TODO: add type annotations?
  TypeDecl (Ann _ (Type _)) -> noEmit
  -- In Python, it is not possible to have side-effects on all the types of
  -- arguments. Therefore, procedures are turned into functions returning
  -- tuples.
  CallableDecl cdecl@(Ann _ (Callable callableType _ args _ _
                                      (MagnoliaBody body))) ->
      emitProto cdecl <> mkIndent bodyIndent <> case callableType of
        Axiom     -> emitPyExpr bodyIndent body <> "\n"
        -- TODO: what to do with this?
        Procedure -> emitPyExpr bodyIndent body <> mkIndent bodyIndent <>
                     emitProcReturn args <> "\n"
        -- Predicates are functions, so MCallable {Function,Predicate} requires
        -- one unique behavior.
        -- TODO: make sure the chosen return name is free.
        _         -> emitPrefixedPyExpr bodyIndent
                                        (mkIndent bodyIndent <> "return ") body
  -- Ignore prototypes and external functions
  CallableDecl (Ann _ (Callable _ _ _ _ _ EmptyBody)) -> noEmit
  CallableDecl (Ann _ (Callable _ _ _ _ _ ExternalBody)) -> noEmit
  where
    emitProcReturn :: [TCTypedVar] -> PythonSource
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
      MVar (Ann _ v) -> prefix <> emitName (_varName v)
      MCall name args _ -> prefix <> emitName name <> "(" <>
          intercalate "," (map (emitPyExpr 0) args) <> ")"
      MBlockExpr stmts ->
        intercalate ("\n" <> strIndent)
                    (map (emitPyExpr indent) (NE.init stmts)) <> "\n" <>
        emitPrefixedPyExpr indent prefix (NE.last stmts)
      -- TODO: prefix == "" or error?
      MLet _ name _ maybeAssignmentExpr -> prefix <> case maybeAssignmentExpr of
        Nothing -> emitName name <> " = None"
        Just assignmentExpr ->
          emitPrefixedPyExpr indent (emitName name <> " = ") assignmentExpr
      MIf cond bTrue bFalse ->
        emitPrefixedPyExpr indent "if " cond <> ":\n" <>
        emitPrefixedPyExpr (incIndent indent) prefix bTrue <> "\n" <>
        strIndent <> "else:\n" <>
        emitPrefixedPyExpr (incIndent indent) prefix bFalse <> "\n"
      -- TODO: prefix == "" or error?
      MAssert cond -> prefix <> emitPrefixedPyExpr indent "assert " cond
      MSkip -> "pass"
      --MTypedExpr expr' _ -> emitPyExpr indent expr'

-- === utils ===

mkIndent :: Int -> PythonSource
mkIndent = flip replicate ' '

incIndent :: Int -> Int
incIndent = (+2)

noEmit :: PythonSource
noEmit = ""

emitName :: Name -> PythonSource
emitName (Name _ s) = mkPythonSource s

emitProto :: TCCallableDecl -> PythonSource
emitProto (Ann _ (Callable _ name args _ _ _)) = "def " <> emitName name
  <> "(" <> intercalate ", " (map emitVarName args) <> "):\n"

emitVarName :: TCTypedVar -> PythonSource
emitVarName (Ann _ v) = emitName $ _varName v

mkPythonSource :: String -> PythonSource
mkPythonSource = id
