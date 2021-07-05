{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wno-orphans #-}

module PPrint (pprint, pprintList, pshow, render) where

import Control.Monad (join)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO
import Data.Text.Prettyprint.Doc
import Data.Text.Prettyprint.Doc.Render.Text
import Data.Void (absurd)

import Env
import Syntax

-- The below pprinting and error-handling related utils are directly inspired
-- from their equivalent in https://github.com/google-research/dex-lang.

p :: Pretty a => a -> Doc ann
p = pretty

-- TODO: Text.Lazy -> Text.Strict (renderLazy -> renderStrict)
pprint :: Pretty a => a -> IO ()
pprint = printDoc . p

printDoc :: Doc ann -> IO ()
printDoc = TIO.putStrLn . render

render :: Doc ann -> T.Text
render = renderLazy . layoutPretty defaultLayoutOptions

pshow :: Pretty a => a -> T.Text
pshow = render . p

pprintList :: Pretty a => [a] -> IO ()
pprintList =  printDoc . vsep . punctuate line . map p

instance Pretty Err where -- TODO: change error to have error types
  pretty (Err errType src parentScopes txt) =
    p src <> ":" <+> p errType <> (case parentScopes of
      [] -> ""
      ps -> " in " <> concatWith (surround dot) (map p ps)) <> ":" <+> p txt

instance Pretty SrcCtx where
  pretty (SrcCtx msrcInfo) = case msrcInfo of
    Nothing -> "<unknown location>"
    Just ((filename, startLine, startColumn), _) ->
      p filename <> ":" <> p startLine <> ":" <> p startColumn

instance Pretty ErrType where
  pretty e = case e of
    AmbiguousFunctionRefErr -> ambiguous "functions"
    AmbiguousTopLevelRefErr -> ambiguous "top-level references"
    AmbiguousProcedureRefErr -> ambiguous "procedures"
    CompilerErr -> "Compiler bug!" <> line
      <> "Please report this at github.com/magnolia-lang/magnolia-lang"
      <> line
    CyclicCallableErr -> cyclic "callables"
    CyclicModuleErr -> cyclic "modules"
    CyclicNamedRenamingErr -> cyclic "named renamings"
    CyclicPackageErr -> cyclic "packages"
    DeclContextErr -> "Declaration context error"
    InvalidDeclErr -> "Declaration error"
    MiscErr -> "Error"
    ModeMismatchErr -> "Mode error"
    NotImplementedErr -> "Not implemented"
    ParseErr -> "Parse error"
    TypeErr -> "Type error"
    UnboundFunctionErr -> unbound "function"
    UnboundTopLevelErr -> unbound "top-level reference"
    UnboundNameErr -> unbound "name"
    UnboundProcedureErr -> unbound "procedure"
    UnboundTypeErr -> unbound "type"
    UnboundVarErr -> unbound "variable"
    where
      ambiguous s = "Error: could not disambiguate between" <+> s
      cyclic s = "Error: found cyclic dependency between" <+> s
      unbound s = "Error:" <+> s <+> "not in scope"

instance Pretty Name where
  pretty (Name _ str) = p str

instance Pretty NameSpace where
  pretty ns = (case ns of
    NSDirectory -> "directory"
    NSFunction -> "function"
    NSGenerated -> "generated"
    NSModule -> "module"
    NSPackage -> "package"
    NSProcedure -> "procedure"
    NSRenaming -> "renaming"
    NSSatisfaction -> "satisfaction"
    NSType -> "type"
    NSUnspecified -> "unspecified"
    NSVariable -> "variable") <+> "namespace"

instance Pretty FullyQualifiedName where
  pretty (FullyQualifiedName mscopeName targetName) =
    maybe "" (\n -> p n <+> ".") mscopeName <+> p targetName

instance (Show (e p), Pretty (e p)) => Pretty (Ann p e) where
  pretty = p . _elem

instance Pretty (MPackage' PhCheck) where
  pretty (MPackage name decls deps) =
    let importHeader = case deps of
          [] -> ""
          _  -> "imports" <+> align (vsep (punctuate comma (map p deps)))
    in "package" <+> p name <+> importHeader <+> ";" <> "\n\n"
        <> vsep (map p (join (M.elems decls)))

instance Pretty (MPackageDep' PhCheck) where
  pretty (MPackageDep name) = p name

instance Pretty (MTopLevelDecl PhCheck) where
  pretty decl = case decl of
    MNamedRenamingDecl namedRenaming -> p namedRenaming
    MModuleDecl modul -> p modul
    MSatisfactionDecl satisfaction -> p satisfaction

instance Pretty (MNamedRenaming' PhCheck) where
  pretty (MNamedRenaming name block) =
    "renaming" <+> p name <+> "=" <+> p block

instance Pretty (MSatisfaction' PhCheck) where
  pretty = undefined -- TODO

instance Pretty (RenamedModule PhCheck) where
  pretty (RenamedModule modul renamingBlocks) =
    let renamingTrailer = if null renamingBlocks then ""
                          else align (vsep (map p renamingBlocks))
    in p modul <> renamingTrailer

instance Pretty (MModule' PhCheck) where
  pretty (MModule moduleType name declMap depList) =
    vsep [ nest 4 (vsep (  p moduleType <+> p name <+> "= {"
                        :  map p depList
                        <> map ((<> ";") . p)
                               (join (M.elems (M.map cleanUp declMap)))
                        ))
         , "}"
         ]
    where
      cleanUp :: [TcDecl] -> [TcDecl]
      cleanUp = foldl deDup []

      deDup :: [TcDecl] -> TcDecl -> [TcDecl]
      deDup decls d = case d of
        TypeDecl _ -> if d `elem` decls then decls else d : decls
        CallableDecl cd ->
          let anonD = mkAnonProto <$$> cd in
          if any (matchAnonProtoDecl anonD) decls
          then if callableIsImplemented cd
               then d : filter (not . matchAnonProtoDecl anonD) decls
               else decls
          else d : decls

      matchAnonProtoDecl target decl = case decl of
        TypeDecl _ -> False
        CallableDecl cd -> mkAnonProto <$$> cd == target


  pretty (RefModule _ _ v) = absurd v

instance Pretty (MModuleDep' PhCheck) where
  pretty (MModuleDep name renamingBlocks castToSig) =
    let pName = if castToSig then "signature(" <+> p name <+> ")"
                             else p name
    in pName <+> align (vsep (map p renamingBlocks)) <> ";"

instance Pretty (MRenamingBlock' PhCheck) where
  pretty (MRenamingBlock renamings) =
    brackets $ hsep (punctuate comma (map p renamings))

instance Pretty (MRenaming' PhCheck) where
  pretty renaming = case renaming of
    InlineRenaming (src, tgt) -> p src <+> "=>" <+> p tgt
    RefRenaming v -> absurd v

instance Pretty MModuleType where
  pretty typ = case typ of
    Signature -> "signature"
    Concept -> "concept"
    Implementation -> "implementation"
    Program -> "program"
    External backend fqn -> "external" <+> p backend <+> p fqn

instance Pretty Backend where
  pretty backend = case backend of
    Cxx -> "C++"
    JavaScript -> "JavaScript"
    Python -> "Python"

instance Pretty (MDecl PhCheck) where
  pretty decl = case decl of
    TypeDecl tdecl -> pretty tdecl
    CallableDecl cdecl -> pretty cdecl

instance Pretty (TypeDecl' p) where
  pretty (Type typ isRequired) = (if isRequired then "require" else "") <>
    "type" <+> p typ

instance Pretty (CallableDecl' p) where
  pretty (Callable callableType name args ret mguard cbody) =
    let pret = if callableType == Function then " : " <> p ret else ""
        pbody = case cbody of EmptyBody -> ""
                              MagnoliaBody body -> " = " <> p body
                              ExternalBody -> " = <external impl>;"
        pguard = case mguard of Nothing -> ""
                                Just guard -> " guard " <> p guard
    in p callableType <+> p name <> prettyArgs <>
        pret <> pguard <> pbody
    where
      prettyArgs :: Doc ann
      prettyArgs = case callableType of
        Procedure -> parens $ hsep $ punctuate comma (map p args)
        _ -> parens $ hsep $ punctuate comma $
            map (\(Ann _ a) -> p (nodeName a) <+> ":" <+> p (_varType a)) args

instance Pretty CallableType where
  pretty callableType = case callableType of
    Axiom -> "axiom"
    Function -> "function"
    Predicate -> "predicate"
    Procedure -> "procedure"

instance Pretty (MExpr' p) where
  pretty e = pNoSemi e <> semi
    where
      pNoSemi :: MExpr' p -> Doc ann
      pNoSemi expr = case expr of
        MVar v -> p (nodeName v)
        MCall name args mcast -> let parglist = map (pNoSemi . _elem) args in
          p name <> parens (hsep (punctuate comma parglist)) <>
          (case mcast of Nothing -> ""; Just cast -> " : " <> p cast)
        MBlockExpr _ block ->
          vsep [ nest 4 (vsep ( "{" : map p (NE.toList block)))
               , "}"
               ]
        MValue expr' -> "value" <+> p expr'
        -- TODO: modes are for now ignore in MLet
        MLet (Ann _ (Var _ name mcast)) mass ->
          let pcast = case mcast of Nothing -> ""; Just cast -> " : " <> p cast
              pass = case mass of Nothing -> ""
                                  Just ass -> " = " <> pNoSemi (_elem ass)
          in "var" <+> p name <> pcast <> pass
        MIf cond etrue efalse -> align $ vsep [ "if" <+> p cond
                                              , "then" <+> p etrue
                                              , "else" <+> p efalse <+> "end"
                                              ]
        MAssert aexpr -> "assert" <+> pNoSemi (_elem aexpr)
        MSkip -> "skip"

instance Pretty (MaybeTypedVar' p) where
  pretty (Var mode name mtyp) = case mtyp of
    Nothing -> p mode <+> p name
    Just typ -> p mode <+> p name <+> ":" <+> p typ

instance Pretty (TypedVar' p) where
  pretty (Var mode name typ) = p mode <+> p name <+> ":" <+> p typ

instance Pretty MVarMode where
  pretty mode = case mode of
    MObs -> "obs"
    MOut -> "out"
    MUnk -> "unk"
    MUpd -> "upd"