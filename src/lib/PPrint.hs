{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wno-orphans #-}

module PPrint (pprint, pshow, render) where -- TODO: remove pshow?

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
pprint = TIO.putStrLn . render

render :: Pretty a => a -> T.Text
render = renderLazy . layoutPretty defaultLayoutOptions . p

pshow :: Pretty a => a -> T.Text
pshow = render

instance Pretty Err where -- TODO: change error to have error types
  pretty (Err errType srcInfo txt) = case srcInfo of
    Nothing -> "<no context>:" <+> p errType <+> p txt
    Just ((filename, startLine, startColumn), _) ->
      p filename <> ":" <> p startLine <> ":" <> p startColumn <>
      ":" <+> p errType <+> p txt

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
    DeclContextErr -> "Declaration context error:"
    InvalidDeclErr -> "Declaration error:"
    MiscErr -> "Error:"
    ModeMismatchErr -> "Mode error:"
    NotImplementedErr -> "Not implemented:"
    ParseErr -> "Parse error:"
    TypeErr -> "Type error:"
    UnboundFunctionErr -> unbound "function"
    UnboundTopLevelErr -> unbound "top-level reference"
    UnboundNameErr -> unbound "name"
    UnboundProcedureErr -> unbound "procedure"
    UnboundTypeErr -> unbound "type"
    UnboundVarErr -> unbound "variable"
    where
      ambiguous s = "Error: could not disambiguate between" <+> s <> ":"
      cyclic s = "Error: found cyclic dependency between" <+> s <> ":"
      unbound s = "Error:" <+> s <+> "not in scope:"

instance Pretty Name where
  pretty (Name _ str) = p str

instance Pretty NameSpace where
  pretty ns = (case ns of
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

instance Pretty (UPackage' PhCheck) where
  pretty (UPackage name decls deps) =
    let importHeader = case deps of
          [] -> ""
          _  -> "imports" <+> align (vsep (punctuate comma (map p deps)))
    in "package" <+> p name <+> importHeader <+> ";" <> "\n\n"
        <> vsep (map p (join (M.elems decls)))

instance Pretty (UPackageDep' PhCheck) where
  pretty (UPackageDep name) = p name

instance Pretty (UTopLevelDecl PhCheck) where
  pretty decl = case decl of
    UNamedRenamingDecl namedRenaming -> p namedRenaming
    UModuleDecl modul -> p modul
    USatisfactionDecl satisfaction -> p satisfaction

instance Pretty (UNamedRenaming' PhCheck) where
  pretty (UNamedRenaming name block) =
    "renaming" <+> p name <+> "=" <+> p block

instance Pretty (USatisfaction' PhCheck) where
  pretty = undefined -- TODO

instance Pretty (RenamedModule PhCheck) where
  pretty (RenamedModule modul renamingBlocks) =
    let renamingTrailer = if null renamingBlocks then ""
                          else align (vsep (map p renamingBlocks))
    in p modul <> renamingTrailer

instance Pretty (UModule' PhCheck) where
  pretty (UModule moduleType name declMap depMap) =
    vsep [ nest 4 (vsep (  p moduleType <+> p name <+> "= {"
                        :  map p (join (M.elems depMap))
                        <> map p (join (M.elems (M.map cleanUp declMap)))
                        ))
         , "};"
         ]
    where
      cleanUp :: [TCDecl] -> [TCDecl]
      cleanUp = foldl deDup []

      deDup :: [TCDecl] -> TCDecl -> [TCDecl]
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

instance Pretty (UModuleDep' PhCheck) where
  pretty (UModuleDep name renamingBlocks castToSig) =
    let pName = if castToSig then "signature(" <+> p name <+> ")"
                             else p name
    in pName <+> align (vsep (map p renamingBlocks)) <> ";"

instance Pretty (URenamingBlock' PhCheck) where
  pretty (URenamingBlock renamings) =
    brackets $ hsep (punctuate comma (map p renamings))

instance Pretty (URenaming' PhCheck) where
  pretty renaming = case renaming of
    InlineRenaming (src, tgt) -> p src <+> "=>" <+> p tgt
    RefRenaming v -> absurd v

instance Pretty UModuleType where
  pretty typ = case typ of
    Signature -> "signature"
    Concept -> "concept"
    Implementation -> "implementation"
    Program -> "program"
    External -> "external"

instance Pretty (UDecl PhCheck) where
  pretty decl = case decl of
    TypeDecl tdecl -> pretty tdecl
    CallableDecl cdecl -> pretty cdecl

instance Pretty (TypeDecl' p) where
  pretty (Type typ) = "type" <+> p typ <> ";"

instance Pretty (CallableDecl' p) where
  pretty (Callable callableType name args ret mguard cbody) =
    let pret = if callableType == Function then " : " <> p ret else ""
        pbody = case cbody of EmptyBody -> ";"
                              MagnoliaBody body -> " = " <> p body
                              ExternalBody -> " = <external impl>;"
        pguard = case mguard of Nothing -> ""
                                Just guard -> " guard " <> p guard
    in p callableType <+> p name <+> prettyArgs <>
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

instance Pretty (UExpr' p) where
  pretty e = pNoSemi e <> semi
    where
      pNoSemi :: UExpr' p -> Doc ann
      pNoSemi expr = case expr of
        UVar v -> p (nodeName v)
        UCall name args mcast -> let parglist = map (pNoSemi . _elem) args in
          p name <> parens (hsep (punctuate comma parglist)) <>
          (case mcast of Nothing -> ""; Just cast -> " : " <> p cast)
        UBlockExpr block ->
          vsep [ nest 4 (vsep ( "{" : map p (NE.toList block)))
               , "}"
               ]
        -- TODO: modes are for now ignore in ULet
        ULet _ name mcast mass ->
          let pcast = case mcast of Nothing -> ""; Just cast -> " : " <> p cast
              pass = case mass of Nothing -> ""
                                  Just ass -> " = " <> pNoSemi (_elem ass)
          in "var" <+> p name <> pcast <> pass
        UIf cond etrue efalse -> align $ vsep [ "if" <+> p cond
                                              , "then" <+> p etrue
                                              , "else" <+> p efalse <+> "end"
                                              ]
        UAssert aexpr -> "assert" <+> pNoSemi (_elem aexpr)
        USkip -> "skip"

instance Pretty (MaybeTypedVar' p) where
  pretty (Var mode name mtyp) = case mtyp of
    Nothing -> p mode <+> p name
    Just typ -> p mode <+> p name <+> ":" <+> p typ

instance Pretty (TypedVar' p) where
  pretty (Var mode name typ) = p mode <+> p name <+> ":" <+> p typ

instance Pretty UVarMode where
  pretty mode = case mode of
    UObs -> "obs"
    UOut -> "out"
    UUnk -> "unk"
    UUpd -> "upd"