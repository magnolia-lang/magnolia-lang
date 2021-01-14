{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE PatternSynonyms #-}

module Syntax (
    TExpr, TExpr' (..), UCallable (..), UDecl, UDecl' (..), UExpr, UExpr' (..),
    UModule, UModule' (..), UModuleDep, UModuleDep' (..), UModuleType (..),
    UPackage (..), UType, UVar, UVar' (..), UVarMode (..), WithSrc (..),
    GlobalEnv, Module, Package, Renaming, RenamingBlock,
    pattern Pred, pattern Unit,
    pattern NoCtx,
    pattern UCon, pattern UProg, pattern UImpl, pattern USig,
    pattern UAxiom, pattern UFunc, pattern UPred, pattern UProc)
  where

import Data.List.NonEmpty as NE

import Env

-- === front-end language CST ===

-- TODO: should we have an IR for functional optimisations?

-- === AST ===

type GlobalEnv = Env Package
type Package = Env Module
type Module = Env [UDecl]

data UPackage = UPackage Name [UModule] [Name]
                deriving Show

-- TODO: namespace
type UModule = WithSrc UModule'
data UModule' = UModule UModuleType Name [UDecl] [UModuleDep]
                deriving Show

-- Expose out for DAG building
type UModuleDep = WithSrc UModuleDep'
data UModuleDep' = UModuleDep Name [RenamingBlock]
                   deriving Show

type RenamingBlock = WithSrc [Renaming]

-- TODO: make annotated binders to keep typing information
type Renaming = (Name, Name)

data UModuleType = Signature | Concept | Implementation | Program
                   deriving (Eq, Show)

type UDecl = WithSrc UDecl'
data UDecl' = UType UType
            -- TODO: guards/partiality
            | UCallable UCallable Name [UVar] (WithSrc UType) (Maybe UExpr)
              deriving (Eq, Show)

type UType = Name

-- Predicate type, used for Conditionals
pattern Pred :: UType
pattern Pred = GenName "Predicate"

-- Unit type, used for Statements
pattern Unit :: UType
pattern Unit = GenName "Unit"

-- TODO: can I simplify? Axioms and Predicates are function, except with a
--       predefined return type.
data UCallable = Axiom | Function | Predicate | Procedure
                 deriving (Eq, Show)

-- The duplication of the AST seems to be the cheapest way to solve the "AST
-- typing problem". See http://blog.ezyang.com/2013/05/the-ast-typing-problem/
-- for more context.
-- TODO: use only a single return type.
type TExpr = (TExpr', [UType])
data TExpr' = TVar UVar
            | TCall Name [TExpr] (Maybe UType)
            | TBlockExpr (NE.NonEmpty TExpr)
            | TLet UVarMode Name (Maybe UType) (Maybe TExpr)
            | TIf TExpr TExpr TExpr
            | TAssert TExpr
            | TSkip
            | TUnk UExpr
              deriving (Eq, Show)

-- TODO: split frontend and backend?
type UExpr = WithSrc UExpr'
data UExpr' = UVar UVar
            -- TODO: add Procedure/FunctionLike namespaces to Name?
            | UCall Name [UExpr] (Maybe UType)
            | UBlockExpr (NE.NonEmpty UStmt)
            | ULet UVarMode Name (Maybe UType) (Maybe UExpr)
            | UIf UPred UExpr UExpr
            | UAssert UPred
            | USkip
              deriving (Eq, Show)

-- Statement are expressions with Unit type.
type UStmt = UExpr
-- Predicates are expressions with Predicate type.
type UPred = UExpr

type UVar = WithSrc UVar'
data UVar' = Var { _varMode :: UVarMode
                 , _varName :: Name
                 , _varType :: Maybe UType
                 }
             deriving (Eq, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data UVarMode = UObs | UOut | UUnk | UUpd
                deriving (Eq, Show)

-- TODO: deal with annotations
type SrcPos = (String, Int, Int)

-- Example from Dex
type SrcCtx = Maybe (SrcPos, SrcPos)
data WithSrc a = WithSrc SrcCtx a
                 deriving (Functor, Show)

instance Eq a => Eq (WithSrc a) where
  (WithSrc _ x) == (WithSrc _ y) = x == y

pattern NoCtx :: a -> WithSrc a
pattern NoCtx item = WithSrc Nothing item

pattern USig :: Name -> [UDecl] -> [UModuleDep] -> UModule'
pattern USig name decls deps = UModule Signature name decls deps

pattern UCon :: Name -> [UDecl] -> [UModuleDep] -> UModule'
pattern UCon name decls deps = UModule Concept name decls deps

pattern UProg :: Name -> [UDecl] -> [UModuleDep] -> UModule'
pattern UProg name decls deps = UModule Program name decls deps

pattern UImpl :: Name -> [UDecl] -> [UModuleDep] -> UModule'
pattern UImpl name decls deps = UModule Implementation name decls deps

pattern UAxiom :: Name -> [UVar] -> WithSrc UType -> Maybe UExpr -> UDecl'
pattern UAxiom name args retType body = UCallable Axiom name args retType body

pattern UFunc :: Name -> [UVar] -> WithSrc UType -> Maybe UExpr -> UDecl'
pattern UFunc name args retType body = UCallable Function name args retType body

pattern UPred :: Name -> [UVar] -> WithSrc UType -> Maybe UExpr -> UDecl'
pattern UPred name args retType body =
  UCallable Predicate name args retType body

pattern UProc :: Name -> [UVar] -> WithSrc UType -> Maybe UExpr -> UDecl'
pattern UProc name args retType body =
  UCallable Procedure name args retType body
