{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Syntax (
    UCallable (..), UDecl, UDecl' (..), UExpr, UExpr' (..), UModule,
    UModule' (..), UModuleDep, UModuleDep' (..), UModuleType (..),
    UPackage, UPackage' (..), URenamingBlock, URenamingBlock' (..), UType,
    UVar, UVar' (..), UVarMode (..), WithSrc (..),
    GlobalEnv, ModuleEnv, PackageEnv, Renaming,
    TCDecl, TCExpr, TCModule, TCPackage, TCVar,
    PhParse, PhCheck, PhCodeGen, SrcCtx,
    Ann (..), XAnn, (<$$>), (<$$),
    pattern Pred, pattern Unit,
    pattern NoCtx,
    pattern UCon, pattern UProg, pattern UImpl, pattern USig,
    pattern UAxiom, pattern UFunc, pattern UPred, pattern UProc)
  where

import Data.List.NonEmpty as NE
--import Data.Void

import Env

-- === front-end language CST ===

-- TODO: should we have an IR for functional optimisations?

-- === env utils ===

type GlobalEnv e = Env (PackageEnv e)
type PackageEnv e = Env (ModuleEnv e)
type ModuleEnv e = Env [UDecl e]

type TCPackage = PackageEnv PhCheck
type TCModule = ModuleEnv PhCheck
type TCDecl = UDecl PhCheck
type TCExpr = UExpr PhCheck
type TCVar = UVar PhCheck

-- Ann [compilation phase] [node type]
data Ann p e = Ann { _ann :: XAnn p e
                   , _elem :: e p
                   }

instance Eq (e p) => Eq (Ann p e) where
  Ann _ e1 == Ann _ e2 = e1 == e2

instance Show (e p) => Show (Ann p e) where
  show = show . _elem

(<$$>) :: XAnn p e ~ XAnn p' e' => (e p -> e' p') -> Ann p e -> Ann p' e'
(<$$>) f e = Ann { _ann = _ann e, _elem = f (_elem e) }

(<$$) :: XAnn p e ~ XAnn p' e' => e' p' -> Ann p e -> Ann p' e'
(<$$) e (Ann ann _) = Ann { _ann = ann, _elem = e }

-- === AST ===

type UPackage p = Ann p UPackage'
data UPackage' p = UPackage Name [UModule p] [Name]
                   deriving (Eq, Show)

type UModule p = Ann p UModule'
data UModule' p = UModule { _moduleType :: UModuleType
                          , _moduleName :: Name
                          , _moduleDecls :: [UDecl p]
                          , _moduleDeps :: [UModuleDep p]
                          }
                  deriving (Eq, Show)

-- Expose out for DAG building
type UModuleDep p = Ann p UModuleDep'
data UModuleDep' p = UModuleDep Name [URenamingBlock p]
                     deriving (Eq, Show)

type URenamingBlock p = Ann p URenamingBlock'
newtype URenamingBlock' p = URenamingBlock [Renaming]
                            deriving (Eq, Show)


-- TODO: make annotated binders to keep typing information
type Renaming = (Name, Name)

data UModuleType = Signature | Concept | Implementation | Program
                   deriving (Eq, Show)

type UDecl p = Ann p UDecl'
data UDecl' p = UType UType
              -- TODO: guards/partiality
              | UCallable UCallable Name [UVar p] UType (Maybe (UExpr p))
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

type UExpr p = Ann p UExpr'
data UExpr' p = UVar (UVar p)
              -- TODO: add Procedure/FunctionLike namespaces to Name?
              | UCall Name [UExpr p] (Maybe UType)
              | UBlockExpr (NE.NonEmpty (UExpr p))
              | ULet UVarMode Name (Maybe UType) (Maybe (UExpr p))
              | UIf (UExpr p) (UExpr p) (UExpr p)
              | UAssert (UExpr p)
              | USkip
              deriving (Eq, Show)
              -- TODO: "upstream" to Ann?
              -- | UExpr (XExprExt p)

type UVar p = Ann p UVar'
data UVar' p = Var { _varMode :: UVarMode
                   , _varName :: Name
                   , _varType :: Maybe UType
                   }
               deriving (Eq, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data UVarMode = UObs | UOut | UUnk | UUpd
                deriving (Eq, Show)

-- TODO: deal with annotations
type SrcPos = (String, Int, Int)

type SrcCtx = Maybe (SrcPos, SrcPos)
data WithSrc a = WithSrc SrcCtx a
                 deriving (Functor, Show)

instance Eq a => Eq (WithSrc a) where
  (WithSrc _ x) == (WithSrc _ y) = x == y

-- === compilation phases ===

data PhParse
data PhCheck
data PhCodeGen

-- === XAnn type family ===

type family XAnn p (e :: * -> *)

type instance XAnn PhParse UPackage' = SrcCtx
type instance XAnn PhCheck UPackage' = SrcCtx

type instance XAnn PhParse UModule' = SrcCtx
type instance XAnn PhCheck UModule' = SrcCtx

type instance XAnn PhParse UModuleDep' = SrcCtx
type instance XAnn PhCheck UModuleDep' = SrcCtx

type instance XAnn PhParse URenamingBlock' = SrcCtx
type instance XAnn PhCheck URenamingBlock' = SrcCtx

type instance XAnn PhParse UDecl' = SrcCtx
type instance XAnn PhCheck UDecl' = SrcCtx

type instance XAnn PhParse UExpr' = SrcCtx
type instance XAnn PhCheck UExpr' = UType

type instance XAnn PhParse UVar' = SrcCtx
type instance XAnn PhCheck UVar' = UType

-- === useful patterns ===

pattern NoCtx :: a -> WithSrc a
pattern NoCtx item = WithSrc Nothing item

pattern USig :: Name -> [UDecl p] -> [UModuleDep p] -> UModule' p
pattern USig name decls deps = UModule Signature name decls deps

pattern UCon :: Name -> [UDecl p] -> [UModuleDep p] -> UModule' p
pattern UCon name decls deps = UModule Concept name decls deps

pattern UProg :: Name -> [UDecl p] -> [UModuleDep p] -> UModule' p
pattern UProg name decls deps = UModule Program name decls deps

pattern UImpl :: Name -> [UDecl p] -> [UModuleDep p] -> UModule' p
pattern UImpl name decls deps = UModule Implementation name decls deps

pattern UAxiom :: Name -> [UVar p] -> UType -> Maybe (UExpr p) -> UDecl' p
pattern UAxiom name args retType body = UCallable Axiom name args retType body

pattern UFunc :: Name -> [UVar p] -> UType -> Maybe (UExpr p) -> UDecl' p
pattern UFunc name args retType body = UCallable Function name args retType body

pattern UPred :: Name -> [UVar p] -> UType -> Maybe (UExpr p) -> UDecl' p
pattern UPred name args retType body =
  UCallable Predicate name args retType body

pattern UProc :: Name -> [UVar p] -> UType -> Maybe (UExpr p) -> UDecl' p
pattern UProc name args retType body =
  UCallable Procedure name args retType body
