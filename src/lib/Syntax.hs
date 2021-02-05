{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Syntax (
    UCallable (..), UDecl, UDecl' (..), UExpr, UExpr' (..), UModule,
    UModule' (..), UModuleDep, UModuleDep' (..), UModuleType (..),
    UNamedRenaming, UNamedRenaming' (..),
    UPackage, UPackage' (..), UPackageDep, UPackageDep' (..), URenamingBlock,
    URenamingBlock' (..), USatisfaction, USatisfaction' (..),
    UTopLevelDecl (..), UType, UVar, UVar' (..), UVarMode (..), WithSrc (..),
    GlobalEnv, InlineRenaming, PackageHead (..), RenamedModule (..),
    URenaming' (..), URenaming,
    TCDecl, TCExpr, TCModule, TCModuleDep, TCPackage, TCTopLevelDecl, TCVar,
    NamedNode (..),
    DeclOrigin (..), Err,
    PhParse, PhCheck, PhCodeGen, SrcCtx,
    Ann (..), XAnn, (<$$>), (<$$),
    XRef,
    pattern Pred, pattern Unit,
    pattern NoCtx,
    pattern UCon, pattern UProg, pattern UImpl, pattern USig,
    pattern UAxiom, pattern UFunc, pattern UPred, pattern UProc)
  where

import Data.List.NonEmpty as NE
import Data.Map as M
import Data.Text.Lazy as T
import Data.Void

import Env

-- === preprocessing utils ===

data PackageHead = PackageHead { _packageHeadPath :: FilePath
                               , _packageHeadStr :: String
                               , _packageHeadName :: Name
                               , _packageHeadImports :: [WithSrc Name]
                               }
                   deriving (Eq, Show)

-- === env utils ===

type GlobalEnv p = Env (UPackage p)

--type GlobalEnv p = Env (Env (Env [UDecl p]))

type TCPackage = UPackage PhCheck
type TCTopLevelDecl = UTopLevelDecl PhCheck
type TCModule = UModule PhCheck
type TCModuleDep = UModuleDep PhCheck
type TCDecl = UDecl PhCheck
type TCExpr = UExpr PhCheck
type TCVar = UVar PhCheck

-- Ann [compilation phase] [node type]
data Ann p e = Ann { _ann :: XAnn p e
                   , _elem :: e p
                   }

instance Eq (e p) => Eq (Ann p e) where
  Ann _ e1 == Ann _ e2 = e1 == e2

-- TODO: display annotation using UndecidableInstances?
instance Show (e p) => Show (Ann p e) where
  show = show . _elem

(<$$>) :: XAnn p e ~ XAnn p' e' => (e p -> e' p') -> Ann p e -> Ann p' e'
(<$$>) f e = Ann { _ann = _ann e, _elem = f (_elem e) }

(<$$) :: XAnn p e ~ XAnn p' e' => e' p' -> Ann p e -> Ann p' e'
(<$$) e (Ann ann _) = Ann { _ann = ann, _elem = e }

-- === AST ===

type UPackage p = Ann p UPackage'
data UPackage' p = UPackage { _packageName :: Name
                            , _packageDecls :: XPhasedContainer p (UTopLevelDecl p)
                            , _packageDeps :: [UPackageDep p]
                            }

type UPackageDep p = Ann p UPackageDep'
newtype UPackageDep' p = UPackageDep Name -- Renaming blocks?
                         deriving (Eq, Show)

-- TODO: split in package?
data UTopLevelDecl p = UNamedRenamingDecl (UNamedRenaming p)
                     | UModuleDecl (UModule p)
                     | USatisfactionDecl (USatisfaction p)

type UNamedRenaming p = Ann p UNamedRenaming'
data UNamedRenaming' p = UNamedRenaming Name (URenamingBlock p)

type USatisfaction p = Ann p USatisfaction'
data USatisfaction' p =
  USatisfaction Name (RenamedModule p) (Maybe (RenamedModule p)) (RenamedModule p)

data RenamedModule p = RenamedModule (UModule p) [URenamingBlock p]

type UModule p = Ann p UModule'
data UModule' p = UModule UModuleType Name (XPhasedContainer p (UDecl p)) (XPhasedContainer p (UModuleDep p))
                | RefModule UModuleType Name (XRef p)

-- Expose out for DAG building
type UModuleDep p = Ann p UModuleDep'
data UModuleDep' p = UModuleDep Name [URenamingBlock p]

type URenamingBlock p = Ann p URenamingBlock'
newtype URenamingBlock' p = URenamingBlock [URenaming p]

-- TODO: make annotated binders to keep typing information
type URenaming p = Ann p URenaming'
data URenaming' p = InlineRenaming InlineRenaming
                  | RefRenaming (XRef p)

type InlineRenaming = (Name, Name)

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

type UVar p = Ann p UVar'
data UVar' p = Var { _varMode :: UVarMode
                   , _varName :: Name
                   , _varType :: Maybe UType
                   }
               deriving (Eq, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data UVarMode = UObs | UOut | UUnk | UUpd
                deriving (Eq, Show)

-- == annotation utils ==

type SrcPos = (String, Int, Int)

type SrcCtx = Maybe (SrcPos, SrcPos)
data WithSrc a = WithSrc { _srcCtx :: SrcCtx
                         , _fromSrc :: a
                         }
                 deriving (Functor, Show)

instance Eq a => Eq (WithSrc a) where
  (WithSrc _ x) == (WithSrc _ y) = x == y

type Err = WithSrc T.Text

-- TODO: External
-- TODO: actually deal with ImportedDecl
data DeclOrigin = LocalDecl | ImportedDecl Name Name -- or | External Name
                  deriving (Eq, Show)

-- === compilation phases ===

data PhParse
data PhCheck
data PhCodeGen

-- === XAnn type family ===

type family XAnn p (e :: * -> *) where
  XAnn PhParse UPackage' = SrcCtx
  XAnn PhCheck UPackage' = SrcCtx

  XAnn PhParse UPackageDep' = SrcCtx
  XAnn PhCheck UPackageDep' = SrcCtx

  XAnn PhParse UNamedRenaming' = SrcCtx
  XAnn PhCheck UNamedRenaming' = (SrcCtx, DeclOrigin)

  XAnn PhParse USatisfaction' = SrcCtx
  XAnn PhCheck USatisfaction' = (SrcCtx, DeclOrigin)

  XAnn PhParse UModule' = SrcCtx
  XAnn PhCheck UModule' = (SrcCtx, DeclOrigin)

  XAnn PhParse UModuleDep' = SrcCtx
  XAnn PhCheck UModuleDep' = SrcCtx

  XAnn PhParse URenamingBlock' = SrcCtx
  XAnn PhCheck URenamingBlock' = SrcCtx

  XAnn PhParse URenaming' = SrcCtx
  XAnn PhCheck URenaming' = (SrcCtx, DeclOrigin)

  XAnn PhParse UDecl' = SrcCtx
  XAnn PhCheck UDecl' = (SrcCtx, [DeclOrigin])

  XAnn PhParse UExpr' = SrcCtx
  XAnn PhCheck UExpr' = UType

  XAnn PhParse UVar' = SrcCtx
  XAnn PhCheck UVar' = UType

-- === other useful type families ===

type family XPhasedContainer p e where
  XPhasedContainer PhParse e = [e]
  XPhasedContainer PhCheck e = M.Map Name [e]

-- The goal of XRef is to statically prevent the existence of references to
-- named top level elements after the consistency/type checking phase.
type family XRef p where
  XRef PhParse = Name
  XRef PhCheck = Void

-- TODO: move?
-- === standalone show instances ===

deriving instance Show (URenaming' PhCheck)
deriving instance Show (URenamingBlock' PhCheck)
deriving instance Show (UModuleDep' PhCheck)
deriving instance Show (UNamedRenaming' PhCheck)
deriving instance Show (UModule' PhCheck)
deriving instance Show (RenamedModule PhCheck)
deriving instance Show (USatisfaction' PhCheck)
deriving instance Show (UTopLevelDecl PhCheck)
deriving instance Show (UPackage' PhCheck)

-- === useful typeclasses ===

class NamedNode n where
  nodeName :: n -> Name

instance NamedNode (e p) => NamedNode (Ann p e) where
  nodeName = nodeName . _elem

instance NamedNode (UPackage' p) where
  nodeName = _packageName

instance NamedNode (UPackageDep' p) where
  nodeName (UPackageDep name) = name

instance NamedNode (UTopLevelDecl p) where
  nodeName topLevelDecl = case topLevelDecl of
    UNamedRenamingDecl namedRenaming -> nodeName namedRenaming
    UModuleDecl modul -> nodeName modul
    USatisfactionDecl satisfaction -> nodeName satisfaction

instance NamedNode (UNamedRenaming' p) where
  nodeName (UNamedRenaming name _) = name

instance NamedNode (UModule' p) where
  nodeName (UModule _ name _ _) = name
  nodeName (RefModule _ name _) = name

instance NamedNode (USatisfaction' p) where
  nodeName (USatisfaction name _ _ _) = name

instance NamedNode (UModuleDep' p) where
  nodeName (UModuleDep name _) = name

instance NamedNode (UDecl' p) where
  nodeName decl = case decl of
    UType name -> name
    UCallable _ name _ _ _ -> name

-- === useful patterns ===

pattern NoCtx :: a -> WithSrc a
pattern NoCtx item = WithSrc Nothing item

pattern USig
  :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p)
  -> UModule' p
pattern USig name decls deps = UModule Signature name decls deps

pattern UCon :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p) -> UModule' p
pattern UCon name decls deps = UModule Concept name decls deps

pattern UProg :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p) -> UModule' p
pattern UProg name decls deps = UModule Program name decls deps

pattern UImpl :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p) -> UModule' p
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
