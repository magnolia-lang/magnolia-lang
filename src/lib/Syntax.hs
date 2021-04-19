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
{-# LANGUAGE UndecidableInstances #-}

module Syntax (
    CallableDecl, CallableDecl' (..), CallableType (..), MaybeTypedVar,
    MaybeTypedVar', TypeDecl, TypeDecl' (..), TypedVar, TypedVar',
    UDecl (..), UExpr, UExpr' (..), UModule,
    UModule' (..), UModuleDep, UModuleDep' (..), UModuleType (..),
    UNamedRenaming, UNamedRenaming' (..),
    UPackage, UPackage' (..), UPackageDep, UPackageDep' (..), URenamingBlock,
    URenamingBlock' (..), USatisfaction, USatisfaction' (..),
    UTopLevelDecl (..), UType, UVar (..), UVarMode (..), WithSrc (..),
    GlobalEnv, InlineRenaming, PackageHead (..), RenamedModule (..),
    URenaming' (..), URenaming,
    CBody (..), CGuard,
    TCCallableDecl, TCDecl, TCExpr, TCMaybeTypedVar, TCModule, TCModuleDep,
    TCPackage, TCTopLevelDecl, TCTypeDecl, TCTypedVar,
    HasSrcCtx (..), NamedNode (..),
    Command (..),
    DeclOrigin (..), Err (..), ErrType (..),
    PhParse, PhCheck, PhCodeGen, SrcCtx,
    Ann (..), XAnn, (<$$>), (<$$),
    XRef,
    pattern Pred, pattern Unit,
    pattern NoCtx,
    pattern UCon, pattern UProg, pattern UImpl, pattern USig,
    pattern UAxiom, pattern UFunc, pattern UPred, pattern UProc,
    getModules, getNamedRenamings,
    moduleDecls, moduleDepNames,
    getTypeDecls, getCallableDecls,
    callableIsImplemented, mkAnonProto, replaceGuard)
  where

import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import qualified Data.Text.Lazy as T
import Data.Void

import Env

-- === repl utils ===

data Command = LoadPackage String
             | ReloadPackage String
             | InspectModule Name -- TODO: add optional package name
             | InspectPackage Name
             | ListPackages
             | ShowMenu

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
type TCCallableDecl = CallableDecl PhCheck
type TCTypeDecl = TypeDecl PhCheck
type TCExpr = UExpr PhCheck
type TCTypedVar = TypedVar PhCheck
type TCMaybeTypedVar = MaybeTypedVar PhCheck

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
data UModule' p = UModule UModuleType Name (XPhasedContainer p (UDecl p))
                                           (XPhasedContainer p (UModuleDep p))
                | RefModule UModuleType Name (XRef p)

-- Expose out for DAG building
type UModuleDep p = Ann p UModuleDep'
-- Represents a dependency to a module with an associated list of renaming
-- blocks, as well as whether to only extract the signature of the dependency.
data UModuleDep' p = UModuleDep Name [URenamingBlock p] Bool

type URenamingBlock p = Ann p URenamingBlock'
newtype URenamingBlock' p = URenamingBlock [URenaming p]

-- TODO: make annotated binders to keep typing information
type URenaming p = Ann p URenaming'
data URenaming' p = InlineRenaming InlineRenaming
                  | RefRenaming (XRef p)

type InlineRenaming = (Name, Name)

data UModuleType = Signature | Concept | Implementation | Program | External
                   deriving (Eq, Show)

data UDecl p = TypeDecl (TypeDecl p)
             | CallableDecl (CallableDecl p)
               deriving (Eq, Show)

type TypeDecl p = Ann p TypeDecl'
newtype TypeDecl' p = Type UType
                      deriving (Eq, Show)

type CallableDecl p = Ann p CallableDecl'
data CallableDecl' p =
  Callable CallableType Name [TypedVar p] UType (CGuard p) (CBody p)
  deriving (Eq, Show)

data CBody p = ExternalBody | EmptyBody | MagnoliaBody (UExpr p)
  deriving (Eq, Show)
type CGuard p = Maybe (UExpr p)

type UType = Name

-- Predicate type, used for Conditionals
pattern Pred :: UType
pattern Pred = GenName "Predicate"

-- Unit type, used for Statements
pattern Unit :: UType
pattern Unit = GenName "Unit"

-- TODO: can I simplify? Axioms and Predicates are function, except with a
--       predefined return type.
data CallableType = Axiom | Function | Predicate | Procedure
                    deriving (Eq, Show)

-- TODO: make a constructor for coercedexpr to remove (Maybe UType) from calls
type UExpr p = Ann p UExpr'
data UExpr' p = UVar (MaybeTypedVar p)
              -- TODO: add Procedure/FunctionLike namespaces to Name?
              | UCall Name [UExpr p] (Maybe UType)
              | UBlockExpr (NE.NonEmpty (UExpr p))
              | ULet UVarMode Name (Maybe UType) (Maybe (UExpr p))
              | UIf (UExpr p) (UExpr p) (UExpr p)
              | UAssert (UExpr p)
              | USkip
              deriving (Eq, Show)

type TypedVar p = Ann p TypedVar'
type TypedVar' = UVar UType

type MaybeTypedVar p = Ann p MaybeTypedVar'
type MaybeTypedVar' = UVar (Maybe UType)

data UVar typAnnType p = Var { _varMode :: UVarMode
                             , _varName :: Name
                             , _varType :: typAnnType
                             }
                          deriving (Eq, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data UVarMode = UObs | UOut | UUnk | UUpd
                deriving (Eq, Show)

-- == code generation utils ==

--data Backend = Cxx | JavaScript | Python

-- == annotation utils ==

type SrcPos = (String, Int, Int)

type SrcCtx = Maybe (SrcPos, SrcPos)
data WithSrc a = WithSrc { _srcCtx :: SrcCtx
                         , _fromSrc :: a
                         }
                 deriving (Functor, Show)

instance Eq a => Eq (WithSrc a) where
  (WithSrc _ x) == (WithSrc _ y) = x == y

data Err = Err ErrType SrcCtx T.Text --WithSrc T.Text
           deriving Show

data ErrType = AmbiguousFunctionRefErr
             | AmbiguousModuleRefErr
             | AmbiguousNamedRenamingRefErr
             | AmbiguousProcedureRefErr
             | CompilerErr
             | CyclicCallableErr
             | CyclicModuleErr
             | CyclicNamedRenamingErr
             | CyclicPackageErr
             | DeclContextErr
             | InvalidDeclErr
             | MiscErr
             | ModeMismatchErr
             | NotImplementedErr
             | ParseErr
             | TypeErr
             | UnboundFunctionErr
             | UnboundModuleErr
             | UnboundNameErr
             | UnboundProcedureErr
             | UnboundTypeErr
             | UnboundNamedRenamingErr
             | UnboundVarErr
               deriving Show

-- TODO: External
-- TODO: actually deal with ImportedDecl
data DeclOrigin = LocalDecl SrcCtx | ImportedDecl Name Name SrcCtx -- or | External Name
                  deriving (Eq, Ord, Show)

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
  XAnn PhCheck UNamedRenaming' = DeclOrigin

  XAnn PhParse USatisfaction' = SrcCtx
  XAnn PhCheck USatisfaction' = DeclOrigin

  XAnn PhParse UModule' = SrcCtx
  XAnn PhCheck UModule' = DeclOrigin

  XAnn PhParse UModuleDep' = SrcCtx
  XAnn PhCheck UModuleDep' = SrcCtx

  XAnn PhParse URenamingBlock' = SrcCtx
  XAnn PhCheck URenamingBlock' = SrcCtx

  XAnn PhParse URenaming' = SrcCtx
  XAnn PhCheck URenaming' = DeclOrigin

  XAnn PhParse TypeDecl' = SrcCtx
  XAnn PhCheck TypeDecl' = [DeclOrigin]

  XAnn PhParse CallableDecl' = SrcCtx
  XAnn PhCheck CallableDecl' = [DeclOrigin]

  XAnn PhParse UExpr' = SrcCtx
  XAnn PhCheck UExpr' = UType

  XAnn PhParse (UVar _) = SrcCtx
  XAnn PhCheck (UVar _) = UType

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
  nodeName (UModuleDep name _ _) = name

instance NamedNode (UDecl p) where
  nodeName decl = case decl of
    TypeDecl tdecl -> nodeName tdecl
    CallableDecl cdecl -> nodeName cdecl

instance NamedNode (TypeDecl' p) where
  nodeName (Type name) = name

instance NamedNode (CallableDecl' p) where
  nodeName (Callable _ name _ _ _ _) = name

instance NamedNode (UVar typAnnType p) where
  nodeName (Var _ name _) = name


class HasSrcCtx a where
  srcCtx :: a -> SrcCtx

instance HasSrcCtx SrcCtx where
  srcCtx = id

instance HasSrcCtx DeclOrigin where
  srcCtx declO = case declO of
    LocalDecl src -> src
    ImportedDecl _ _ src -> src

instance HasSrcCtx (SrcCtx, a) where
  srcCtx (src, _) = src

instance HasSrcCtx (WithSrc a) where
  srcCtx = _srcCtx

instance HasSrcCtx (XAnn p e) => HasSrcCtx (Ann p e) where
  srcCtx = srcCtx . _ann

-- === useful patterns ===

pattern NoCtx :: a -> WithSrc a
pattern NoCtx item = WithSrc Nothing item

pattern USig
  :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p)
  -> UModule' p
pattern USig name decls deps = UModule Signature name decls deps

pattern UCon
  :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p)
  -> UModule' p
pattern UCon name decls deps = UModule Concept name decls deps

pattern UProg
  :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p)
  -> UModule' p
pattern UProg name decls deps = UModule Program name decls deps

pattern UImpl
  :: Name -> XPhasedContainer p (UDecl p) -> XPhasedContainer p (UModuleDep p)
  -> UModule' p
pattern UImpl name decls deps = UModule Implementation name decls deps

pattern UAxiom
  :: Name -> [TypedVar p] -> UType -> CGuard p -> CBody p -> CallableDecl' p
pattern UAxiom name args retType guard body =
  Callable Axiom name args retType guard body

pattern UFunc
  :: Name -> [TypedVar p] -> UType -> CGuard p -> CBody p -> CallableDecl' p
pattern UFunc name args retType guard body =
  Callable Function name args retType guard body

pattern UPred
  :: Name -> [TypedVar p] -> UType -> CGuard p -> CBody p -> CallableDecl' p
pattern UPred name args retType guard body =
  Callable Predicate name args retType guard body

pattern UProc
  :: Name -> [TypedVar p] -> UType -> CGuard p -> CBody p -> CallableDecl' p
pattern UProc name args retType guard body =
  Callable Procedure name args retType guard body

-- === top level declarations manipulation ===

getModules :: Foldable t => t (UTopLevelDecl p) -> [UModule p]
getModules = foldl extractModule []
  where
    extractModule :: [UModule p] -> UTopLevelDecl p -> [UModule p]
    extractModule acc topLevelDecl
      | UModuleDecl m <- topLevelDecl = m:acc
      | otherwise = acc

getNamedRenamings :: Foldable t => t (UTopLevelDecl p) -> [UNamedRenaming p]
getNamedRenamings = foldl extractNamedRenaming []
  where
    extractNamedRenaming
      :: [UNamedRenaming p] -> UTopLevelDecl p -> [UNamedRenaming p]
    extractNamedRenaming acc topLevelDecl
      | UNamedRenamingDecl nr <- topLevelDecl = nr:acc
      | otherwise = acc

-- === modules manipulation ===

moduleDecls :: UModule PhCheck -> Env [UDecl PhCheck]
moduleDecls (Ann _ modul) = case modul of
  UModule _ _ decls _ -> decls
  RefModule _ _ v -> absurd v

moduleDepNames :: UModule PhParse -> [Name]
moduleDepNames (Ann _ modul) = case modul of
  UModule _ _ _ deps -> map nodeName deps
  RefModule _ _ refName -> [refName]

-- === module declarations manipulation ===

getTypeDecls :: Foldable t => t (UDecl p) -> [TypeDecl p]
getTypeDecls = foldl extractType []
  where
    extractType :: [TypeDecl p] -> UDecl p -> [TypeDecl p]
    extractType acc decl = case decl of
      TypeDecl tdecl -> tdecl:acc
      _ -> acc

getCallableDecls :: Foldable t => t (UDecl p) -> [CallableDecl p]
getCallableDecls = foldl extractCallable []
  where
    extractCallable :: [CallableDecl p] -> UDecl p -> [CallableDecl p]
    extractCallable acc decl = case decl of
      CallableDecl cdecl -> cdecl:acc
      _ -> acc

-- === declarations manipulation ===

replaceGuard :: CallableDecl' p -> CGuard p -> CallableDecl' p
replaceGuard (Callable callableType name args retType _ mbody) mguard =
  Callable callableType name args retType mguard mbody

-- TODO: avoid partiality
callableIsImplemented :: CallableDecl p -> Bool
callableIsImplemented (Ann _ (Callable _ _ _ _ _ mbody)) = case mbody of
  EmptyBody -> False
  _ -> True

mkAnonProto :: CallableDecl' p -> CallableDecl' p
mkAnonProto (Callable ctype callableName args retType mguard _) =
      Callable ctype callableName (map (mkAnonVar <$$>) args) retType
               mguard EmptyBody
  where mkAnonVar (Var mode _ typ) = Var mode (GenName "#anon#") typ