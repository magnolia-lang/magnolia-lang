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
    MDecl (..), MExpr, MExpr' (..), MModule,
    MModule' (..), MModuleDep, MModuleDep' (..), MModuleType (..),
    MNamedRenaming, MNamedRenaming' (..),
    MPackage, MPackage' (..), MPackageDep, MPackageDep' (..), MRenamingBlock,
    MRenamingBlock' (..), MSatisfaction, MSatisfaction' (..),
    MTopLevelDecl (..), MType, MVar (..), MVarMode (..), WithSrc (..),
    GlobalEnv, InlineRenaming, PackageHead (..), RenamedModule (..),
    MRenaming' (..), MRenaming,
    CBody (..), CGuard,
    TCCallableDecl, TCDecl, TCExpr, TCMaybeTypedVar, TCModule, TCModuleDep,
    TCPackage, TCTopLevelDecl, TCTypeDecl, TCTypedVar,
    HasDependencies (..), HasSrcCtx (..), NamedNode (..),
    Command (..),
    DeclOrigin (..), Err (..), ErrType (..),
    PhParse, PhCheck, PhCodeGen, SrcCtx,
    Ann (..), XAnn, (<$$>), (<$$),
    XRef,
    pattern Pred, pattern Unit,
    pattern NoCtx,
    pattern MCon, pattern MProg, pattern MImpl, pattern MSig,
    pattern MAxiom, pattern MFunc, pattern MPred, pattern MProc,
    getModules, getNamedRenamings,
    moduleDecls,
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

type GlobalEnv p = Env (MPackage p)

--type GlobalEnv p = Env (Env (Env [MDecl p]))

type TCPackage = MPackage PhCheck
type TCTopLevelDecl = MTopLevelDecl PhCheck
type TCModule = MModule PhCheck
type TCModuleDep = MModuleDep PhCheck
type TCDecl = MDecl PhCheck
type TCCallableDecl = CallableDecl PhCheck
type TCTypeDecl = TypeDecl PhCheck
type TCExpr = MExpr PhCheck
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

-- TODO: use fully qualified names consistently at the package level
type MPackage p = Ann p MPackage'
data MPackage' p = MPackage { _packageName :: Name
                            , _packageDecls :: XPhasedContainer p (MTopLevelDecl p)
                            , _packageDeps :: [MPackageDep p]
                            }

type MPackageDep p = Ann p MPackageDep'
newtype MPackageDep' p = MPackageDep FullyQualifiedName
                         deriving (Eq, Show)

-- TODO: split in package?
data MTopLevelDecl p = MNamedRenamingDecl (MNamedRenaming p)
                     | MModuleDecl (MModule p)
                     | MSatisfactionDecl (MSatisfaction p)

type MNamedRenaming p = Ann p MNamedRenaming'
data MNamedRenaming' p = MNamedRenaming Name (MRenamingBlock p)

type MSatisfaction p = Ann p MSatisfaction'
data MSatisfaction' p =
  MSatisfaction Name (RenamedModule p) (Maybe (RenamedModule p)) (RenamedModule p)

data RenamedModule p = RenamedModule (MModule p) [MRenamingBlock p]

type MModule p = Ann p MModule'
data MModule' p = MModule MModuleType Name (XPhasedContainer p (MDecl p))
                                           [MModuleDep p]
                | RefModule MModuleType Name (XRef p)

-- Expose out for DAG building
type MModuleDep p = Ann p MModuleDep'
-- Represents a dependency to a module with an associated list of renaming
-- blocks, as well as whether to only extract the signature of the dependency.
data MModuleDep' p = MModuleDep { _depName :: FullyQualifiedName
                                , _depRenamingBlock :: [MRenamingBlock p]
                                , _depCastToSig :: Bool
                                }

type MRenamingBlock p = Ann p MRenamingBlock'
newtype MRenamingBlock' p = MRenamingBlock [MRenaming p]

-- TODO: make annotated binders to keep typing information
type MRenaming p = Ann p MRenaming'
data MRenaming' p = InlineRenaming InlineRenaming
                  | RefRenaming (XRef p)

type InlineRenaming = (Name, Name)

data MModuleType = Signature | Concept | Implementation | Program | External
                   deriving (Eq, Show)

data MDecl p = TypeDecl (TypeDecl p)
             | CallableDecl (CallableDecl p)
               deriving (Eq, Show)

type TypeDecl p = Ann p TypeDecl'
newtype TypeDecl' p = Type MType
                      deriving (Eq, Show)

type CallableDecl p = Ann p CallableDecl'
data CallableDecl' p =
  Callable CallableType Name [TypedVar p] MType (CGuard p) (CBody p)
  deriving (Eq, Show)

data CBody p = ExternalBody | EmptyBody | MagnoliaBody (MExpr p)
  deriving (Eq, Show)
type CGuard p = Maybe (MExpr p)

type MType = Name

-- Predicate type, used for Conditionals
pattern Pred :: MType
pattern Pred = GenName "Predicate"

-- Unit type, used for Statements
pattern Unit :: MType
pattern Unit = GenName "Unit"

-- TODO: can I simplify? Axioms and Predicates are function, except with a
--       predefined return type.
data CallableType = Axiom | Function | Predicate | Procedure
                    deriving (Eq, Show)

-- TODO: make a constructor for coercedexpr to remove (Maybe MType) from calls
type MExpr p = Ann p MExpr'
data MExpr' p = MVar (MaybeTypedVar p)
              -- TODO: add Procedure/FunctionLike namespaces to Name?
              | MCall Name [MExpr p] (Maybe MType)
              | MBlockExpr (NE.NonEmpty (MExpr p))
              | MLet MVarMode Name (Maybe MType) (Maybe (MExpr p))
              | MIf (MExpr p) (MExpr p) (MExpr p)
              | MAssert (MExpr p)
              | MSkip
              deriving (Eq, Show)

type TypedVar p = Ann p TypedVar'
type TypedVar' = MVar MType

type MaybeTypedVar p = Ann p MaybeTypedVar'
type MaybeTypedVar' = MVar (Maybe MType)

data MVar typAnnType p = Var { _varMode :: MVarMode
                             , _varName :: Name
                             , _varType :: typAnnType
                             }
                          deriving (Eq, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data MVarMode = MObs | MOut | MUnk | MUpd
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

data Err = Err ErrType SrcCtx T.Text
           deriving (Eq, Show)

instance Ord Err where
  Err e1 src1 txt1 `compare` Err e2 src2 txt2 = (src1 `compare` src2) <>
    (txt1 `compare` txt2) <> (e1 `compare` e2)

data ErrType = AmbiguousFunctionRefErr
             | AmbiguousProcedureRefErr
             | AmbiguousTopLevelRefErr
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
             | UnboundNameErr
             | UnboundProcedureErr
             | UnboundTopLevelErr
             | UnboundTypeErr
             | UnboundVarErr
               deriving (Eq, Ord, Show)

-- TODO: External
-- TODO: actually deal with ImportedDecl
data DeclOrigin = LocalDecl SrcCtx | ImportedDecl FullyQualifiedName SrcCtx -- or | External Name
                  deriving (Eq, Ord, Show)

-- === compilation phases ===

data PhParse
data PhCheck
data PhCodeGen

-- === XAnn type family ===

type family XAnn p (e :: * -> *) where
  XAnn PhParse MPackage' = SrcCtx
  XAnn PhCheck MPackage' = SrcCtx

  XAnn PhParse MPackageDep' = SrcCtx
  XAnn PhCheck MPackageDep' = SrcCtx

  XAnn PhParse MNamedRenaming' = SrcCtx
  XAnn PhCheck MNamedRenaming' = DeclOrigin

  XAnn PhParse MSatisfaction' = SrcCtx
  XAnn PhCheck MSatisfaction' = DeclOrigin

  XAnn PhParse MModule' = SrcCtx
  XAnn PhCheck MModule' = DeclOrigin

  XAnn PhParse MModuleDep' = SrcCtx
  XAnn PhCheck MModuleDep' = SrcCtx

  XAnn PhParse MRenamingBlock' = SrcCtx
  XAnn PhCheck MRenamingBlock' = SrcCtx

  XAnn PhParse MRenaming' = SrcCtx
  XAnn PhCheck MRenaming' = DeclOrigin

  XAnn PhParse TypeDecl' = SrcCtx
  XAnn PhCheck TypeDecl' = [DeclOrigin]

  XAnn PhParse CallableDecl' = SrcCtx
  XAnn PhCheck CallableDecl' = [DeclOrigin]

  XAnn PhParse MExpr' = SrcCtx
  XAnn PhCheck MExpr' = MType

  XAnn PhParse (MVar _) = SrcCtx
  XAnn PhCheck (MVar _) = MType

-- === other useful type families ===

type family XPhasedContainer p e where
  XPhasedContainer PhParse e = [e]
  XPhasedContainer PhCheck e = M.Map Name [e]

-- The goal of XRef is to statically prevent the existence of references to
-- named top level elements after the consistency/type checking phase.
type family XRef p where
  XRef PhParse = FullyQualifiedName
  XRef PhCheck = Void

-- TODO: move?
-- === standalone show instances ===

deriving instance Show (MRenaming' PhCheck)
deriving instance Show (MRenamingBlock' PhCheck)
deriving instance Show (MModuleDep' PhCheck)
deriving instance Show (MNamedRenaming' PhCheck)
deriving instance Show (MModule' PhCheck)
deriving instance Show (RenamedModule PhCheck)
deriving instance Show (MSatisfaction' PhCheck)
deriving instance Show (MTopLevelDecl PhCheck)
deriving instance Show (MPackage' PhCheck)

-- === useful typeclasses ===

class NamedNode n where
  nodeName :: n -> Name

instance NamedNode (e p) => NamedNode (Ann p e) where
  nodeName = nodeName . _elem

instance NamedNode (MPackage' p) where
  nodeName = _packageName

instance NamedNode (MPackageDep' p) where
  nodeName (MPackageDep name) = fromFullyQualifiedName name

instance NamedNode (MTopLevelDecl p) where
  nodeName topLevelDecl = case topLevelDecl of
    MNamedRenamingDecl namedRenaming -> nodeName namedRenaming
    MModuleDecl modul -> nodeName modul
    MSatisfactionDecl satisfaction -> nodeName satisfaction

instance NamedNode (MNamedRenaming' p) where
  nodeName (MNamedRenaming name _) = name

instance NamedNode (MModule' p) where
  nodeName (MModule _ name _ _) = name
  nodeName (RefModule _ name _) = name

instance NamedNode (MSatisfaction' p) where
  nodeName (MSatisfaction name _ _ _) = name

instance NamedNode (MModuleDep' p) where
  nodeName (MModuleDep name _ _) = fromFullyQualifiedName name

instance NamedNode (MDecl p) where
  nodeName decl = case decl of
    TypeDecl tdecl -> nodeName tdecl
    CallableDecl cdecl -> nodeName cdecl

instance NamedNode (TypeDecl' p) where
  nodeName (Type name) = name

instance NamedNode (CallableDecl' p) where
  nodeName (Callable _ name _ _ _ _) = name

instance NamedNode (MVar typAnnType p) where
  nodeName (Var _ name _) = name

class HasDependencies a where
  dependencies :: a -> [FullyQualifiedName]

instance HasDependencies (MModule PhParse) where
  dependencies (Ann _ modul) = case modul of
    MModule _ _ _ deps -> map (_depName . _elem) deps
    RefModule _ _ refName -> [refName]

instance HasDependencies (MModule PhCheck) where
  dependencies (Ann _ modul) = case modul of
    MModule _ _ _ deps -> map (_depName . _elem) deps
    RefModule _ _ v -> absurd v

instance HasDependencies (MPackage p) where
  dependencies (Ann _ (MPackage _ _ deps)) =
    map (\(Ann _ (MPackageDep depName)) -> depName) deps

class HasSrcCtx a where
  srcCtx :: a -> SrcCtx

instance HasSrcCtx SrcCtx where
  srcCtx = id

instance HasSrcCtx DeclOrigin where
  srcCtx declO = case declO of
    LocalDecl src -> src
    ImportedDecl _ src -> src

instance HasSrcCtx (SrcCtx, a) where
  srcCtx (src, _) = src

instance HasSrcCtx (WithSrc a) where
  srcCtx = _srcCtx

instance HasSrcCtx (XAnn p e) => HasSrcCtx (Ann p e) where
  srcCtx = srcCtx . _ann

-- === useful patterns ===

pattern NoCtx :: a -> WithSrc a
pattern NoCtx item = WithSrc Nothing item

pattern MSig
  :: Name -> XPhasedContainer p (MDecl p) -> [MModuleDep p]
  -> MModule' p
pattern MSig name decls deps = MModule Signature name decls deps

pattern MCon
  :: Name -> XPhasedContainer p (MDecl p) -> [MModuleDep p]
  -> MModule' p
pattern MCon name decls deps = MModule Concept name decls deps

pattern MProg
  :: Name -> XPhasedContainer p (MDecl p) -> [MModuleDep p]
  -> MModule' p
pattern MProg name decls deps = MModule Program name decls deps

pattern MImpl
  :: Name -> XPhasedContainer p (MDecl p) -> [MModuleDep p]
  -> MModule' p
pattern MImpl name decls deps = MModule Implementation name decls deps

pattern MAxiom
  :: Name -> [TypedVar p] -> MType -> CGuard p -> CBody p -> CallableDecl' p
pattern MAxiom name args retType guard body =
  Callable Axiom name args retType guard body

pattern MFunc
  :: Name -> [TypedVar p] -> MType -> CGuard p -> CBody p -> CallableDecl' p
pattern MFunc name args retType guard body =
  Callable Function name args retType guard body

pattern MPred
  :: Name -> [TypedVar p] -> MType -> CGuard p -> CBody p -> CallableDecl' p
pattern MPred name args retType guard body =
  Callable Predicate name args retType guard body

pattern MProc
  :: Name -> [TypedVar p] -> MType -> CGuard p -> CBody p -> CallableDecl' p
pattern MProc name args retType guard body =
  Callable Procedure name args retType guard body

-- === top level declarations manipulation ===

getModules :: Foldable t => t (MTopLevelDecl p) -> [MModule p]
getModules = foldl extractModule []
  where
    extractModule :: [MModule p] -> MTopLevelDecl p -> [MModule p]
    extractModule acc topLevelDecl
      | MModuleDecl m <- topLevelDecl = m:acc
      | otherwise = acc

getNamedRenamings :: Foldable t => t (MTopLevelDecl p) -> [MNamedRenaming p]
getNamedRenamings = foldl extractNamedRenaming []
  where
    extractNamedRenaming
      :: [MNamedRenaming p] -> MTopLevelDecl p -> [MNamedRenaming p]
    extractNamedRenaming acc topLevelDecl
      | MNamedRenamingDecl nr <- topLevelDecl = nr:acc
      | otherwise = acc

-- === modules manipulation ===

moduleDecls :: MModule PhCheck -> Env [MDecl PhCheck]
moduleDecls (Ann _ modul) = case modul of
  MModule _ _ decls _ -> decls
  RefModule _ _ v -> absurd v

-- === module declarations manipulation ===

getTypeDecls :: Foldable t => t (MDecl p) -> [TypeDecl p]
getTypeDecls = foldl extractType []
  where
    extractType :: [TypeDecl p] -> MDecl p -> [TypeDecl p]
    extractType acc decl = case decl of
      TypeDecl tdecl -> tdecl:acc
      _ -> acc

getCallableDecls :: Foldable t => t (MDecl p) -> [CallableDecl p]
getCallableDecls = foldl extractCallable []
  where
    extractCallable :: [CallableDecl p] -> MDecl p -> [CallableDecl p]
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