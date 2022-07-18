{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Magnolia.Syntax (
  -- * AST nodes
  --
  -- $mgAst
  -- TODO(bchetioui, 07/07/21): cleanup node names. This is put on hold to
  -- avoid too much rebasing of a big PR.
  -- ** Generic AST nodes
    CBody (..)
  , CGuard
  , CudaDim3 (..)
  , ExternalModuleInfo (..)
  , InlineRenaming
  , MaybeTypedVar
  , MaybeTypedVar'
  , MBlockType (..)
  , MCallableDecl
  , MCallableDecl' (..)
  , MCallableType (..)
  , MDecl (..)
  , MExpr
  , MExpr' (..)
  , MModifier (..)
  , MModule
  , MModule' (..)
  , MModuleExpr
  , MModuleExpr' (..)
  , MModuleDep
  , MModuleDep' (..)
  , MModuleDepType (..)
  , MModuleMorphism (..)
  , MModuleType (..)
  , MNamedRenaming
  , MNamedRenaming' (..)
  , MPackage
  , MPackage' (..)
  , MPackageDep
  , MPackageDep' (..)
  , MRenaming
  , MRenaming' (..)
  , MRenamingBlock
  , MRenamingBlock' (..)
  , MRenamingBlockType (..)
  , MSatisfaction
  , MSatisfaction' (..)
  , MTopLevelDecl (..)
  , MType
  , MTypeDecl
  , MTypeDecl' (..)
  , MVar (..)
  , MVarMode (..)
  , TypedVar
  , TypedVar'
  -- ** AST nodes after the parsing phase
  , Parsed
  , ParsedCallableDecl
  , ParsedDecl
  , ParsedExpr
  , ParsedMaybeTypedVar
  , ParsedModule
  , ParsedModuleDep
  , ParsedModuleExpr
  , ParsedNamedRenaming
  , ParsedPackage
  , ParsedRenaming
  , ParsedRenamingBlock
  , ParsedSatisfaction
  , ParsedTopLevelDecl
  , ParsedTypeDecl
  , ParsedTypedVar
  -- ** AST nodes after the type checking phase
  , Tc
  , TcCallableDecl
  , TcDecl
  , TcExpr
  , TcMaybeTypedVar
  , TcModule
  , TcModuleDep
  , TcModuleExpr
  , TcNamedRenaming
  , TcPackage
  , TcRenaming
  , TcRenamingBlock
  , TcSatisfaction
  , TcTopLevelDecl
  , TcTypeDecl
  , TcTypedVar
  -- ** \"Primitive\" Magnolia types
  , pattern Pred
  , pattern Unit
  -- ** Utils
  , getCallableDecls
  , getCallableDeclsAndModifiers
  , getModules
  , getNamedRenamings
  , getTypeDecls
  , getTypeDeclsAndModifiers
  , moduleDecls
  , moduleExprDecls
  , toBackend
  -- * Classes
  , HasDependencies (..)
  , HasName (..)
  , HasSrcCtx (..)
  -- * Annotation utils
  -- ** Annotation types
  , AbstractDeclOrigin
  , ConcreteDeclOrigin (GeneratedBuiltin)
  , DeclOrigin (..)
  , ExternalDeclDetails
  , SrcCtx (..)
  -- ** Annotation constructors and projections
  , concreteDeclOriginRequirements
  , externalDeclBackend
  , externalDeclElementName
  , externalDeclFilePath
  , externalDeclModuleInfo
  , externalDeclModuleName
  , externalDeclRequirements
  , mkConcreteLocalDecl
  , transformRequirements
  -- ** Annotation-related patterns
  , pattern AbstractLocalDecl
  , pattern ConcreteExternalDecl
  , pattern ConcreteImportedMagnoliaDecl
  , pattern ConcreteLocalMagnoliaDecl
  , pattern ConcreteMagnoliaDecl
  -- ** Annotation wrapper utils
  , Ann (..)
  , XAnn
  , (<$$>)
  , (<$$)
  -- * Parsing utils
  , PackageHead (..)
  -- * Compilation phases
  , PhParse
  , PhCheck
  , PhCodeGen
  -- * Re-exporting backends for codegen
  , Backend (..)
  -- * Repl utils
  , Command (..)
  )
  where

import Control.Monad (join)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Void

import Backend
import Env
import Err
import {-# SOURCE #-} Magnolia.PPrint
import Monad

-- === repl utils ===

data Command = LoadPackage String
             | ReloadPackage String
             | InspectModule Name -- TODO: add optional package name
             | InspectPackage Name
             | ListPackages
             | ShowMenu

-- === preprocessing utils ===

data PackageHead = PackageHead { _packageHeadPath :: FilePath
                               , _packageHeadFileContent :: String
                               , _packageHeadName :: FullyQualifiedName
                               , _packageHeadImports :: [FullyQualifiedName]
                               , _packageHeadSrcCtx :: SrcCtx
                               }
                   deriving (Eq, Show)

-- === useful type aliases ===

type Tc e = Ann PhCheck e
type TcPackage = MPackage PhCheck
type TcTopLevelDecl = MTopLevelDecl PhCheck
type TcNamedRenaming = MNamedRenaming PhCheck
type TcRenamingBlock = MRenamingBlock PhCheck
type TcRenaming = MRenaming PhCheck
type TcModule = MModule PhCheck
type TcModuleDep = MModuleDep PhCheck
type TcModuleExpr = MModuleExpr PhCheck
type TcDecl = MDecl PhCheck
type TcCallableDecl = MCallableDecl PhCheck
type TcTypeDecl = MTypeDecl PhCheck
type TcExpr = MExpr PhCheck
type TcTypedVar = TypedVar PhCheck
type TcMaybeTypedVar = MaybeTypedVar PhCheck
type TcSatisfaction = MSatisfaction PhCheck

type Parsed e = Ann PhParse e
type ParsedPackage = MPackage PhParse
type ParsedTopLevelDecl = MTopLevelDecl PhParse
type ParsedNamedRenaming = MNamedRenaming PhParse
type ParsedRenamingBlock = MRenamingBlock PhParse
type ParsedRenaming = MRenaming PhParse
type ParsedModule = MModule PhParse
type ParsedModuleDep = MModuleDep PhParse
type ParsedModuleExpr = MModuleExpr PhParse
type ParsedDecl = MDecl PhParse
type ParsedCallableDecl = MCallableDecl PhParse
type ParsedTypeDecl = MTypeDecl PhParse
type ParsedExpr = MExpr PhParse
type ParsedTypedVar = TypedVar PhParse
type ParsedMaybeTypedVar = MaybeTypedVar PhParse
type ParsedSatisfaction = MSatisfaction PhParse

-- Ann [compilation phase] [node type]
data Ann p e = Ann { _ann :: XAnn p e
                   , _elem :: e p
                   }

instance Eq (e p) => Eq (Ann p e) where
  Ann _ e1 == Ann _ e2 = e1 == e2

instance Ord (e p) => Ord (Ann p e) where
  Ann _ e1 `compare` Ann _ e2 = e1 `compare` e2

-- TODO: display annotation using UndecidableInstances?
instance Show (e p) => Show (Ann p e) where
  show = show . _elem

(<$$>) :: XAnn p e ~ XAnn p' e' => (e p -> e' p') -> Ann p e -> Ann p' e'
(<$$>) f e = Ann { _ann = _ann e, _elem = f (_elem e) }

(<$$) :: XAnn p e ~ XAnn p' e' => e' p' -> Ann p e -> Ann p' e'
(<$$) e (Ann ann _) = Ann { _ann = ann, _elem = e }

-- === AST ===

-- $mgAst
--
-- The Magnolia AST defined here follows the patterns described in the
-- [Trees that Grow](https://www.microsoft.com/en-us/research/uploads/prod/2016/11/trees-that-grow.pdf)
-- paper to carry different types of annotations depending on the current
-- compiler phase.
--
-- When two types named T and T\' are defined, T is an annotated version of
-- T\', i.e. it is a parameterized type synonym of the form
-- > type T p = Ann p T'

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

deriving instance Eq (MTopLevelDecl PhCheck)

type MNamedRenaming p = Ann p MNamedRenaming'
data MNamedRenaming' p = MNamedRenaming Name (MRenamingBlock p)

deriving instance Eq (MNamedRenaming' PhCheck)

type MSatisfaction p = Ann p MSatisfaction'
data MSatisfaction' p =
  MSatisfaction Name (MModuleExpr p) (Maybe (MModuleExpr p)) (MModuleExpr p)

deriving instance Eq (MSatisfaction' PhCheck)

type MModule p = Ann p MModule'
data MModule' p = MModule MModuleType Name (MModuleExpr p)

deriving instance Eq (MModule' PhCheck)

-- TODO: make renaming a module morphism
type MModuleExpr p = Ann p MModuleExpr'
data MModuleExpr' p =
    MModuleDef (XPhasedContainer p (MDecl p)) [MModuleDep p]
  | MModuleRef (XRef p)
  | MModuleAsSignature (XRef p)
  | MModuleTransform (MModuleMorphism p) (XTransformTarget p)
  | MModuleExternal ExternalModuleInfo FullyQualifiedName (XExternalModule p)

-- | 'ExternalModuleInfo' stores the information of which functions within the
-- module determine the dimensions with which to make kernel calls when needed,
-- as well as a list of global procedures.
data ExternalModuleInfo = ExternalModuleInfo'Cxx
                        | ExternalModuleInfo'Cuda CudaDim3 [Name]
                        | ExternalModuleInfo'JavaScript
                        | ExternalModuleInfo'Python
                          deriving (Eq, Ord, Show)

-- TODO: Int can be negative, do we care?
data CudaDim3 = CudaDim3 Name Name Name
                deriving (Eq, Ord, Show)

deriving instance Eq (MModuleExpr' PhCheck)

data MModuleMorphism p =
    MModuleMorphism'ToSignature
  | MModuleMorphism'Rename (MRenamingBlock p)
  | MModuleMorphism'RewriteWith (MModuleExpr p) Int
  | MModuleMorphism'ImplementWith (MModuleExpr p)

deriving instance Eq (MModuleMorphism PhCheck)

-- Expose out for DAG building
type MModuleDep p = Ann p MModuleDep'
-- Represents a dependency to a module with an associated list of renaming
-- blocks, as well as whether to only extract the signature of the dependency.
data MModuleDep' p =
  MModuleDep { _mmoduleDepType :: MModuleDepType
             , _mmoduleDepModuleExpr :: MModuleExpr p
             }

deriving instance Eq (MModuleDep' PhCheck)

data MModuleDepType = MModuleDepRequire | MModuleDepUse
                      deriving (Eq, Show)

-- | There are two types of renaming blocks: partial renaming blocks, and
-- total renaming blocks. A partial renaming block is a renaming block that
-- may contain source names that do not exist in the module expression it is
-- applied to. In contrast, a total renaming block expects that all its source
-- names exist in the module expression it is applied to.
type MRenamingBlock p = Ann p MRenamingBlock'
data MRenamingBlock' p = MRenamingBlock MRenamingBlockType [MRenaming p]

deriving instance Eq (MRenamingBlock' PhCheck)

data MRenamingBlockType = PartialRenamingBlock | TotalRenamingBlock
                          deriving (Eq, Show)

type MRenaming p = Ann p MRenaming'
data MRenaming' p = InlineRenaming InlineRenaming
                  | RefRenaming (XRef p)

deriving instance Eq (MRenaming' PhCheck)

type InlineRenaming = (Name, Name)

data MModuleType = Signature
                 | Concept
                 | Implementation
                 | Program
                   deriving (Eq, Show)

data MDecl p = MTypeDecl [MModifier] (MTypeDecl p)
             | MCallableDecl [MModifier] (MCallableDecl p)

deriving instance Eq (MDecl PhParse)
deriving instance Ord (MDecl PhParse)
deriving instance Show (MDecl PhParse)

deriving instance Eq (MDecl PhCheck)
deriving instance Ord (MDecl PhCheck)
deriving instance Show (MDecl PhCheck)

data MModifier = Require -- TODO: add other modifiers, such as 'assume', etc
                 deriving (Eq, Ord, Show)

type MTypeDecl p = Ann p MTypeDecl'
newtype MTypeDecl' p = Type { _typeName :: MType }
                       deriving (Eq, Ord, Show)

type MCallableDecl p = Ann p MCallableDecl'
data MCallableDecl' p =
  Callable { _callableType :: MCallableType
           , _callableName :: Name
           , _callableArgs :: [TypedVar p]
           , _callableReturnType :: MType
           , _callableGuard :: CGuard p
           , _callableBody :: CBody p
           }

deriving instance Eq (MCallableDecl' PhParse)
deriving instance Ord (MCallableDecl' PhParse)
deriving instance Show (MCallableDecl' PhParse)

deriving instance Eq (MCallableDecl' PhCheck)
deriving instance Ord (MCallableDecl' PhCheck)
deriving instance Show (MCallableDecl' PhCheck)

-- TODO: at the moment, with only C++ as a backend, we assume any external body
--       comes from C++. When we actually implement other backends, we will need
--       to carry information about the external bodies. For instance, a file
--       can contain both a JS and a C++ implementation for the same external
--       functions. These two concrete implementations will be joinable, since
--       they are backend-dependent (and there is always only one backend).
--       This will need to be handled at the ConcreteDecl level as well.
data CBody p = ExternalBody (XExternalBody p)
             | EmptyBody
             | MagnoliaBody (MExpr p)
             | BuiltinBody

deriving instance Eq (CBody PhParse)
deriving instance Ord (CBody PhParse)
deriving instance Show (CBody PhParse)

deriving instance Eq (CBody PhCheck)
deriving instance Ord (CBody PhCheck)
deriving instance Show (CBody PhCheck)

type CGuard p = Maybe (MExpr p)

type MType = Name

-- | Predicate type, used for conditionals.
pattern Pred :: MType
pattern Pred = GenName "Predicate"

-- | Unit type, used for stateful computations.
pattern Unit :: MType
pattern Unit = GenName "Unit"

data MCallableType = Axiom | Function | Predicate | Procedure
                    deriving (Eq, Ord, Show)

-- TODO: make a constructor for coercedexpr to remove (Maybe MType) from calls?
type MExpr p = Ann p MExpr'
data MExpr' p = MVar (MaybeTypedVar p)
              -- TODO: add Procedure/FunctionLike namespaces to Name?
              | MCall Name [MExpr p] (Maybe MType)
              | MBlockExpr MBlockType (NE.NonEmpty (MExpr p))
              | MValue (MExpr p)
              | MLet (MaybeTypedVar p) (Maybe (MExpr p))
              | MIf (MExpr p) (MExpr p) (MExpr p)
              | MAssert (MExpr p)
              | MSkip
                deriving (Eq, Ord, Show)

data MBlockType = MValueBlock | MEffectfulBlock
                  deriving (Eq, Ord, Show)

type TypedVar p = Ann p TypedVar'
type TypedVar' = MVar MType

type MaybeTypedVar p = Ann p MaybeTypedVar'
type MaybeTypedVar' = MVar (Maybe MType)

data MVar typAnnType p = Var { _varMode :: MVarMode
                             , _varName :: Name
                             , _varType :: typAnnType
                             }
                         deriving (Eq, Ord, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data MVarMode = MObs | MOut | MUnk | MUpd
                deriving (Eq, Ord, Show)

-- == annotation utils ==

-- | Wraps the necessary information associated with external declarations.
-- We prevent access to the constructor to ensure we only build these details
-- through 'mkConcreteLocalDecl'.
-- TODO: perhaps this data structure should be different depending on the
-- backend considered. For the moment, we consider the requirements of C++
-- only, and will revisit later if necessary.
data ExternalDeclDetails =
  ExternalDeclDetails { -- | The information associated with the external
                        -- module that contains the declaration
                        _externalDeclModuleInfo :: ExternalModuleInfo
                        -- | The external file path in which the external
                        -- declaration can be found
                      , _externalDeclFilepath :: FilePath
                        -- | The name of the module in which the external
                        -- declaration can be found within the relevant file
                      , _externalDeclModuleName :: Name
                        -- | The name of the declaration within the external
                        -- module
                      , _externalDeclElementName :: Name
                        -- | The requirements to instantiate in the external
                        -- module. The key in the map corresponds to the
                        -- original required declaration, while the value
                        -- corresponds to the same declaration with
                        -- relevant renamings applied to it in the module
                        -- considered.
                      , _externalDeclRequirements :: M.Map TcDecl TcDecl
                      }
  deriving (Eq, Ord, Show)

-- | The backend corresponding to the external declaration's
-- '_externalDeclModuleInfo'.
externalDeclBackend :: ExternalDeclDetails -> Backend
externalDeclBackend = toBackend . _externalDeclModuleInfo

-- | See '_externalDeclModuleInfo'.
externalDeclModuleInfo :: ExternalDeclDetails -> ExternalModuleInfo
externalDeclModuleInfo = _externalDeclModuleInfo

-- | See '_externalDeclFilepath'.
externalDeclFilePath :: ExternalDeclDetails -> FilePath
externalDeclFilePath = _externalDeclFilepath

-- | See '_externalDeclModuleName'.
externalDeclModuleName :: ExternalDeclDetails -> Name
externalDeclModuleName = _externalDeclModuleName

-- | See '_externalDeclElementName'.
externalDeclElementName :: ExternalDeclDetails -> Name
externalDeclElementName = _externalDeclElementName

-- | See '_externalDeclRequirements'.
externalDeclRequirements :: ExternalDeclDetails -> M.Map TcDecl TcDecl
externalDeclRequirements = _externalDeclRequirements

-- | Wraps the source location information of a declaration.
data DeclOrigin
  -- | Annotates a local declaration. At the module level, any declaration
  -- carries a local annotation. At the package level, only the modules
  -- defined in the current module should carry this annotation.
  = LocalDecl SrcCtx
  -- | Annotates an imported declaration. At the module level, no declaration
  -- carries such an annotation at the moment (TODO: should we change that
  -- behavior?). The reason is that within a module, declarations always have
  -- a corresponding local declaration. At the package level, all the modules
  -- imported from different packages should carry this annotation.
  | ImportedDecl FullyQualifiedName SrcCtx
    deriving (Eq, Show)

instance Ord DeclOrigin where
  compare declO1 declO2 = compare (srcCtx declO1) (srcCtx declO2)

-- | Wraps the source location information of a concrete declaration (i.e. a
-- declaration that is not required). If the inner constructor is Left, then
-- the origin of the declaration is interpreted as being internal (i.e. the
-- declaration is defined purely using Magnolia code). A contrario, if the
-- inner constructor is Right, then the origin of the declaration is interpreted
-- as being external (i.e. the declaration is defined to exist on some
-- backend). In that case, the declaration is also accompanied with the fully
-- qualified name that corresponds to the function in the external context,
-- and the relevant backend.
-- The second constructor, 'GeneratedBuiltin', is used specifically for
-- Magnolia declarations that are built in and generated by the compiler.
data ConcreteDeclOrigin =
    ConcreteDeclOrigin (Either DeclOrigin
                               (DeclOrigin, ExternalDeclDetails))
  | GeneratedBuiltin
    deriving (Eq, Show)

instance Ord ConcreteDeclOrigin where
  compare conDeclO1 conDeclO2 = compare (srcCtx conDeclO1) (srcCtx conDeclO2)

-- | Exposed constructor for ConcreteDeclOrigins. If a backend and an external
-- name are provided, produces an external ConcreteDeclOrigin. Otherwise,
-- produces an internal one. The module name provided with the backend should
-- correspond to the name of the external data structure for the relevant
-- backend.
-- TODO: fix data types to get more type safety when it comes to the names.
mkConcreteLocalDecl :: Maybe (ExternalModuleInfo, FullyQualifiedName, [TcDecl])
                    -> SrcCtx
                    -> Name
                    -> MgMonad ConcreteDeclOrigin
mkConcreteLocalDecl mexternalInfo src name = case mexternalInfo of
  Nothing -> pure $ ConcreteLocalMagnoliaDecl src
  Just externalInfo -> mkExternalDeclDetails externalInfo src name
    >>= \extDeclDetails -> pure $ ConcreteLocalExternalDecl extDeclDetails src

-- | Safely builds external declaration details based on a backend, its fully
-- qualified name (where the '_scopeName' corresponds to the path to the
-- external file, and the '_targetName' corresponds to the name of the
-- external module within that file), the abstract requirements for the
-- external module, the source information of the external declaration within
-- the Magnolia package, and the name of the declaration within the external
-- module.
-- Throws an error if the path to the external file is 'Nothing'.
mkExternalDeclDetails :: (ExternalModuleInfo, FullyQualifiedName, [TcDecl])
                      -> SrcCtx
                      -> Name
                      -> MgMonad ExternalDeclDetails
mkExternalDeclDetails (extModuleInfo, extModuleFqn, requirements) src declName =
  case _scopeName extModuleFqn of
    Nothing -> throwLocatedE MiscErr src $ "external " <>
      pshow (toBackend extModuleInfo) <> " block " <> pshow extModuleFqn <>
      " was specified without an include path"
    Just (Name _ filepath) -> pure $
      ExternalDeclDetails
        extModuleInfo filepath (_targetName extModuleFqn)
        declName (M.fromList (map (\d -> (d, d)) requirements))

-- | Extracts the requirements of an external module that are wrapped within a
-- 'ConcreteDeclOrigin' if relevant.
-- These requirements are used to disambiguate between external callables that
-- expose the same prototype but may rely on different assumptions. For example,
-- given the following implementation IExt:
--
-- implementation IExt = external C++ SomeFile.SomeStruct {
--   require type U;
--   type T;
--   function f(): T;
-- }
--
-- the lines \'use IExt[U => U];\' and \'use IExt[U => V];\' each import a
-- different version of f() in scope: one that assumes that U is bound to U,
-- and one that assumes that U is bound to V.
concreteDeclOriginRequirements :: ConcreteDeclOrigin -> [TcDecl]
concreteDeclOriginRequirements conDeclO = case conDeclO of
  GeneratedBuiltin -> []
  ConcreteMagnoliaDecl _ -> []
  ConcreteExternalDecl _ extDeclDetails ->
    M.elems $ externalDeclRequirements extDeclDetails

-- | Applies a transformation to the requirements of an external module
-- that are wrapped within a 'ConcreteDeclOrigin' if relevant.
transformRequirements :: (TcDecl -> TcDecl)
                      -> ConcreteDeclOrigin
                      -> ConcreteDeclOrigin
transformRequirements f conDeclO = case conDeclO of
  GeneratedBuiltin -> conDeclO
  ConcreteMagnoliaDecl _ -> conDeclO
  ConcreteExternalDecl declO extDeclDetails ->
    let newRequirements = M.map f (_externalDeclRequirements extDeclDetails)
    in ConcreteExternalDecl declO
        extDeclDetails { _externalDeclRequirements = newRequirements }

-- | Wraps the source location information of an abstract declaration (i.e. a
-- declaration that is required).
newtype AbstractDeclOrigin = AbstractDeclOrigin DeclOrigin
                             deriving (Eq, Ord)

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

  XAnn PhParse MModuleExpr' = SrcCtx
  XAnn PhCheck MModuleExpr' = SrcCtx

  XAnn PhParse MModuleDep' = SrcCtx
  XAnn PhCheck MModuleDep' = (SrcCtx, [FullyQualifiedName])

  XAnn PhParse MRenamingBlock' = SrcCtx
  XAnn PhCheck MRenamingBlock' = SrcCtx

  XAnn PhParse MRenaming' = SrcCtx
  XAnn PhCheck MRenaming' = DeclOrigin

  XAnn PhParse MTypeDecl' = SrcCtx
  XAnn PhCheck MTypeDecl' = ( Maybe ConcreteDeclOrigin
                            , NE.NonEmpty AbstractDeclOrigin
                            )

  XAnn PhParse MCallableDecl' = SrcCtx
  XAnn PhCheck MCallableDecl' = ( Maybe ConcreteDeclOrigin
                                , NE.NonEmpty AbstractDeclOrigin
                                )

  XAnn PhParse MExpr' = SrcCtx
  XAnn PhCheck MExpr' = MType

  XAnn PhParse (MVar _) = SrcCtx
  XAnn PhCheck (MVar _) = MType

-- === other useful type families ===

type family XPhasedContainer p e where
  XPhasedContainer PhParse e = [e]
  XPhasedContainer PhCheck e = M.Map Name [e]

-- | The goal of XRef is to statically prevent the existence of references to
-- named top level elements after the consistency/type checking phase.
type family XRef p where
  XRef PhParse = FullyQualifiedName
  XRef PhCheck = Void

-- TODO: document
type family XTransformTarget p where
  XTransformTarget PhParse = MModuleExpr PhParse
  XTransformTarget PhCheck = Void

-- | The goal of XExternalModule is to flatten external module expressions
-- during the type checking phase. See 'MModuleExpr\''.
type family XExternalModule p where
  XExternalModule PhParse = ParsedModuleExpr
  XExternalModule PhCheck = Void

-- | The goal of XExternalBody is to allow refining parsed declarations into
-- external ones after the parsing phase – more cleanly separating
-- interpretation from parsing.
type family XExternalBody p where
  XExternalBody PhParse = Void
  XExternalBody PhCheck = ()

-- === standalone show instances ===

deriving instance Show (MRenaming' PhCheck)
deriving instance Show (MRenamingBlock' PhCheck)
deriving instance Show (MModuleDep' PhCheck)
deriving instance Show (MNamedRenaming' PhCheck)
deriving instance Show (MModule' PhCheck)
deriving instance Show (MModuleExpr' PhCheck)
deriving instance Show (MModuleMorphism PhCheck)
deriving instance Show (MSatisfaction' PhCheck)
deriving instance Show (MTopLevelDecl PhCheck)
deriving instance Show (MPackage' PhCheck)

-- === useful typeclasses ===

class HasName n where
  nodeName :: n -> Name

instance HasName PackageHead where
  nodeName = fromFullyQualifiedName  . _packageHeadName

instance HasName (e p) => HasName (Ann p e) where
  nodeName = nodeName . _elem

instance HasName (MPackage' p) where
  nodeName = _packageName

instance HasName (MPackageDep' p) where
  nodeName (MPackageDep name) = fromFullyQualifiedName name

instance HasName (MTopLevelDecl p) where
  nodeName topLevelDecl = case topLevelDecl of
    MNamedRenamingDecl namedRenaming -> nodeName namedRenaming
    MModuleDecl modul -> nodeName modul
    MSatisfactionDecl satisfaction -> nodeName satisfaction

instance HasName (MNamedRenaming' p) where
  nodeName (MNamedRenaming name _) = name

instance HasName (MModule' p) where
  nodeName (MModule _ name _) = name

instance HasName (MSatisfaction' p) where
  nodeName (MSatisfaction name _ _ _) = name

instance HasName (MDecl p) where
  nodeName decl = case decl of
    MTypeDecl _ tdecl -> nodeName tdecl
    MCallableDecl _ cdecl -> nodeName cdecl

instance HasName (MTypeDecl' p) where
  nodeName (Type name) = name

instance HasName (MCallableDecl' p) where
  nodeName (Callable _ name _ _ _ _) = name

instance HasName (MVar typAnnType p) where
  nodeName (Var _ name _) = name

class HasDependencies a where
  dependencies :: a -> [FullyQualifiedName]

instance HasDependencies (e p) => HasDependencies (Ann p e) where
  dependencies = dependencies . _elem

instance HasDependencies PackageHead where
  dependencies = _packageHeadImports

instance HasDependencies (MModule' PhParse) where
  dependencies (MModule _ _ moduleExpr) = dependencies moduleExpr

instance HasDependencies (MModule' PhCheck) where
  dependencies (MModule _ _ moduleExpr) = dependencies moduleExpr

instance HasDependencies (MModuleExpr' PhParse) where
  dependencies moduleExpr = case moduleExpr of
    MModuleDef _ deps -> join $
      map (dependencies . _elem . _mmoduleDepModuleExpr . _elem) deps
    MModuleRef refName -> [refName]
    MModuleAsSignature refName -> [refName]
    MModuleTransform transformation moduleExpr' -> (case transformation of
      MModuleMorphism'ToSignature -> []
      MModuleMorphism'Rename _ -> []
      MModuleMorphism'RewriteWith rewriteRules _ -> dependencies rewriteRules
      MModuleMorphism'ImplementWith generator -> dependencies generator) <>
        dependencies moduleExpr'
    MModuleExternal _ _ moduleExpr' -> dependencies moduleExpr'

instance HasDependencies (MModuleExpr' PhCheck) where
  dependencies modul = case modul of
    MModuleDef _ deps -> join $ map (snd . _ann) deps
    MModuleRef v -> absurd v
    MModuleAsSignature v -> absurd v
    MModuleTransform _ v -> absurd v
    MModuleExternal _ _ v -> absurd v

instance HasDependencies (MNamedRenaming' PhParse) where
  dependencies (MNamedRenaming _ renamingBlock) =
    dependencies renamingBlock

instance HasDependencies (MRenamingBlock' PhParse) where
  dependencies (MRenamingBlock _ renamings) =
    foldr (\(Ann _ r) acc -> case r of RefRenaming n -> n:acc ; _ -> acc) []
          renamings

instance HasDependencies (MPackage' p) where
  dependencies (MPackage _ _ deps) =
    map (\(Ann _ (MPackageDep depName)) -> depName) deps

class HasSrcCtx a where
  srcCtx :: a -> SrcCtx

instance HasSrcCtx SrcCtx where
  srcCtx = id

instance HasSrcCtx PackageHead where
  srcCtx = _packageHeadSrcCtx

instance HasSrcCtx DeclOrigin where
  srcCtx declO = case declO of
    LocalDecl src -> src
    ImportedDecl _ src -> src

instance HasSrcCtx ConcreteDeclOrigin where
  srcCtx (ConcreteDeclOrigin edeclO) = case edeclO of
    Left declO -> srcCtx declO
    Right (declO, _) -> srcCtx declO
  srcCtx GeneratedBuiltin = SrcCtx Nothing

instance HasSrcCtx AbstractDeclOrigin where
  srcCtx (AbstractDeclOrigin declO) = srcCtx declO

instance HasSrcCtx (SrcCtx, a) where
  srcCtx (src, _) = src

instance HasSrcCtx (XAnn p e) => HasSrcCtx (Ann p e) where
  srcCtx = srcCtx . _ann

-- === useful patterns ===

pattern AbstractLocalDecl :: SrcCtx -> AbstractDeclOrigin
pattern AbstractLocalDecl src = AbstractDeclOrigin (LocalDecl src)

pattern ConcreteExternalDecl :: DeclOrigin
                             -> ExternalDeclDetails
                             -> ConcreteDeclOrigin
pattern ConcreteExternalDecl declO extDeclDetails =
  ConcreteDeclOrigin (Right (declO, extDeclDetails))

pattern ConcreteMagnoliaDecl :: DeclOrigin
                             -> ConcreteDeclOrigin
pattern ConcreteMagnoliaDecl declO = ConcreteDeclOrigin (Left declO)

{-# COMPLETE ConcreteExternalDecl, ConcreteMagnoliaDecl, GeneratedBuiltin #-}

pattern ConcreteImportedMagnoliaDecl
  :: FullyQualifiedName -> SrcCtx -> ConcreteDeclOrigin
pattern ConcreteImportedMagnoliaDecl fqn src =
  ConcreteDeclOrigin (Left (ImportedDecl fqn src))

pattern ConcreteLocalMagnoliaDecl :: SrcCtx -> ConcreteDeclOrigin
pattern ConcreteLocalMagnoliaDecl src =
  ConcreteDeclOrigin (Left (LocalDecl src))

pattern ConcreteLocalExternalDecl :: ExternalDeclDetails
                                  -> SrcCtx
                                  -> ConcreteDeclOrigin
pattern ConcreteLocalExternalDecl extDeclDetails src =
  ConcreteDeclOrigin (Right (LocalDecl src, extDeclDetails))

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

-- | Extracts declarations from a type checked module.
moduleDecls :: TcModule -> Env [TcDecl]
moduleDecls (Ann _ (MModule _ _ tcModuleExpr)) = moduleExprDecls tcModuleExpr

-- | Extracts declarations from a type checked module expression.
moduleExprDecls :: TcModuleExpr -> Env [TcDecl]
moduleExprDecls (Ann _ moduleExpr) = case moduleExpr of
  MModuleDef decls _ -> decls
  MModuleRef v -> absurd v
  MModuleAsSignature v -> absurd v
  MModuleTransform _ v -> absurd v
  MModuleExternal _ _ v -> absurd v

-- === module declarations manipulation ===

getTypeDeclsAndModifiers :: Foldable t
                         => t (MDecl p)
                         -> [([MModifier], MTypeDecl p)]
getTypeDeclsAndModifiers = foldr extractTypeAndModifiers []
  where
    extractTypeAndModifiers :: MDecl p
                            -> [([MModifier], MTypeDecl p)]
                            -> [([MModifier], MTypeDecl p)]
    extractTypeAndModifiers decl acc = case decl of
      MTypeDecl modifiers tdecl -> (modifiers, tdecl):acc
      _ -> acc

getTypeDecls :: Foldable t => t (MDecl p) -> [MTypeDecl p]
getTypeDecls = map snd . getTypeDeclsAndModifiers

getCallableDeclsAndModifiers :: Foldable t
                             => t (MDecl p)
                             -> [([MModifier], MCallableDecl p)]
getCallableDeclsAndModifiers = foldr extractCallableAndModifiers []
  where
    extractCallableAndModifiers :: MDecl p
                                -> [([MModifier], MCallableDecl p)]
                                -> [([MModifier], MCallableDecl p)]
    extractCallableAndModifiers decl acc = case decl of
      MCallableDecl modifiers cdecl -> (modifiers, cdecl):acc
      _ -> acc

getCallableDecls :: Foldable t => t (MDecl p) -> [MCallableDecl p]
getCallableDecls = map snd . getCallableDeclsAndModifiers

-- === other utils ===

toBackend :: ExternalModuleInfo -> Backend
toBackend extModuleInfo = case extModuleInfo of
  ExternalModuleInfo'Cxx -> Cxx
  ExternalModuleInfo'Cuda {} -> Cuda
  ExternalModuleInfo'JavaScript -> JavaScript
  ExternalModuleInfo'Python -> Python