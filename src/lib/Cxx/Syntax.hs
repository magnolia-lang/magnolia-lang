{-# LANGUAGE OverloadedStrings #-}

module Cxx.Syntax (
    CxxAccessSpec (..)
  , CxxDef (..)
  , CxxExpr (..)
  , CxxFunctionDef (..)
  , CxxLambdaCaptureDefault (..)
  , CxxModule (..)
  , CxxModuleRefSpecialization (..)
  , CxxName
  , CxxNamespaceName
  , CxxPackage (..)
  , CxxStmt (..)
  , CxxStmtBlock
  , CxxStmtInline (..)
  , CxxType (..)
  , CxxVar (..)
  , mkCxxName
  , mkCxxType
  )
  where

import qualified Data.Map as M
import Data.Maybe (fromMaybe)
import qualified Data.Set as S
import Data.Text.Prettyprint.Doc

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util

p :: Pretty a => a -> Doc ann
p = pretty

cxxIndent :: Int
cxxIndent = 4

newtype CxxName = CxxName String
                deriving Show

-- TODO: handle operators and "invalid C++ names", maybe?
-- | Makes a CxxName out of a Name. The function may throw an error if the name
-- can not be used in C++. The infix functions that are allowed in Magnolia
-- (e.g. '_+_', or '_*_'...) are translated using custom translation rules.
mkCxxName :: Name -> MgMonad CxxName
mkCxxName n | _name n `S.member` cxxKeywords = throwNonLocatedE MiscErr $
  "could not make '" <> pshow n <> "' a C++ identifier, because it is a " <>
  "reserved keyword"
mkCxxName n = return $ fromMaybe (CxxName $ _name n) (mkCustomCxxName n)

-- TODO: at the moment, we only give special translation rules for given names.
--       Eventually, we can come up with custom translation rules definable
--       directly in Magnolia for calls to certain functions/procedures.
mkCustomCxxName :: Name -> Maybe CxxName
mkCustomCxxName (Name _ s) = CxxName <$> M.lookup s customCxxTranslations

-- TODO: deal better with this. For the moment, this is good enough.
-- | Defines custom translation rules for special infix functions allowed in
-- Magnolia but whose name would not yield a valid C++ identifier.
customCxxTranslations :: M.Map String String
customCxxTranslations = M.fromList
  [ ("_=_", "assign") -- This may not be necessary, as "_=_" can never be
                      -- custom declared (or renamed) by the user for the
                      -- moment. We should thus be able to ignore code
                      -- generation for that.
  , ("+_", "unary_add")
  , ("-_", "unary_sub")
  , ("!_", "logical_not")
  , ("~_", "bitwise_not")
  , ("_*_", "mul")
  , ("_/_", "div")
  , ("_%_", "mod")
  , ("_+_", "binary_add")
  , ("_-_", "binary_sub")
  , ("_<<_", "left_shift")
  , ("_>>_", "right_shift")
  , ("_.._", "range")
  , ("_<_", "lt")
  , ("_>_", "gt")
  , ("_<=_", "le")
  , ("_>=_", "ge")
  , ("_==_", "eq")
  , ("_!=_", "neq")
  , ("_===_", "triple_eq")
  , ("_!==_", "triple_neq")
  , ("_&&_", "logical_and")
  , ("_||_", "logical_or")
  , ("_=>_", "implies")
  , ("_<=>_", "iff")
  ]

-- | Makes a CxxType out of a Magnolia type. The function may throw an error
-- if the name of the type can not be used in C++.
mkCxxType :: MType -> MgMonad CxxType
mkCxxType typeName = case typeName of
  Unit -> return CxxVoid
  Pred -> return CxxBool
  _ -> CxxCustomType <$> mkCxxName typeName

-- | The set of reserved keywords in C++.
cxxKeywords :: S.Set String
cxxKeywords = S.fromList
  [ "alignas", "alignof", "and", "and_eq"
  , "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept"
  , "auto", "bitand", "bitor", "bool"
  , "break", "case", "catch", "char"
  , "char8_t", "char16_t", "char32_t", "class"
  , "compl", "concept", "const", "consteval"
  , "constexpr", "constinit", "const_cast", "continue"
  , "co_await", "co_return", "co_yield", "decltype"
  , "default", "delete", "do", "double"
  , "dynamic_cast", "else", "enum", "explicit"
  , "export", "extern", "false", "float"
  , "for", "friend", "goto", "if"
  , "inline", "int", "long", "mutable"
  , "namespace", "new", "noexcept", "not"
  , "not_eq", "nullptr", "operator", "or"
  , "or_eq", "private", "protected", "public"
  , "reflexpr", "register", "reinterpret_cast", "requires"
  , "return", "short", "signed", "sizeof"
  , "static", "static_assert", "static_cast", "struct"
  , "switch", "synchronized", "template", "this"
  , "thread_local", "throw", "true", "try"
  , "typedef", "typeid", "typename", "union"
  , "unsigned", "using", "virtual", "void"
  , "volatile", "wchar_t", "while", "xor"
  , "xor_eq"]

-- | The equivalent of a Magnolia package in C++. A C++ package contains a list
-- of dependencies, as well as a list of locally defined modules.
data CxxPackage =
  CxxPackage { _cxxPackageImports :: [CxxImport]
             , _cxxPackageModules :: [CxxModule]
             }
  deriving Show

newtype CxxImport = CxxImport FullyQualifiedName
                    deriving Show

-- | The equivalent of a Magnolia module in C++.
data CxxModule =
  CxxModule { -- | The namespaces wrapping the module.
              _cxxModuleNamespaces :: [CxxNamespaceName]
              -- | The name of the module.
            , _cxxModuleName :: CxxModuleName
              -- | The base classes to inherit from, along with their
              -- potential template specializations.
            , _cxxModuleBaseClasses :: [CxxModuleRefSpecialization]
              -- | The outer template parameters of the module. They should
              -- correspond to the concrete types from the corresponding
              -- Magnolia module.
            , _cxxModuleOuterTemplateParameters :: [CxxTemplateParameter]
              -- | The inner template parameters of the module. They should
              -- correspond to the required types from the corresponding
              -- Magnolia module.
            , _cxxModuleInnerTemplateParameters :: [CxxTemplateParameter]
              -- | The definitions within the module, along with access
              -- specifiers. The set of public definitions should correspond
              -- to the API exposed by the corresponding Magnolia module.
            , _cxxModuleDefinitions :: [(CxxAccessSpec, CxxDef)]
            }
  deriving Show

-- | A reference to a (possibly templated) C++ module along with the necessary
-- arguments for template specialization.
data CxxModuleRefSpecialization =
  CxxModuleRefSpecialization [CxxNamespaceName] CxxModuleName [CxxTypeName]
  deriving Show

type CxxModuleName = CxxName
type CxxNamespaceName = CxxName
type CxxTypeName = CxxName

-- | The equivalent of a Magnolia declaration in C++.
data CxxDef
    -- | A C++ type definition. 'CxxTypeDef src tgt' corresponds to the C++
    -- code 'typedef src tgt'.
  = CxxTypeDef CxxName CxxName -- TODO: do I need more?
    -- | A C++ function definition.
  | CxxFunctionDef CxxFunctionDef
  deriving Show

-- TODO: are cxxFnIsVirtual and cxxFnBody redundant?
-- | The equivalent of a Magnolia function definition in C++.
data CxxFunctionDef =
  CxxFunction { -- | Whether the function is virtual. This field is set to true
                -- whenever the corresponding Magnolia declaration is abstract.
                _cxxFnIsVirtual :: Bool
                -- | Whether the function has side effects. This field is set to
                -- true whenever the corresponding Magnolia declaration is a
                -- procedure.
              , _cxxFnHasSideEffects :: Bool
                -- | The name of the function.
              , _cxxFnName :: CxxName
                -- | The template parameters to the function. This is at least
                -- necessary when generating C++ code for Magnolia functions
                -- that are overloaded only on their return type.
              , _cxxFnTemplateParameters :: [CxxTemplateParameter]
                -- | The parameters to the function.
              , _cxxFnParams :: [CxxVar]
                -- | The return type of the function.
              , _cxxFnReturnType :: CxxType
                -- | The body of the function if it has one.
              , _cxxFnBody :: Maybe CxxStmtBlock
              }
  deriving Show

data CxxVar = CxxVar { _cxxVarIsConst :: Bool
                     , _cxxVarIsRef :: Bool
                     , _cxxVarName :: CxxName
                     , _cxxVarType :: CxxType
                     }
            deriving Show

data CxxStmt = CxxStmtBlock CxxStmtBlock
             | CxxStmtInline CxxStmtInline
             deriving Show

type CxxStmtBlock = [CxxStmt]

data CxxStmtInline = CxxAssign CxxName CxxExpr
                   | CxxVarDecl CxxVar (Maybe CxxExpr)
                   | CxxAssert CxxCond
                   | CxxIf CxxCond CxxStmt CxxStmt
                   -- ^ TODO: make second CxxStmtBlock a Maybe. This is because
                   --         else branches are not necessary when the
                   --         expression has type void.
                   | CxxExpr CxxExpr
                   | CxxReturn (Maybe CxxExpr)
                   | CxxSkip
                   deriving Show

-- TODO: should we, in Magnolia, allow assigning to a variable in a condition?
--       Probably not, since we should only have stateless computations there.
type CxxCond = CxxExpr

data CxxExpr = CxxCall CxxName [CxxTemplateParameter] [CxxExpr]
             | CxxLambdaCall CxxLambdaCaptureDefault CxxStmtBlock
             -- ^ Lambda calls can be used to translate Magnolia blocks.
             --   We always capture the entire environment in blocks, but:
             --   - value blocks capture by value (the outside environment can
             --     not be modified);
             --   - stateful blocks capture by reference (the outside
             --     environment can be modified).
             | CxxVarRef CxxName
             | CxxIfExpr CxxCond CxxExpr CxxExpr
             deriving Show

data CxxLambdaCaptureDefault = CxxLambdaCaptureDefaultValue
                             | CxxLambdaCaptureDefaultReference
                             deriving Show

data CxxType = CxxVoid | CxxBool | CxxCustomType CxxName
             deriving Show
type CxxTemplateParameter = CxxName
data CxxAccessSpec = CxxPublic | CxxPrivate | CxxProtected
  deriving (Eq, Show)

-- === pretty instances ===

instance Pretty CxxName where
  pretty (CxxName name) = p name

instance Pretty CxxModule where
  pretty (CxxModule namespaces name refSpecs outerTys innerTys defs) =
    let (outerTys', outerRefSpecs, innerTys', innerRefSpecs) =
          if null outerTys then (innerTys, refSpecs, [], [])
          else (outerTys, [], innerTys, refSpecs)
    -- 1. Open namespaces
    in align (vsep (map (\ns -> "namespace" <+> p ns <+> "{") namespaces)) <>
    line <>
    -- 2. Build class template
    (case outerTys' of [] -> "" ; _ -> mkTemplateParams outerTys' <> line) <>
    -- 3. Build class header
    "class" <+> p name <+> pRefSpecs outerRefSpecs <> "{" <> line <>
    -- TODO: add inheritance here
    -- 4. Build optional inner class header + content
    (case innerTys' of
      [] -> classContent
      _ -> mkTemplateParams innerTys' <> line <>
        "class" <+> "_" <> p name <+> pRefSpecs innerRefSpecs <> "{" <> line <>
        classContent <> line <>
        "};") <> line <>
    "};" <> line <>
    -- x. Close namespaces
    align (vsep (map (\ns -> "} //" <+> p ns) namespaces))
    where
      mkSection cxxAS = vsep $
        map (p . snd) (filter (\(as, _) -> cxxAS == as) defs)

      classContent =
        "private:" <> line <>
          indent cxxIndent (mkSection CxxPrivate) <> line <>
        "protected:" <> line <>
          indent cxxIndent (mkSection CxxProtected) <> line <>
        "public:" <> line <>
          indent cxxIndent (mkSection CxxPublic) <> line

      mkTemplateParams tys = "template <" <>
        hsep (punctuate comma (map (("class" <+>) . p) tys)) <> ">"

      pRefSpecs [] = ""
      pRefSpecs refSpecs' = ": " <>
        hsep (punctuate comma (map p refSpecs')) <> " "


instance Pretty CxxModuleRefSpecialization where
  pretty (CxxModuleRefSpecialization namespaces moduleName templateParams) =
    hcat (map (\ns -> p ns <> "::") namespaces) <> p moduleName <>
    (case templateParams of
      []  -> ""
      tps -> "<" <> hsep (punctuate comma (map p tps)) <> ">")

instance Pretty CxxDef where
  pretty (CxxTypeDef source target) = "typedef" <+> p source <+> p target <> ";"
  pretty (CxxFunctionDef fn) = p fn <> ";"

instance Pretty CxxFunctionDef where
  pretty (CxxFunction isVirtual hasSideEffects name templateParams params retTy
                      body) =
    (if not (null templateParams) then pTemplateParams <> line else "") <>
    (if isVirtual then "virtual " else "") <> p retTy <+> p name <> "(" <>
    hsep (punctuate comma (map p params)) <> ")" <>
    (if hasSideEffects then "" else " const") <>
    maybe "" ((" " <>) . p . CxxStmtBlock) body
    where
      pTemplateParams = "template <" <>
        hsep (punctuate comma (map (("typename" <+>) . p) templateParams)) <>
        ">"

instance Pretty CxxVar where
  pretty (CxxVar isConst isRef name ty) =
    (if isConst then "const " else "") <> p ty <>
    (if isRef then "&" else "") <+> p name

instance Pretty CxxStmt where
  pretty (CxxStmtBlock stmts) = "{" <> line <>
    indent cxxIndent (vsep (map p stmts)) <> line <> "}"
  pretty (CxxStmtInline stmt) = p stmt <> semi

instance Pretty CxxStmtInline where
  pretty inlineStmt = case inlineStmt of
    CxxAssign name stmt -> p name <+> "=" <+> p stmt
    CxxVarDecl var mExpr -> p var <> maybe "" ((" =" <+>) . p) mExpr
    CxxAssert cond -> "assert" <+> p cond
    CxxIf cond trueBranch falseBranch -> "if (" <> p cond <> ")" <>
      line <> p trueBranch <> line <> "else" <> line <> p falseBranch
    CxxExpr expr -> p expr
    CxxReturn mExpr -> "return" <> maybe "" ((" " <>) . p) mExpr
    CxxSkip -> ""

instance Pretty CxxExpr where
  pretty expr = case expr of
    CxxCall name templateParams args -> p name <> (case templateParams of
        [] -> ""
        _  -> "<" <> hsep (punctuate comma (map p templateParams)) <> ">") <>
      "(" <> hsep (punctuate comma (map p args)) <> ")"
    CxxLambdaCall captureDefault stmts -> p captureDefault <> "()" <+> "{" <>
      line <> indent cxxIndent (vsep (map p stmts)) <> line <>
      "}()"
    CxxVarRef name -> p name
    CxxIfExpr cond trueExpr falseExpr -> p cond <+> "?" <+> p trueExpr <+>
      ":" <+> p falseExpr

instance Pretty CxxLambdaCaptureDefault where
  pretty CxxLambdaCaptureDefaultValue = "[=]"
  pretty CxxLambdaCaptureDefaultReference = "[&]"

instance Pretty CxxType where
  pretty cxxTy = case cxxTy of
    CxxVoid -> "void"
    CxxBool -> "bool"
    CxxCustomType ty -> p ty

instance Pretty CxxAccessSpec where
  pretty cxxAS = case cxxAS of
    CxxPublic -> "public"
    CxxPrivate -> "private"
    CxxProtected -> "protected"