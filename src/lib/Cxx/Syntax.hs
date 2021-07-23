{-# LANGUAGE OverloadedStrings #-}

module Cxx.Syntax (
    -- * C++ AST
    CxxAccessSpec (..)
  , CxxBinOp (..)
  , CxxDef (..)
  , CxxExpr (..)
  , CxxFunctionDef (..)
  , CxxInclude
  , CxxLambdaCaptureDefault (..)
  , CxxModule (..)
  , CxxNamespaceName
  , CxxOutMode (..)
  , CxxPackage (..)
  , CxxStmt (..)
  , CxxStmtBlock
  , CxxStmtInline (..)
  , CxxType (..)
  , CxxUnOp (..)
  , CxxVar (..)
    -- * name-related utils
  , CxxName
  , mkClassMemberCxxType
  , mkCxxClassMemberAccess
  , mkCxxName
  , mkCxxNamespaceMemberAccess
  , mkCxxObjectMemberAccess
  , mkCxxRelativeCxxIncludeFromName
  , mkCxxRelativeMgIncludeFromName
  , mkCxxSystemInclude
  , mkCxxType
    -- * pprinting utils
  , pshowCxxPackage
  , pprintCxxPackage
  )
  where

import qualified Data.Map as M
import Data.Maybe (fromMaybe)
import qualified Data.List as L
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO

import Data.Text.Prettyprint.Doc
import Data.Text.Prettyprint.Doc.Render.Text

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util

-- | A name (or identifier) in a C++ context. There are three types of names:
-- * simple names, created using the 'CxxName' constructor;
-- * \"qualified\" names, created using the 'CxxNestedName' constructor, which
-- represent either an access to a member of a class/namespace, or to a member
-- of an object.
data CxxName = -- | A simple identifier
               CxxName String
               -- | A nested identifier
             | CxxNestedName CxxMemberAccessKind CxxName CxxName
               deriving (Eq, Ord, Show)

data CxxMemberAccessKind = CxxClassOrNamespaceMember
                         | CxxObjectMember
                           deriving (Eq, Ord, Show)

-- TODO: handle operators and "invalid C++ names", maybe?
-- | Makes a CxxName out of a Name. The function may throw an error if the name
-- can not be used in C++. The infix functions that are allowed in Magnolia
-- (e.g. '_+_', or '_*_'...) are translated using custom translation rules.
-- When the name contains an hyphen, the hyphen is replaced by an underscore
-- as C++ identifiers can not contain hyphens. In practice, this may happen
-- when converting directory names into C++ namespace names.
-- TODO: this means that if within the same Magnolia projects, two folders
-- x- and x_ contain leaf source files, then these namespaces might end up
-- merged during codegen. This may lead to errors, and hence should be cleaned
-- up. This will be addressed in a future PR, for instance by only allowing
-- paths to external modules to contain hyphens.
mkCxxName :: Name -> MgMonad CxxName
mkCxxName n | _name n `S.member` cxxKeywords = throwNonLocatedE MiscErr $
  "could not make '" <> pshow n <> "' a C++ identifier, because it is a " <>
  "reserved keyword"
mkCxxName n = let cleanName = n { _name = replaceHyphen (_name n) } in return $
  fromMaybe (CxxName $ _name cleanName) (mkCustomCxxName cleanName)
  where
    replaceHyphen l
      | [] <- l = []
      | c:cs <- l = (if c == '-' then '_' else c) : replaceHyphen cs

-- TODO: at the moment, we only give special translation rules for given names.
--       Eventually, we can come up with custom translation rules definable
--       directly in Magnolia for calls to certain functions/procedures.
mkCustomCxxName :: Name -> Maybe CxxName
mkCustomCxxName (Name _ s) = CxxName <$> M.lookup s customCxxTranslations

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

-- | Given two C++ names N1 and N2, builds the C++ name N1::N2.
mkCxxNamespaceMemberAccess :: CxxName -> CxxName -> CxxName
mkCxxNamespaceMemberAccess = CxxNestedName CxxClassOrNamespaceMember

-- | Convenience alias for 'mkCxxNamespaceMemberAccess'.
mkCxxClassMemberAccess :: CxxName -> CxxName -> CxxName
mkCxxClassMemberAccess = mkCxxNamespaceMemberAccess

-- | Given two C++ names N1 and N2, builds the C++ name N1.N2.
mkCxxObjectMemberAccess :: CxxName -> CxxName -> CxxName
mkCxxObjectMemberAccess = CxxNestedName CxxObjectMember

-- | Makes a CxxType out of a Magnolia type. The function may throw an error
-- if the name of the type can not be used in C++.
mkCxxType :: MType -> MgMonad CxxType
mkCxxType typeName = case typeName of
  Unit -> return CxxVoid
  Pred -> return CxxBool
  _ -> CxxCustomType <$> mkCxxName typeName

-- | Similar to 'mkCxxType' except that the C++ type produced is represented
-- as a class member of the enclosing module.
mkClassMemberCxxType :: MType -> MgMonad CxxType
mkClassMemberCxxType typeName = case typeName of
  Unit -> return CxxVoid
  Pred -> return CxxBool
  _ -> do
    moduleCxxName <- getParentModuleName >>= mkCxxName
    cxxTyName <- mkCxxClassMemberAccess moduleCxxName <$> mkCxxName typeName
    return $ CxxCustomType cxxTyName

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
-- of imports, as well as a list of locally defined modules. A C++ package is
-- used to produce both a C++ header and a C++ implementation file for its
-- corresponding Magnolia package.
data CxxPackage =
  CxxPackage { _cxxPackageName :: Name
             , _cxxPackageImports :: [CxxInclude]
             , _cxxPackageModules :: [CxxModule]
             }
  deriving Show

-- | A C++ file inclusion. In the future, we will look into using the C++20
-- import construct and let it wrap a Magnolia FullyQualifiedName instead of
-- this. At the moment, however, C++ modules do not seem to be well supported
-- by C++ compilers. Hence, we fall back to using includes.
data CxxInclude = CxxInclude CxxHeaderLoc FilePath
                  deriving (Eq, Ord, Show)

data CxxHeaderLoc = RelativeMgDir | RelativeCxxDir | SystemDir
                    deriving (Eq, Ord, Show)

-- | Exposed constructor for C++ relative includes of files generated by the
-- Magnolia compiler. Includes created through this constructor are given a
-- ".hpp" suffix.
mkCxxRelativeMgIncludeFromName :: Name -> CxxInclude
mkCxxRelativeMgIncludeFromName =
  CxxInclude RelativeMgDir . (<> ".hpp") .
    map (\c -> if c == '.' then '/' else c) . _name

-- | Exposed constructor for C++ relative includes of external C++ files.
-- Includes created through this constructor are given a ".hpp" suffix.
mkCxxRelativeCxxIncludeFromName :: Name -> CxxInclude
mkCxxRelativeCxxIncludeFromName =
  CxxInclude RelativeCxxDir . (<> ".hpp") .
    map (\c -> if c == '.' then '/' else c) . _name

-- | Exposed constructor for C++ system includes. No suffix is appended to the
-- filepath passed as a parameter through this constructor.
mkCxxSystemInclude :: FilePath -> CxxInclude
mkCxxSystemInclude = CxxInclude SystemDir

-- | The equivalent of a Magnolia module in C++.
data CxxModule =
  CxxModule { -- | The namespaces wrapping the module.
              _cxxModuleNamespaces :: [CxxNamespaceName]
              -- | The name of the module.
            , _cxxModuleName :: CxxModuleName
              -- | The definitions within the module, along with access
              -- specifiers. The set of public definitions should correspond
              -- to the API exposed by the corresponding Magnolia module.
            , _cxxModuleDefinitions :: [(CxxAccessSpec, CxxDef)]
            }
  deriving Show

type CxxModuleName = CxxName
type CxxNamespaceName = CxxName

-- | The equivalent of a Magnolia declaration in C++.
data CxxDef
    -- | A C++ type definition. 'CxxTypeDef src tgt' corresponds to the C++
    -- code 'typedef src tgt'.
  = CxxTypeDef CxxName CxxName
    -- | A C++ function definition.
  | CxxFunctionDef CxxFunctionDef
    -- | An instance of an external module/struct. Modules will carry such
    --   references in order to appropriately call externally defined callables.
    --   TODO: external references may be templated with required types,
    --         requiring thus a specialization. This is not handled yet.
  | CxxExternalInstance CxxType CxxName
    deriving (Eq, Show)

instance Ord CxxDef where
  -- Order is as follows:
  -- type definitions < function definitions < external instances
  compare def1 def2 = case def1 of
    CxxTypeDef src1 tgt1 -> case def2 of
      CxxTypeDef src2 tgt2 ->
        src1 `compare` src2 <> tgt1 `compare` tgt2
      _ -> LT
    CxxFunctionDef cxxFn1 -> case def2 of
      CxxTypeDef {} -> GT
      CxxFunctionDef cxxFn2 -> cxxFn1 `compare` cxxFn2
      CxxExternalInstance {} -> LT
    CxxExternalInstance cxxTy1 cxxName1 -> case def2 of
      CxxExternalInstance cxxTy2 cxxName2 ->
        cxxTy1 `compare` cxxTy2 <> cxxName1 `compare` cxxName2
      _ -> GT


-- TODO: note: we do not care about trying to return const. It has become
-- redundant according to https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out.

-- | The equivalent of a concrete Magnolia function definition in C++.
data CxxFunctionDef =
  CxxFunction { -- | Whether the function should be inlined. This field is set
                -- to true typically when the function is a simple wrapping
                -- over an external implementation.
                _cxxFnIsInline :: Bool
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
                -- | The body of the function.
              , _cxxFnBody :: CxxStmtBlock
              }
  deriving (Eq, Ord, Show)

-- | A variable in C++.
data CxxVar = CxxVar { _cxxVarIsConst :: Bool
                     , _cxxVarIsRef :: Bool
                     , _cxxVarName :: CxxName
                     , _cxxVarType :: CxxType
                     }
              deriving (Eq, Ord, Show)

data CxxStmt = CxxStmtBlock CxxStmtBlock
             | CxxStmtInline CxxStmtInline
               deriving (Eq, Ord, Show)

type CxxStmtBlock = [CxxStmt]

-- | Useful inline statements in the C++ world.
data CxxStmtInline = -- | A variable assignement.
                     CxxAssign CxxName CxxExpr
                     -- | A variable declaration.
                   | CxxVarDecl CxxVar (Maybe CxxExpr)
                     -- | An assertion.
                   | CxxAssert CxxCond
                     -- | An if-then-else statement.
                   | CxxIf CxxCond CxxStmt CxxStmt
                     -- | A statement wrapper around an expression.
                   | CxxExpr CxxExpr
                     -- | A return statement.
                   | CxxReturn (Maybe CxxExpr)
                   | CxxSkip
                     deriving (Eq, Ord, Show)

type CxxCond = CxxExpr

-- | Useful expressions in the C++ world. Calls to inline closures are used to
-- translate the Magnolia stateful and value blocks. The entire environment is
-- captured in these blocks. The only difference is that in the case of value
-- blocks, the environment is captured by value, whereas in stateful blocks it
-- is captured by reference.
data CxxExpr = -- | A call to a function.
               CxxCall CxxName [CxxTemplateParameter] [CxxExpr]
               -- | A call to an inline lambda definition.
             | CxxLambdaCall CxxLambdaCaptureDefault CxxStmtBlock
               -- | A variable access.
             | CxxVarRef CxxName
               -- | An if-then-else expression.
             | CxxIfExpr CxxCond CxxExpr CxxExpr
               -- | A unary operator wrapper for convenience.
             | CxxUnOp CxxUnOp CxxExpr
               -- | A binary operator wrapper for convenience.
             | CxxBinOp CxxBinOp CxxExpr CxxExpr
               deriving (Eq, Ord, Show)

data CxxUnOp = CxxLogicalNot
               deriving (Eq, Ord, Show)

data CxxBinOp = CxxLogicalAnd | CxxLogicalOr | CxxEqual
                deriving (Eq, Ord, Show)

data CxxLambdaCaptureDefault = CxxLambdaCaptureDefaultValue
                             | CxxLambdaCaptureDefaultReference
                             deriving (Eq, Ord, Show)

data CxxType = CxxVoid | CxxBool | CxxCustomType CxxName
               deriving (Eq, Ord, Show)

type CxxTemplateParameter = CxxName

data CxxAccessSpec = CxxPublic | CxxPrivate | CxxProtected
                     deriving (Eq, Ord, Show)

-- === pretty utils ===

-- TODO: PrettyCxx might be a bit wrong if we need to carry around a path.

-- | An enumeration of possible output modes for the PrettyCxx typeclass.
data CxxOutMode = CxxHeader | CxxImplementation

-- | Pretty prints a C++ package as an implementation file or as a header
-- file. A base path can be provided to derive includes correctly.
pprintCxxPackage :: Maybe FilePath -> CxxOutMode -> CxxPackage -> IO ()
pprintCxxPackage mbasePath cxxOutMode =
  TIO.putStrLn . renderLazy . layoutPretty defaultLayoutOptions .
    prettyCxxPackage mbasePath cxxOutMode

-- | Pretty formats a C++ package as an implementation file or as a header
-- file. A base path can be provided to derive includes correctly.
pshowCxxPackage :: Maybe FilePath -> CxxOutMode -> CxxPackage -> T.Text
pshowCxxPackage mbasePath cxxOutMode =
  render . prettyCxxPackage mbasePath cxxOutMode

-- | Pretty formats a C++ package as an implementation file or as a header
-- file. A base path can be provided to derive includes correctly.
prettyCxxPackage :: Maybe FilePath -> CxxOutMode -> CxxPackage -> Doc ann
prettyCxxPackage mbasePath cxxOutMode (CxxPackage pkgName includes modules) =
    (case cxxOutMode of
      CxxHeader -> vsep (map (prettyCxxInclude mbasePath) $ L.sort includes)
      CxxImplementation -> prettyCxxInclude mbasePath
        (mkCxxRelativeMgIncludeFromName pkgName)) <> line <> line <>
    vsep (map ((line <>) . prettyCxxModule cxxOutMode) $
      L.sortOn _cxxModuleName modules)

prettyCxxInclude :: Maybe FilePath -> CxxInclude -> Doc ann
prettyCxxInclude mbasePath (CxxInclude headerLoc filePath) =
  let realFilePath = case mbasePath of
        Nothing -> filePath
        Just basePath -> basePath <> "/" <> filePath
  in "#include" <+> case headerLoc of
    RelativeCxxDir -> dquotes (p filePath)
    RelativeMgDir -> dquotes (p realFilePath)
    SystemDir -> langle <> p filePath <> rangle

prettyCxxModule :: CxxOutMode -> CxxModule -> Doc ann
prettyCxxModule CxxHeader (CxxModule namespaces name defs) =
  -- 1. Open namespaces
  align (vsep (map (\ns -> "namespace" <+> p ns <+> "{") namespaces)) <>
  line <>
  -- 2. Build class / struct header
  "struct" <+> p name <+> "{" <> line <>
  -- 3. Input class / struct definitions
  structContent <> line <>
  "};" <> line <>
  -- 4. Close namespaces
  align (vsep (map (\ns -> "} //" <+> p ns) namespaces))
  where
    sortedDefs = L.sortOn snd defs

    mkSection cxxAS = vsep $
      map (prettyCxxDef CxxHeader . snd)
          (filter (\(as, _) -> cxxAS == as) sortedDefs)

    structContent =
        "private:" <> line <>
          indent cxxIndent (mkSection CxxPrivate) <> line <>
        "protected:" <> line <>
          indent cxxIndent (mkSection CxxProtected) <> line <>
        "public:" <> line <>
          indent cxxIndent (mkSection CxxPublic)
prettyCxxModule CxxImplementation (CxxModule namespaces cxxModuleName defs) =
  -- 1. Open namespaces
  align (vsep (map (\ns -> "namespace" <+> p ns <+> "{") namespaces)) <>
  line <>
  -- 2. Build class / struct definitions
  indent cxxIndent
    (vsep (map ((<> ";") . prettyCxxFnDef CxxImplementation) defsToImpl)) <>
    line <>
  -- 3. Close namespaces
  align (vsep (map (\ns -> "} //" <+> p ns) namespaces))
  where
    defsToImpl = L.sort $ map mkImplName $
      filter (not . _cxxFnIsInline) $
        foldl (\acc -> getFnDefs acc . snd) [] defs

    mkImplName cxxFn =
      let newName = mkCxxClassMemberAccess cxxModuleName (_cxxFnName cxxFn)
      in cxxFn { _cxxFnName = newName}

    getFnDefs acc cxxDef = case cxxDef of
      CxxFunctionDef cxxFn -> cxxFn:acc
      _ -> acc

prettyCxxDef :: CxxOutMode -> CxxDef -> Doc ann
prettyCxxDef _ (CxxTypeDef source target) =
    "typedef" <+> p source <+> p target <> ";"
prettyCxxDef cxxOutMode (CxxFunctionDef fn) =
  prettyCxxFnDef cxxOutMode fn <> ";"
prettyCxxDef _ (CxxExternalInstance ty name) = p ty <+> p name <> ";"

prettyCxxFnDef :: CxxOutMode -> CxxFunctionDef -> Doc ann
prettyCxxFnDef cxxOutMode
              (CxxFunction isInline name templateParams params retTy body) =
  (if isTemplated then pTemplateParams <> line else "") <>
  (if isInline then "inline " else "") <> p retTy <+> p name <> "(" <>
  hsep (punctuate comma (map p params)) <> ")" <>
  case cxxOutMode of
    CxxHeader -> if isTemplated || isInline
                  then " " <> p (CxxStmtBlock body)
                  else ""
    CxxImplementation -> " " <> p (CxxStmtBlock body)
  where
    pTemplateParams = "template <" <>
      hsep (punctuate comma (map (("typename" <+>) . p) templateParams)) <>
      ">"

    isTemplated = not (null templateParams)

p :: Pretty a => a -> Doc ann
p = pretty

cxxIndent :: Int
cxxIndent = 4

-- === pretty instances ===

instance Pretty CxxName where
  pretty (CxxName name) = p name
  pretty (CxxNestedName accessKind parentName childName) =
    p parentName <> p accessKind <> p childName

instance Pretty CxxMemberAccessKind where
  pretty CxxClassOrNamespaceMember = "::"
  pretty CxxObjectMember = "."

instance Pretty CxxVar where
  pretty (CxxVar isConst isRef name ty) =
    (if isConst then "const " else "") <> p ty <>
    (if isRef then "&" else "") <+> p name

instance Pretty CxxStmt where
  pretty (CxxStmtBlock stmts) = "{" <> line <>
    indent cxxIndent (vsep (map p stmts)) <> line <> "}"
  pretty (CxxStmtInline stmt) = case stmt of
    CxxIf {} -> p stmt
    _ -> p stmt <> semi

instance Pretty CxxStmtInline where
  pretty inlineStmt = case inlineStmt of
    CxxAssign name expr -> p name <+> "=" <+> p expr
    CxxVarDecl var mExpr -> p var <> maybe "" ((" =" <+>) . p) mExpr
    CxxAssert cond -> "assert(" <> p cond <> ")"
    CxxIf cond trueBranch falseBranch ->
      let pbranch branchStmt = case branchStmt of
            CxxStmtBlock {} -> p branchStmt
            CxxStmtInline {} -> indent cxxIndent (p branchStmt)
      in "if (" <> p cond <> ")" <> line <>
         pbranch trueBranch <> line <>
         "else" <> line
         <> pbranch falseBranch
    CxxExpr expr -> p expr
    CxxReturn mExpr -> "return" <> maybe "" ((" " <>) . p) mExpr
    CxxSkip -> ""

instance Pretty CxxExpr where
  pretty inExpr = case inExpr of
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
    CxxUnOp unOp expr -> p unOp <> p expr
    CxxBinOp binOp lhsExpr rhsExpr ->
      -- TODO: add parens only when necessary
      parens (p lhsExpr) <+> p binOp <+> parens (p rhsExpr)

instance Pretty CxxUnOp where
  pretty CxxLogicalNot = "!"

instance Pretty CxxBinOp where
  pretty binOp = case binOp of
    CxxLogicalOr -> "||"
    CxxLogicalAnd -> "&&"
    CxxEqual -> "=="

instance Pretty CxxLambdaCaptureDefault where
  -- When capturing by value, we explicitly capture "this", to avoid the
  -- deprecation warning regarding its implicit capture in C++20.
  pretty CxxLambdaCaptureDefaultValue = "[=, this]"
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