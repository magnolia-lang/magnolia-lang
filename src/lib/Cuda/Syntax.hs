{-# LANGUAGE OverloadedStrings #-}

module Cuda.Syntax (
    -- * CUDA AST
    CudaAccessSpec (..)
  , CudaBinOp (..)
  , CudaDef (..)
  , CudaExpr (..)
  , CudaFunctionDef (..)
  , CudaFunctionType (..)
  , CudaInclude
  , CudaLambdaCaptureDefault (..)
  , CudaModule (..)
  , CudaNamespaceName
  , CudaObject (..)
  , CudaOutMode (..)
  , CudaPackage (..)
  , CudaStmt (..)
  , CudaStmtBlock
  , CudaStmtInline (..)
  , CudaType (..)
  , CudaUnOp (..)
  , CudaVar (..)
    -- * name-related utils
  , CudaName
  , cudaFunctionCallOperatorName
  , mkClassMemberCudaType
  , mkCudaClassMemberAccess
  , mkCudaName
  , mkCudaNamespaceMemberAccess
  , mkCudaObjectMemberAccess
  , mkCudaRelativeCudaIncludeFromPath
  , mkCudaRelativeMgIncludeFromName
  , mkCudaSystemInclude
  , mkCudaType
    -- * requirements utils
  , extractAllChildrenNamesFromCudaClassTypeInCudaName
  , extractAllChildrenNamesFromCudaClassTypeInCudaType
  , extractTemplateParametersFromCudaName
  , extractTemplateParametersFromCudaType
    -- * pprinting utils
  , pshowCudaPackage
  , pprintCudaPackage
  )
  where

import Control.Monad (join)
import qualified Data.Map as M
import Data.Maybe (fromMaybe)
import qualified Data.List as L
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO

import Prettyprinter
import Prettyprinter.Render.Text

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Monad

-- | A name (or identifier) in a CUDA context. There are two types of names:
-- * simple names, created using the 'CudaName' constructor;
-- * \"qualified\" names, created using the 'CudaClassMemberName',
--   'CudaNamespaceMemberName', or 'CudaObjectMemberName' constructors.
data CudaName = -- | A simple identifier
               CudaName String
               -- | An identifier corresponding to a member of a class
             | CudaClassMemberName CudaType CudaName
               -- | An identifier corresponding to a member of a namespace
             | CudaNamespaceMemberName CudaName CudaName
               -- | An identifier corresponding to a member of an object
             | CudaObjectMemberName CudaName CudaName
               deriving (Eq, Ord, Show)

-- | The CUDA name corresponding to the function call operator \'operator()\'.
cudaFunctionCallOperatorName :: CudaName
cudaFunctionCallOperatorName = CudaName "operator()"

-- TODO: handle operators and "invalid CUDA names", maybe?
-- | Makes a CudaName out of a Name. The function may throw an error if the name
-- can not be used in CUDA. The infix functions that are allowed in Magnolia
-- (e.g. '_+_', or '_*_'...) are translated using custom translation rules.
-- When the name contains an hyphen, the hyphen is replaced by an underscore
-- as CUDA identifiers can not contain hyphens. In practice, this may happen
-- when converting directory names into CUDA namespace names.
-- TODO: this means that if within the same Magnolia projects, two folders
-- x- and x_ contain leaf source files, then these namespaces might end up
-- merged during codegen. This may lead to errors, and hence should be cleaned
-- up. This will be addressed in a future PR, for instance by only allowing
-- paths to external modules to contain hyphens.
mkCudaName :: Name -> MgMonad CudaName
mkCudaName n | _name n `S.member` cudaKeywords = throwNonLocatedE MiscErr $
  "could not make '" <> pshow n <> "' a CUDA identifier, because it is a " <>
  "reserved keyword"
mkCudaName n = let cleanName = n { _name = replaceHyphen (_name n) } in return $
  fromMaybe (CudaName $ _name cleanName) (mkCustomCudaName cleanName)
  where
    replaceHyphen l
      | [] <- l = []
      | c:cs <- l = (if c == '-' then '_' else c) : replaceHyphen cs

-- TODO: at the moment, we only give special translation rules for given names.
--       Eventually, we can come up with custom translation rules definable
--       directly in Magnolia for calls to certain functions/procedures.
-- TODO: hackish mapping, fix up naming conventions maybe.
mkCustomCudaName :: Name -> Maybe CudaName
mkCustomCudaName (Name _ s) = CudaName <$> (case s of
    '_':'_':'_':s' -> ("__" <>) <$> M.lookup ('_':s') customCudaTranslations
    '_':'_':s' -> ("_" <>) <$> M.lookup ('_':s') customCudaTranslations
    _ -> M.lookup s customCudaTranslations)


-- | Defines custom translation rules for special infix functions allowed in
-- Magnolia but whose name would not yield a valid CUDA identifier.
customCudaTranslations :: M.Map String String
customCudaTranslations = M.fromList
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

-- | Given two CUDA names N1 and N2, builds the CUDA name N1::N2.
mkCudaNamespaceMemberAccess :: CudaName -> CudaName -> CudaName
mkCudaNamespaceMemberAccess = CudaNamespaceMemberName

-- | Given a CUDA type T and a CUDA name N, builds the CUDA name T::N.
mkCudaClassMemberAccess :: CudaType -> CudaName -> CudaName
mkCudaClassMemberAccess = CudaClassMemberName

-- | Given two CUDA names N1 and N2, builds the CUDA name N1.N2.
mkCudaObjectMemberAccess :: CudaName -> CudaName -> CudaName
mkCudaObjectMemberAccess = CudaObjectMemberName

-- | Makes a CudaType out of a Magnolia type. The function may throw an error
-- if the name of the type can not be used in CUDA.
mkCudaType :: MType -> MgMonad CudaType
mkCudaType typeName = case typeName of
  Unit -> return CudaVoid
  Pred -> return CudaBool
  _ -> CudaCustomType <$> mkCudaName typeName

-- | Similar to 'mkCudaType' except that the CUDA type produced is represented
-- as a class member of the enclosing module.
mkClassMemberCudaType :: MType -> MgMonad CudaType
mkClassMemberCudaType typeName = case typeName of
  Unit -> return CudaVoid
  Pred -> return CudaBool
  _ -> do
    cudaModuleName <- getParentModuleName >>= mkCudaName
    cudaTyName <- mkCudaClassMemberAccess (CudaCustomType cudaModuleName) <$>
      mkCudaName typeName
    return $ CudaCustomType cudaTyName

-- | The set of reserved keywords in CUDA.
cudaKeywords :: S.Set String
cudaKeywords = S.fromList
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

-- | The equivalent of a Magnolia package in CUDA. A CUDA package contains a list
-- of imports, as well as a list of locally defined modules. A CUDA package is
-- used to produce both a CUDA header and a CUDA implementation file for its
-- corresponding Magnolia package.
data CudaPackage =
  CudaPackage { _cudaPackageName :: Name
              , _cudaPackageImports :: [CudaInclude]
              , _cudaPackageModules :: [CudaModule]
              }
  deriving Show

-- | A CUDA file inclusion. As of June 2022, CUDA is bound to C++17.
data CudaInclude = CudaInclude CudaHeaderLoc FilePath
                  deriving (Eq, Ord, Show)

data CudaHeaderLoc = RelativeMgDir | RelativeCudaDir | SystemDir
                    deriving (Eq, Ord, Show)

-- | Exposed constructor for CUDA relative includes of files generated by the
-- Magnolia compiler. Includes created through this constructor are given a
-- ".hpp" suffix.
mkCudaRelativeMgIncludeFromName :: Name -> CudaInclude
mkCudaRelativeMgIncludeFromName =
  CudaInclude RelativeMgDir . (<> ".hpp") .
    map (\c -> if c == '.' then '/' else c) . _name

-- | Exposed constructor for CUDA relative includes of external CUDA files.
-- Includes created through this constructor are given a ".hpp" suffix.
mkCudaRelativeCudaIncludeFromPath :: FilePath -> CudaInclude
mkCudaRelativeCudaIncludeFromPath =
  CudaInclude RelativeCudaDir . (<> ".hpp") .
    map (\c -> if c == '.' then '/' else c)

-- | Exposed constructor for CUDA system includes. No suffix is appended to the
-- filepath passed as a parameter through this constructor.
mkCudaSystemInclude :: FilePath -> CudaInclude
mkCudaSystemInclude = CudaInclude SystemDir

-- | The equivalent of a Magnolia module in CUDA.
data CudaModule =
  CudaModule { -- | The namespaces wrapping the module.
              _cudaModuleNamespaces :: [CudaNamespaceName]
               -- | The name of the module.
             , _cudaModuleName :: CudaModuleName
               -- | The definitions within the module, along with access
               -- specifiers. The set of public definitions should correspond
               -- to the API exposed by the corresponding Magnolia module.
               -- We assume that the definitions here are topologically sorted,
               -- so that printing them in order yields valid CUDA.
             , _cudaModuleDefinitions :: [(CudaAccessSpec, CudaDef)]
             }
  deriving (Eq, Show)

type CudaModuleName = CudaName
type CudaNamespaceName = CudaName

-- | The equivalent of a set of Magnolia declarations in CUDA.
data CudaDef
    -- | A CUDA type definition. 'CudaTypeDef src tgt' corresponds to the CUDA
    -- code 'typedef src tgt'.
  = CudaTypeDef CudaName CudaName
    -- | A CUDA function definition.
  | CudaFunctionDef CudaFunctionDef
    -- | A nested module definition.
  | CudaNestedModule CudaModule
    -- | An instance of a module/struct. Modules will carry such references in
    -- order to appropriately call callables defined in other structs if
    -- necessary.
  | CudaInstance CudaObject
    deriving (Eq, Show)

data CudaObject = CudaObject
  { _cudaObjectType :: CudaType
  , _cudaObjectName :: CudaName
  }
  deriving (Eq, Ord, Show)

-- TODO: is this needed?
-- | Used to differentiate device, host, and global member functions.
data CudaFunctionType = CudaFunctionType'Device
                      | CudaFunctionType'DeviceHost
                      | CudaFunctionType'Global
                      | CudaFunctionType'Host
                        deriving (Eq, Ord, Show)

-- TODO: note: we do not care about trying to return const. It has become
-- redundant according to https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out.

-- | The equivalent of a concrete Magnolia function definition in CUDA.
data CudaFunctionDef =
  CudaFunction { -- | Where the function can be executed. As of June 2020, the
                 -- expected values are 'CudaFunctionType'DeviceHost' and
                 -- 'CudaFunctionType'Global'
                _cudaFnType :: CudaFunctionType
               , -- | Whether the function should be inlined. This field is set
                 -- to true typically when the function is a simple wrapping
                  -- over an external implementation.
                 _cudaFnIsInline :: Bool
                 -- | The name of the function.
               , _cudaFnName :: CudaName
                 -- | The template parameters to the function. This is at least
                 -- necessary when generating CUDA code for Magnolia functions
                 -- that are overloaded only on their return type.
               , _cudaFnTemplateParameters :: [CudaTemplateParameter]
                 -- | The parameters to the function.
               , _cudaFnParams :: [CudaVar]
                 -- | The return type of the function.
               , _cudaFnReturnType :: CudaType
                 -- | The body of the function.
               , _cudaFnBody :: CudaStmtBlock
               }
  deriving (Eq, Ord, Show)

-- | A variable in CUDA.
data CudaVar = CudaVar { _cudaVarIsConst :: Bool
                       , _cudaVarIsRef :: Bool
                       , _cudaVarName :: CudaName
                       , _cudaVarType :: CudaType
                       }
               deriving (Eq, Ord, Show)

data CudaStmt = CudaStmtBlock CudaStmtBlock
              | CudaStmtInline CudaStmtInline
                deriving (Eq, Ord, Show)

type CudaStmtBlock = [CudaStmt]

-- | Useful inline statements in the CUDA world.
data CudaStmtInline = -- | A variable assignment.
                      CudaAssign CudaName CudaExpr
                      -- | A variable declaration.
                    | CudaVarDecl CudaVar (Maybe CudaExpr)
                      -- | An assertion.
                    | CudaAssert CudaCond
                      -- | An if-then-else statement.
                    | CudaIf CudaCond CudaStmt CudaStmt
                      -- | A statement wrapper around an expression.
                    | CudaExpr CudaExpr
                      -- | A return statement.
                    | CudaReturn (Maybe CudaExpr)
                    | CudaSkip
                      deriving (Eq, Ord, Show)

type CudaCond = CudaExpr

-- | Useful expressions in the CUDA world. Calls to inline closures are used to
-- translate the Magnolia stateful and value blocks. The entire environment is
-- captured in these blocks. The only difference is that in the case of value
-- blocks, the environment is captured by value, whereas in stateful blocks it
-- is captured by reference.
data CudaExpr = -- | A call to a function.
                CudaCall CudaName [CudaTemplateParameter] [CudaExpr]
                -- | A call to a global method.
              | CudaGlobalCall CudaDim3 CudaName [CudaTemplateParameter]
                               [CudaExpr]
                -- | A call to an inline lambda definition.
              | CudaLambdaCall CudaLambdaCaptureDefault CudaStmtBlock
                -- | A variable access.
              | CudaVarRef CudaName
                -- | An if-then-else expression.
              | CudaIfExpr CudaCond CudaExpr CudaExpr
                -- | A unary operator wrapper for convenience.
              | CudaUnOp CudaUnOp CudaExpr
                -- | A binary operator wrapper for convenience.
              | CudaBinOp CudaBinOp CudaExpr CudaExpr
              | CudaTrue
              | CudaFalse
                deriving (Eq, Ord, Show)

data CudaUnOp = CudaLogicalNot
               deriving (Eq, Ord, Show)

data CudaBinOp = CudaLogicalAnd | CudaLogicalOr | CudaEqual | CudaNotEqual
                deriving (Eq, Ord, Show)

-- TODO: I think we actually never have to capture by value, since Magnolia
-- takes care of checking that variables are not updated in effectful blocks.
data CudaLambdaCaptureDefault = CudaLambdaCaptureDefaultValue
                             | CudaLambdaCaptureDefaultReference
                             deriving (Eq, Ord, Show)

data CudaType = CudaVoid
             | CudaBool
             | CudaCustomType CudaName
             | CudaCustomTemplatedType CudaName [CudaTemplateParameter]
               deriving (Eq, Ord, Show)

type CudaTemplateParameter = CudaType

data CudaAccessSpec = CudaPublic | CudaPrivate | CudaProtected
                     deriving (Eq, Ord, Show)

-- === requirements utils ===

extractTemplateParametersFromCudaType :: CudaType -> [CudaName]
extractTemplateParametersFromCudaType cudaType = case cudaType of
  CudaVoid -> []
  CudaBool -> []
  CudaCustomType cudaName -> extractTemplateParametersFromCudaName cudaName
  CudaCustomTemplatedType _ cudaTemplateParams -> join $
    map extractTemplateParametersFromTemplateParameter cudaTemplateParams
  where
    extractTemplateParametersFromTemplateParameter :: CudaType -> [CudaName]
    extractTemplateParametersFromTemplateParameter cudaTp = case cudaTp of
      CudaVoid -> []
      CudaBool -> []
      CudaCustomType cudaName ->
        cudaName : extractTemplateParametersFromCudaName cudaName
      CudaCustomTemplatedType cudaName cudaTemplateParams -> cudaName :
        extractTemplateParametersFromCudaName cudaName <>
        join (map extractTemplateParametersFromTemplateParameter
                  cudaTemplateParams)

extractTemplateParametersFromCudaName :: CudaName -> [CudaName]
extractTemplateParametersFromCudaName cudaName = case cudaName of
  CudaName {} -> []
  CudaClassMemberName cudaTy cudaMemberName ->
    extractTemplateParametersFromCudaType cudaTy <>
    extractTemplateParametersFromCudaName cudaMemberName
  CudaNamespaceMemberName _ cudaMemberName ->
    extractTemplateParametersFromCudaName cudaMemberName
  CudaObjectMemberName _ cudaMemberName ->
    extractTemplateParametersFromCudaName cudaMemberName

-- | Extract all the children of a specific class name that can be found in a
-- name.
extractAllChildrenNamesFromCudaClassTypeInCudaName
  :: CudaType -- ^ the class type
  -> CudaName -- ^ the name from which to extract the children names
  -> [CudaName]
extractAllChildrenNamesFromCudaClassTypeInCudaName classTy cudaName =
  case cudaName of
    CudaName {} -> []
    CudaClassMemberName cudaTy cudaMemberName ->
      if cudaTy == classTy then [cudaMemberName]
      else extractAllChildrenNamesFromCudaClassTypeInCudaType classTy cudaTy
    CudaNamespaceMemberName _ cudaMemberName ->
      extractAllChildrenNamesFromCudaClassTypeInCudaName classTy cudaMemberName
    CudaObjectMemberName objectName memberName ->
      extractAllChildrenNamesFromCudaClassTypeInCudaName classTy objectName <>
      extractAllChildrenNamesFromCudaClassTypeInCudaName classTy memberName

-- | See 'extractAllChildrenNamesFromCudaClassTypeInCudaName'.
extractAllChildrenNamesFromCudaClassTypeInCudaType
  :: CudaType -- ^ the class type
  -> CudaType -- ^ the type from which to extract the children names
  -> [CudaName]
extractAllChildrenNamesFromCudaClassTypeInCudaType classTy cudaTy =
  case cudaTy of
    CudaVoid -> []
    CudaBool -> []
    CudaCustomType cudaName ->
      extractAllChildrenNamesFromCudaClassTypeInCudaName classTy cudaName
    CudaCustomTemplatedType cudaName cudaTemplateTys ->
      extractAllChildrenNamesFromCudaClassTypeInCudaName classTy cudaName <>
      join (map (extractAllChildrenNamesFromCudaClassTypeInCudaType classTy)
                cudaTemplateTys)


-- === pretty utils ===

-- TODO: PrettyCuda might be a bit wrong if we need to carry around a path.

-- | An enumeration of possible output modes for the PrettyCuda typeclass.
data CudaOutMode = CudaHeader | CudaImplementation
                  deriving Eq

-- | Pretty prints a CUDA package as an implementation file or as a header
-- file. A base path can be provided to derive includes correctly.
pprintCudaPackage :: Maybe FilePath -> CudaOutMode -> CudaPackage -> IO ()
pprintCudaPackage mbasePath cudaOutMode =
  TIO.putStrLn . renderLazy . layoutPretty defaultLayoutOptions .
    prettyCudaPackage mbasePath cudaOutMode

-- | Pretty formats a CUDA package as an implementation file or as a header
-- file. A base path can be provided to derive includes correctly.
pshowCudaPackage :: Maybe FilePath -> CudaOutMode -> CudaPackage -> T.Text
pshowCudaPackage mbasePath cudaOutMode =
  render . prettyCudaPackage mbasePath cudaOutMode

-- | Pretty formats a CUDA package as an implementation file or as a header
-- file. A base path can be provided to derive includes correctly.
prettyCudaPackage :: Maybe FilePath -> CudaOutMode -> CudaPackage -> Doc ann
prettyCudaPackage mbasePath cudaOutMode (CudaPackage pkgName includes modules) =
    (case cudaOutMode of
      CudaHeader -> "#pragma once" <> line <> line <>
        vsep (map (prettyCudaInclude mbasePath) $ L.sort includes)
      CudaImplementation -> prettyCudaInclude mbasePath
        (mkCudaRelativeMgIncludeFromName pkgName)) <> line <> line <>
    vsep (map ((line <>) . prettyCudaModule cudaOutMode) $
      L.sortOn _cudaModuleName modules)

prettyCudaInclude :: Maybe FilePath -> CudaInclude -> Doc ann
prettyCudaInclude mbasePath (CudaInclude headerLoc filePath) =
  let realFilePath = case mbasePath of
        Nothing -> filePath
        Just basePath -> basePath <> "/" <> filePath
  in "#include" <+> case headerLoc of
    RelativeCudaDir -> dquotes (p filePath)
    RelativeMgDir -> dquotes (p realFilePath)
    SystemDir -> langle <> p filePath <> rangle

-- TODO: pretty print template parameters
prettyCudaModule :: CudaOutMode -> CudaModule -> Doc ann
prettyCudaModule CudaHeader (CudaModule namespaces name defs) =
  -- 1. Open namespaces
  ifNotNull namespaces
    (align (vsep (map (\ns -> "namespace" <+> p ns <+> "{") namespaces)) <>
     line) <>
  -- 2. Build class / struct header
  "struct" <+> p name <+> "{" <> line <>
  -- 3. Input class / struct definitions
  prettyStructContent CudaPublic defs <>
  "};" <> line <>
  -- 5. Close namespaces
  align (vsep (map (\ns -> "} //" <+> p ns) namespaces))
  where
    ifNotNull es doc = if not (null es) then doc else ""

    prettyDef prevAccessSpec accessSpec def =
      if accessSpec == prevAccessSpec
      then indent cudaIndent (prettyCudaDef CudaHeader def)
      else p accessSpec <> colon <> line <>
        indent cudaIndent (prettyCudaDef CudaHeader def)

    prettyStructContent _ [] = ""
    prettyStructContent prevAccessSpec ((accessSpec, def):ds) =
      prettyDef prevAccessSpec accessSpec def <> line <>
      prettyStructContent accessSpec ds

prettyCudaModule CudaImplementation
                (CudaModule namespaces cudaModuleName
                           defsWithAccessSpec) =
  -- 1. Open namespaces
  ifNotNull namespaces
  (align (vsep (map (\ns -> "namespace" <+> p ns <+> "{") namespaces)) <>
   line) <>
  -- 2. Build class / struct definitions
  implDoc <>
  -- 3. Close namespaces if some were opened
  ifNotNull namespaces
    (line <> align (vsep (map (\ns -> "} //" <+> p ns) namespaces)))
  where
    ifNotNull es doc = if not (null es) then doc else ""

    (_, defs) = unzip defsWithAccessSpec

    implDoc = vsep $ foldl accDifferentDefsInOrder [] defs

    accDifferentDefsInOrder docElems cudaDef = case cudaDef of
      CudaFunctionDef cudaFn ->
        if not (_cudaFnIsInline cudaFn)
        then docElems <> [indent cudaIndent $
                prettyCudaFnDef CudaImplementation
                                (mkFunctionImplName cudaFn) <> ";" <> line]
        else docElems
      CudaNestedModule cudaMod ->
        docElems <> [prettyCudaModule CudaImplementation
                                      (mkNestedModuleImplName cudaMod) <> line]
      CudaInstance cudaObj ->
        docElems <> [indent cudaIndent $
          prettyCudaObject CudaImplementation (mkObjectImplName cudaObj) <>
            line]
      _ -> docElems

    mkFunctionImplName cudaFn =
      let newName = mkCudaClassMemberAccess (CudaCustomType cudaModuleName)
                                            (_cudaFnName cudaFn)
      in cudaFn { _cudaFnName = newName}

    mkNestedModuleImplName cudaMod =
      let newName = mkCudaClassMemberAccess (CudaCustomType cudaModuleName)
                                            (_cudaModuleName cudaMod)
      in cudaMod { _cudaModuleName = newName }

    mkObjectImplName cudaObj =
      let newName = mkCudaClassMemberAccess (CudaCustomType cudaModuleName)
                                            (_cudaObjectName cudaObj)
      in cudaObj { _cudaObjectName = newName }

prettyCudaDef :: CudaOutMode -> CudaDef -> Doc ann
prettyCudaDef _ (CudaTypeDef sourceName targetName) =
    "typedef" <+> p sourceName <+> p targetName <> ";"
prettyCudaDef cudaOutMode (CudaFunctionDef fn) =
  prettyCudaFnDef cudaOutMode fn <> ";"
prettyCudaDef cudaOutMode (CudaNestedModule cudaMod) =
  prettyCudaModule cudaOutMode cudaMod
prettyCudaDef cudaOutMode (CudaInstance cudaObject) =
  prettyCudaObject cudaOutMode cudaObject

prettyCudaObject :: CudaOutMode -> CudaObject -> Doc ann
prettyCudaObject _ (CudaObject ty name) = p ty <+> p name <> ";"

prettyCudaFnDef :: CudaOutMode -> CudaFunctionDef -> Doc ann
prettyCudaFnDef cudaOutMode
              (CudaFunction cudaFnType isInline name templateParams
                            params retTy body) =
  (if isTemplated then pTemplateParams <> line else "") <>
  prettyCudaFnType cudaFnType <>
  (if isInline then "inline " else "") <> p retTy <+> p name <> "(" <>
  hsep (punctuate comma (map p params)) <> ")" <>
  case cudaOutMode of
    CudaHeader -> if isTemplated || isInline
                  then " " <> p (CudaStmtBlock body)
                  else ""
    CudaImplementation -> " " <> p (CudaStmtBlock body)
  where
    pTemplateParams = "template <" <>
      hsep (punctuate comma (map (("typename" <+>) . p) templateParams)) <>
      ">"

    isTemplated = not (null templateParams)

prettyCudaFnType :: CudaFunctionType -> Doc ann
prettyCudaFnType cudaFnType = case cudaFnType of
  CudaFunctionType'Device -> "__device__"
  CudaFunctionType'DeviceHost -> "__device__ __host__"
  CudaFunctionType'Global -> "__global__"
  CudaFunctionType'Host -> "__host__"

p :: Pretty a => a -> Doc ann
p = pretty

cudaIndent :: Int
cudaIndent = 4

-- === pretty instances ===

instance Pretty CudaName where
  pretty (CudaName name) = p name
  pretty (CudaNamespaceMemberName parentName childName) =
    p parentName <> "::" <> p childName
  pretty (CudaObjectMemberName objectName elementName) =
    p objectName <> "." <> p elementName
  pretty (CudaClassMemberName classTy childName) =
    p classTy <> "::" <> p childName

instance Pretty CudaVar where
  pretty (CudaVar isConst isRef name ty) =
    (if isConst then "const " else "") <> p ty <>
    (if isRef then "&" else "") <+> p name

instance Pretty CudaStmt where
  pretty (CudaStmtBlock stmts) = "{" <> line <>
    indent cudaIndent (vsep (map p stmts)) <> line <> "}"
  pretty (CudaStmtInline stmt) = case stmt of
    CudaIf {} -> p stmt
    _ -> p stmt <> semi

instance Pretty CudaStmtInline where
  pretty inlineStmt = case inlineStmt of
    CudaAssign name expr -> p name <+> "=" <+> p expr
    CudaVarDecl var mExpr -> p var <> maybe "" ((" =" <+>) . p) mExpr
    CudaAssert cond -> "assert(" <> p cond <> ")"
    CudaIf cond trueBranch falseBranch ->
      let pbranch branchStmt = case branchStmt of
            CudaStmtBlock {} -> p branchStmt
            CudaStmtInline {} -> indent cudaIndent (p branchStmt)
      in "if (" <> p cond <> ")" <> line <>
         pbranch trueBranch <> line <>
         "else" <> line
         <> pbranch falseBranch
    CudaExpr expr -> p expr
    CudaReturn mExpr -> "return" <> maybe "" ((" " <>) . p) mExpr
    CudaSkip -> ""

instance Pretty CudaExpr where
  pretty inExpr = case inExpr of
    CudaCall name templateParams args -> p name <> (case templateParams of
        [] -> ""
        _  -> "<" <> hsep (punctuate comma (map p templateParams)) <> ">") <>
      "(" <> hsep (punctuate comma (map p args)) <> ")"
    CudaLambdaCall captureDefault stmts -> p captureDefault <> "()" <+> "{" <>
      line <> indent cudaIndent (vsep (map p stmts)) <> line <>
      "}()"
    CudaVarRef name -> p name
    CudaIfExpr cond trueExpr falseExpr -> p cond <+> "?" <+> p trueExpr <+>
      ":" <+> p falseExpr
    CudaUnOp unOp expr -> p unOp <> p expr
    CudaBinOp binOp lhsExpr rhsExpr ->
      -- TODO: add parens only when necessary
      parens (p lhsExpr) <+> p binOp <+> parens (p rhsExpr)
    CudaTrue -> "true"
    CudaFalse -> "false"

instance Pretty CudaUnOp where
  pretty CudaLogicalNot = "!"

instance Pretty CudaBinOp where
  pretty binOp = case binOp of
    CudaLogicalOr -> "||"
    CudaLogicalAnd -> "&&"
    CudaEqual -> "=="
    CudaNotEqual -> "!="

instance Pretty CudaLambdaCaptureDefault where
  -- When capturing by value, we explicitly capture "this", to avoid the
  -- deprecation warning regarding its implicit capture in C++17.
  pretty CudaLambdaCaptureDefaultValue = "[=, this]"
  pretty CudaLambdaCaptureDefaultReference = "[&]"

instance Pretty CudaType where
  pretty cudaTy = case cudaTy of
    CudaVoid -> "void"
    CudaBool -> "bool"
    CudaCustomType ty -> p ty
    CudaCustomTemplatedType ty templateParameters ->
      p ty <> "<" <> hsep (punctuate comma (map p templateParameters)) <> ">"

instance Pretty CudaAccessSpec where
  pretty cudaAS = case cudaAS of
    CudaPublic -> "public"
    CudaPrivate -> "private"
    CudaProtected -> "protected"