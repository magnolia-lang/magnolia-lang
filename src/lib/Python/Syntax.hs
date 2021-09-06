{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module Python.Syntax (
    -- * Python AST
    PyBinOp (..)
  , PyDef (..)
  , PyExpr (..)
  , PyFnCall (..)
  , PyFunctionDef (..)
  , PyImport (..)
  , PyImportType (..)
  , PyModule (..)
  , PyPackage (..)
  , PyStmt (..)
  , PyStmtBlock (..)
  , PyUnOp (..)
  , PyVar (..)
    -- * name and type -related utils
  , PyName
  , PyType (PyNamedTuple)
  , mkPyName
  , mkPyObjectMemberAccess
  , mkPyType
  , pyConstructorFromType
    -- * misc utils
  , mkNamedTupleClass
  , overloadWith
  , pyModuleBaseStmt
  , returnTypeKwarg
    -- * requirement utils
  , extractParentObjectName
    -- * pprinting utils
  , pprintPyPackage
  , pshowPyPackage
  )
  where

import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Prettyprinter
import Prettyprinter.Render.Text

import Env
import Magnolia.PPrint
import Magnolia.Syntax

import Monad

data PyName = PyName String
            | PyObjectMemberName PyName PyName
            | PyModuleMemberName PyName PyName
              deriving (Eq, Ord, Show)

-- | Builds a valid Python identifier from a Magnolia name.
-- TODO: dummy implementation, fix by filtering keywords
mkPyName :: Name -> MgMonad PyName
mkPyName = pure . PyName . _name

mkPyObjectMemberAccess :: PyName -> PyName -> PyName
mkPyObjectMemberAccess = PyObjectMemberName

data PyPackage =
  PyPackage { _pyPackageName :: Name
            , _pyPackageImports :: [PyImport]
            , _pyPackageModules :: [PyModule]
            }

data PyImport = PyImportFrom PyImportType String [String]
              | PyImport PyImportType String
                deriving (Eq, Ord)

data PyImportType = RelativeMgImport | RelativePyImport | AbsolutePyImport
                    deriving (Eq, Ord, Show)

-- | The equivalent of a Magnolia module in Python.
-- TODO: document the expected structure.
data PyModule = PyModule { _pyModuleName :: PyName
                         , _pyModuleConstructor :: PyFunctionDef
                         }

-- | A statement to prefix the body of the module constructor with.
pyModuleBaseStmt :: PyStmt
pyModuleBaseStmt = PyAssign (PyName "overload")
  (PyCall $ PyFnCall
    (PyObjectMemberName (PyName "functools") (PyName "partial"))
    [ PyVarRef $
        PyObjectMemberName (PyName "multiple_dispatch") (PyName "overload")
    , PyEmptyDict
    ] [])

data PyDef
  = PyClassDef PyName PyExpr
  | PyFunctionDef PyFunctionDef
  | PyDecoratedFunctionDef PyFnCall PyFunctionDef
  | PyInstanceDef PyName PyExpr

-- | The equivalent of a concrete Magnolia function definition in Python.
-- Function definitions and arguments carry their typing information, in order
-- to generate code allowing for multiple dispatch in Python.
data PyFunctionDef =
  PyFunction { -- | The name of the function.
               _pyFnName :: PyName
               -- | The parameters to the function.
             , _pyFnParams :: [PyVar]
               -- | The return type of the function.
             , _pyFnReturnType :: PyType
               -- | The body of the function.
             , _pyFnBody :: PyStmtBlock
             }

data PyType = PyNone | PyBool | PyNamedTuple | PyCustomType PyName

mkPyType :: Name -> MgMonad PyType
mkPyType name = case name of
  Unit -> pure PyNone
  Pred -> pure PyBool
  _ -> PyCustomType <$> mkPyName name

pyConstructorFromType :: PyType -> PyName
pyConstructorFromType pyTy = case pyTy of
  PyNone -> PyName "None"
  PyBool -> PyName "bool"
  PyNamedTuple -> PyName "namedtuple"
  PyCustomType pyName -> pyName

-- | Function calls in Python, with both arguments and keyword arguments.
data PyFnCall = PyFnCall PyName [PyExpr] [(PyName, PyExpr)]

-- | Creates a call to the external overload function with the specified type
-- parameters.
overloadWith
  :: [PyName] -- ^ the name of the types of the arguments to the function to
              -- overload
  -> PyName   -- ^ the name return type of the function to overload
  -> PyFnCall
overloadWith pyArgTypeNames pyReturnTypeName =
  PyFnCall (PyName "overload")
           (map PyVarRef pyArgTypeNames)
           [(PyName "return_type", PyVarRef pyReturnTypeName)]


returnTypeKwarg :: PyExpr -> (PyName, PyExpr)
returnTypeKwarg = (PyName "return_type",)

-- | A variable in Python.
-- TODO; should we store information about mutability here?
data PyVar = PyVar { _pyVarName :: PyName
                   , _pyVarType :: PyType
                   }

newtype PyStmtBlock = PyStmtBlock (NE.NonEmpty PyStmt)

-- TODO: a lot of things are kind of defined the same as in C++. Perhaps it will
--       be worth combining the ASTs at some point.

-- TODO: variable assignment is a normal "x = y" only for non-updated params.
--       When operating on upd or out parameters, we expect to use the
--       'mutate' method associated with their class.
-- TODO: not quite sure what is the actual good solution for out elements,
--       however. The class needs to have a default constructor, or otherwise
--       it is not possible to do anything. This is fine so long as the only
--       uses of the class come from Magnolia, which is not something we can
--       expect. Let's think more about it.
-- TODO: perhaps statement should contain defs, since they can be done at any
--       level?
data PyStmt = -- | A variable assignment.
              PyAssign PyName PyExpr
              -- | A call to the mutate method of an object. We assume that
              -- all classes define a mutate method, so that procedures can
              -- be used as intended in Python. Otherwise, we would need to
              -- functionalize them.
            | PyMutate PyName PyExpr
              -- | A variable declaration.
            | PyVarDecl PyVar PyExpr
              -- | A local definition of a class, closure, or instance of
              -- a class.
            | PyLocalDef PyDef
              -- | An assertion.
            | PyAssert PyCond
              -- | An if-then-else statement.
            | PyIf PyCond PyStmtBlock (Maybe PyStmtBlock)
              -- | A statement wrapper around an expression.
            | PyExpr PyExpr
              -- | A return statement.
            | PyReturn (Maybe PyExpr)
            | PySkip

type PyCond = PyExpr

-- | Expressions in the Python world. Value blocks are seemingly omitted here,
-- because they are synthesized as a closure declaration accompanied with a
-- function call.
data PyExpr = -- | A call to a function.
              PyCall PyFnCall
              -- | A variable access.
            | PyVarRef PyName
              -- | An if-then-else expression.
            | PyIfExpr PyCond PyExpr PyExpr
              -- | A unary operator wrapper..
            | PyUnOp PyUnOp PyExpr
              -- | A binary operator wrapper.
            | PyBinOp PyBinOp PyExpr PyExpr
              -- | The True constant.
            | PyTrue
              -- | The False constant.
            | PyFalse
              -- | A string. This is for defining the fields of our namedtuples,
              -- and should not be produced by any Magnolia expression.
            | PyString String
              -- | A list. This is for defining the fields of our namedtuples,
              -- and should not be produced by any Magnolia expression.
            | PyList [PyExpr]
              -- | An empty dictionary. This is for building a local overload
              -- resolution table within the module, and should not be produced
              -- by any Magnolia expression.
            | PyEmptyDict

data PyUnOp = PyLogicalNot
              deriving (Eq, Ord, Show)

data PyBinOp = PyLogicalAnd | PyLogicalOr | PyEqual
               deriving (Eq, Ord, Show)

-- | Calls the namedtuple function to construct a named tuple with the given
-- name and fields.
mkNamedTupleClass :: String   -- ^ name of the named tuple
                  -> [String] -- ^ ordered fields of the named tuple
                  -> PyExpr
mkNamedTupleClass namedTupleString fieldNames =
  PyCall (PyFnCall (PyName "namedtuple")
                   [PyString namedTupleString, PyList (map PyString fieldNames)]
                   [])

-- === requirement utils ===

extractParentObjectName :: PyName -> Maybe PyName
extractParentObjectName pyName = case pyName of
  PyObjectMemberName parentObjectName _ -> Just parentObjectName
  _ -> Nothing


-- === pretty utils ===

pyIndent :: Int
pyIndent = 4

p :: Pretty a => a -> Doc ann
p = pretty

-- | Pretty prints a Python package. A base path can be provided to derive
-- imports correctly.
pprintPyPackage :: Maybe FilePath -> PyPackage -> IO ()
pprintPyPackage mbasePath =
  TIO.putStrLn . renderStrict . layoutPretty defaultLayoutOptions .
    prettyPyPackage mbasePath

-- | Pretty formats a Python package. A base path can be provided to derive
-- imports correctly.
pshowPyPackage :: Maybe FilePath -> PyPackage -> T.Text
pshowPyPackage mbasePath =
  render . prettyPyPackage mbasePath

-- | Pretty formats a Python package. A base path can be provided to derive
-- imports correctly.
prettyPyPackage :: Maybe FilePath -> PyPackage -> Doc ann
prettyPyPackage mbasePath (PyPackage _ imports modules) =
    vsep (map (prettyPyImport mbasePath) $ L.sort imports) <> line <> line <>
    vsep (map ((line <>) . p) $
      L.sortOn _pyModuleName modules)

prettyPyImport :: Maybe FilePath -> PyImport -> Doc ann
prettyPyImport mbasePath pyImport = case pyImport of
  PyImport pyImportTy filePath ->
    let realFilePath = case mbasePath of
          Nothing -> filePath
          Just basePath -> importify basePath <> "." <> filePath
    in "import" <+> case pyImportTy of
      RelativeMgImport -> p realFilePath
      RelativePyImport -> p filePath
      AbsolutePyImport -> p filePath
  PyImportFrom pyImportTy filePath elements ->
    let realFilePath = case mbasePath of
          Nothing -> filePath
          Just basePath -> importify basePath <> "." <> filePath
    in "from" <+> (case pyImportTy of
      RelativeMgImport -> p realFilePath
      RelativePyImport -> p filePath
      AbsolutePyImport -> p filePath) <+>
      "import" <+> hsep (punctuate comma (map p elements))
  where
    importify :: FilePath -> FilePath
    importify = map (\c -> if c == '/' then '.' else c)


instance Pretty PyName where
  pretty pyName = case pyName of
    PyName name -> p name
    PyModuleMemberName moduleName memberName ->
      p moduleName <> dot <> p memberName
    PyObjectMemberName objectName memberName ->
      p objectName <> dot <> p memberName

instance Pretty PyModule where
  pretty (PyModule _ moduleConstructor) =
    --vsep (map p moduleDependencies) <> line <> line <>
    p moduleConstructor

instance Pretty PyDef where
  pretty pyDef = case pyDef of
    PyClassDef targetName paramExpr -> p targetName <+> "=" <+> p paramExpr
    PyFunctionDef fnDef -> p fnDef
    PyDecoratedFunctionDef pyFnCall pyFnDef ->
      "@" <> p pyFnCall <> line <>
      p pyFnDef
    -- TODO:
    PyInstanceDef pyName pyExpr -> p pyName <+> "=" <+> p pyExpr

instance Pretty PyFunctionDef where
  pretty (PyFunction pyFnName pyFnParams _ pyFnBody) =
    "def" <+> p pyFnName <>
    parens (hsep (punctuate comma (map p pyFnParams))) <> colon <> line <>
    indent pyIndent (p pyFnBody) <> line

instance Pretty PyFnCall where
  pretty (PyFnCall name args kwargs) = p name <> parens
    (hsep (punctuate comma (map p args <> map pkwarg kwargs)))
    where
      pkwarg (kwargName, kwargExpr) = p kwargName <> "=" <> p kwargExpr

instance Pretty PyVar where
  pretty (PyVar varName _) = p varName

instance Pretty PyStmtBlock where
  pretty (PyStmtBlock stmts) = vsep (map p (NE.toList stmts))

instance Pretty PyStmt where
  pretty inStmt = case inStmt of
    PyAssign varName expr -> p varName <+> "=" <+> p expr
    PyMutate varName valueExpr -> p varName <> ".mutate" <> parens (p valueExpr)
    PyVarDecl var rhsExpr -> p var <+> "=" <+> p rhsExpr
    PyLocalDef pyDef -> p pyDef
    PyAssert cond -> "assert" <+> p cond
    PyIf cond trueBlock mfalseBlock ->
      "if" <+> p cond <> colon <> line <>
        indent pyIndent (p trueBlock) <>
      (case mfalseBlock of
        Nothing -> ""
        Just falseBlock ->
          line <> "else" <> colon <> line <>
          indent pyIndent (p falseBlock))
    PyExpr expr -> p expr
    PyReturn expr -> "return" <+> p expr
    PySkip -> "pass"

instance Pretty PyExpr where
  pretty inExpr = case inExpr of
    PyCall pyFnCall -> p pyFnCall
    PyVarRef varName -> p varName
    PyIfExpr cond trueExpr falseExpr -> p trueExpr <+> "if" <+> p cond <+>
      "else" <+> p falseExpr
    PyUnOp unOp expr -> p unOp <+> p expr
    PyBinOp binOp lhsExpr rhsExpr -> parens (p lhsExpr) <+> p binOp <+>
      parens (p rhsExpr)
    PyTrue -> "True"
    PyFalse -> "False"
    PyString str -> dquotes $ p str
    PyList exprs -> lbracket <> hsep (punctuate comma (map p exprs)) <> rbracket
    PyEmptyDict -> lbrace <> rbrace

instance Pretty PyUnOp where
  pretty PyLogicalNot = "not"

instance Pretty PyBinOp where
  pretty binOp = case binOp of
    PyLogicalOr -> "or"
    PyLogicalAnd -> "and"
    PyEqual -> "=="