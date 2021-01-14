module Parser (parsePackage, parsePackageDependencies) where

import Control.Monad.Combinators.Expr
import Control.Monad.Except
import Data.Functor (($>))
import Data.Maybe (isNothing)
import Data.Void
import Debug.Trace (trace)
import Text.Megaparsec
import Text.Megaparsec.Char

import qualified Data.List.NonEmpty as NE
import qualified Data.Text.Lazy as T
import qualified Text.Megaparsec.Char.Lexer as L

import Env
import PPrint
import Syntax

type Err = String

type Parser = Parsec Void String

-- TODO: find type for imports
parsePackageDependencies = undefined

parsePackage :: String -> String -> Except Err [UModule]
parsePackage filename s =
  case parse (sc >> package) filename s of
    Left e -> throwError $ errorBundlePretty e
    Right (UPackage _ modules _) -> return modules

package :: Parser UPackage
package = do
  pkgName <- keyword PackageKW *> packageName
  deps <- choice [ keyword ImportKW >> (packageName `sepBy1` symbol ",")
                 , return []
                 ] <* symbol ";"
  modules <- manyTill (withSrc (many (symbol ";") *> module' <* many (symbol ";"))) eof
  return $ UPackage pkgName modules deps

module' :: Parser UModule'
module' = do
  cons <- (keyword ConceptKW >> return UCon)
           <|> (keyword ImplementationKW >> return UImpl)
           <|> (keyword SignatureKW >> return USig)
           <|> (keyword ProgramKW >> return UProg)
  name <- ModName <$> nameString
  symbol "="
  declsAndDeps <- braces $ many (try (Left <$> declaration)
                                 <|> (Right <$> moduleDependency))
  let decls = [decl | (Left decl) <- declsAndDeps]
      deps  = [dep  | (Right dep) <- declsAndDeps]
  return $ cons name decls deps

declaration :: Parser UDecl
declaration = withSrc declaration' <* many (symbol ";")
  where declaration' =  typeDecl
                    <|> callable

typeDecl :: Parser UDecl'
typeDecl = do
  keyword TypeKW
  name <- typeName <* symbol ";" -- TODO: make expr
  return $ UType name

callable :: Parser UDecl'
callable = do
  callableType <- (keyword AxiomKW >> return Axiom)
              <|> (keyword FunctionKW >> return Function)
              <|> (keyword PredicateKW >> return Predicate)
              <|> (keyword ProcedureKW >> return Procedure)
  let nameCons = if callableType == Procedure then ProcName else FuncName
  name <- try symOpName <|> (nameCons <$> nameString)
  args <- case callableType of
      Procedure -> parens ((varMode >>= annVar) `sepBy` symbol ",")
      _ -> parens (annVar UObs `sepBy` symbol ",")
  retType <- case callableType of
      Function  -> symbol ":" *> withSrc typeName
      Predicate -> return $ NoCtx Pred
      _         -> return $ NoCtx Unit
  body <- optional (blockExpr
                <|> (symbol "=" *> (blockExpr <|> (expr <* symbol ";"))))
  when (isNothing body) $ symbol ";"
  return $ UCallable callableType name args retType body

annVar :: UVarMode -> Parser UVar
annVar mode = withSrc annVar'
  where
    annVar' = do
      name <- varName
      ann <- symbol ":" *> typeName
      return $ Var mode name (Just ann)

nameVar :: UVarMode -> Parser UVar
nameVar mode = withSrc ((\name -> Var mode name Nothing) <$> varName)

var :: UVarMode -> Parser UVar
var mode = try (annVar mode) <|> nameVar mode

varMode :: Parser UVarMode
varMode = try (keyword ObsKW >> return UObs)
      <|> (keyword OutKW >> return UOut)
      <|> (keyword UpdKW >> return UUpd)

expr :: Parser UExpr
expr = makeExprParser leafExpr ops
  where
    leafExpr = blockExpr
           <|> try ifExpr
           <|> try (callableCall FuncName)
           <|> try (keyword CallKW *> callableCall ProcName)
           <|> withSrc (var UUnk >>= \v -> return $ UVar v)
           <|> parens expr

blockExpr :: Parser UExpr
blockExpr = withSrc blockExpr'
  where
    blockExpr' = do
      block <- braces (many (exprStmt <* some (symbol ";")))
      block' <- if null block then (:[]) <$> withSrc (return USkip)
                else return block
      return $ UBlockExpr (NE.fromList block')

ifExpr :: Parser UExpr
ifExpr = withSrc ifExpr'
  where
    ifExpr' = do
      keyword IfKW
      cond <- expr
      keyword ThenKW
      bTrue <- expr
      keyword ElseKW
      UIf cond bTrue <$> expr

callableCall :: (String -> Name) -> Parser UExpr
callableCall nameCons = withSrc callableCall'
  where
    callableCall' = do
      name <- try symOpName <|> (nameCons <$> nameString)
      args <- parens (expr `sepBy` symbol ",")
      ann <- optional (symbol ":" *> typeName)
      return $ UCall name args ann

moduleDependency :: Parser UModuleDep
moduleDependency = withSrc moduleDependency'
  where
    moduleDependency' = do
      keyword UseKW
      name <- ModName <$> nameString
      renamings <- many (withSrc $ brackets (renaming `sepBy` symbol ","))
      symbol ";"
      return $ UModuleDep name renamings

renaming :: Parser Renaming
renaming = do
  source <- try symOpName <|> (UnspecName <$> nameString)
  symbol "=>"
  target <- try symOpName <|> (UnspecName <$> nameString)
  return (source, target)

exprStmt :: Parser UExpr
exprStmt = assertStmt <|> try letStmt <|> expr

assertStmt :: Parser UExpr
assertStmt = keyword AssertKW *> withSrc (UAssert <$> expr)

letStmt :: Parser UExpr
letStmt = withSrc letStmt'
  where
    letStmt' = do
      keyword LetKW
      isConst <- option False (keyword ObsKW $> True)
      name <- varName
      ann <- optional (symbol ":" *> typeName)
      value <- optional expr <* symbol ";"
      let mode | isConst = UObs
               | isNothing value = UOut
               | otherwise = UUpd
      return $ ULet mode name ann value

ops :: [[Operator Parser UExpr]]
ops = [ map unOp ["+", "-", "!", "~"]                -- unary ops
      , map binOp ["*", "/", "%"]                    -- mult ops
      , map binOp ["+", "-"]                         -- add ops
      , map binOp ["<<", ">>"]                       -- shift ops
      , [binOp ".."]                                 -- range
      , map binOp ["<", ">", ">=", "<=", "==", "!="] -- comparison ops
      , [binOp "&&"]                                 -- logical and
      , [binOp "||"]                                 -- logical or
      , [binOp "=>", binOp "<=>"]                    -- logical implication
      ]

unOp :: String -> Operator Parser UExpr
unOp s = Prefix $ unOpCall <$> withSrc (symbol s)
  where
    unOpCall wrapper e = UCall (FuncName (s <> "_")) [e] Nothing <$ wrapper

binOp :: String -> Operator Parser UExpr
binOp s = InfixL $ binOpCall <$> withSrc (symbol s)
  where
    binOpCall wrapper e1 e2 =
      UCall (FuncName ("_" <> s <> "_")) [e1, e2] Nothing <$ wrapper

-- === utils ===

braces :: Parser a -> Parser a
braces = between (symbol "{") (symbol "}")

brackets :: Parser a -> Parser a
brackets = between (symbol "[") (symbol "]")

parens :: Parser a -> Parser a
parens = between (symbol "(") (symbol ")")

symOpName :: Parser Name
symOpName = choice $ map (try . mkSymOpNameParser) symOps
  where symOps = [ "+_", "-_", "!_", "~_"
                 , "_*_", "_/_", "_%_"
                 , "_+_", "_-_"
                 , "_<<_", "_>>_"
                 , "_.._"
                 , "_<_", "_>_", "_>=_", "_<=_", "_==_", "_!=_"
                 , "_&&_"
                 , "_||_"
                 , "_=>_", "_<=>_"
                 ]
        mkSymOpNameParser s = do
          symbol s <* notFollowedBy nameChar
          return $ FuncName s

data Keyword = ConceptKW | ImplementationKW | ProgramKW | SignatureKW
             | AxiomKW | FunctionKW | PredicateKW | ProcedureKW | TypeKW
             | UseKW
             | ObsKW | OutKW | UpdKW
             | AssertKW | CallKW | IfKW | ThenKW | ElseKW | LetKW
             | PackageKW | ImportKW

keyword :: Keyword -> Parser ()
keyword kw = (lexeme . try) $ string s *> notFollowedBy nameChar
  where
    s = case kw of
      ConceptKW        -> "concept"
      ImplementationKW -> "implementation"
      ProgramKW        -> "program"
      SignatureKW      -> "signature"
      AxiomKW          -> "axiom"
      FunctionKW       -> "function"
      PredicateKW      -> "predicate"
      ProcedureKW      -> "procedure"
      TypeKW           -> "type"
      UseKW            -> "use"
      ObsKW            -> "obs"
      OutKW            -> "out"
      UpdKW            -> "upd"
      AssertKW         -> "assert"
      CallKW           -> "call"
      IfKW             -> "if"
      ThenKW           -> "then"
      ElseKW           -> "else"
      LetKW            -> "var"
      PackageKW        -> "package"
      ImportKW         -> "imports"

sc :: Parser ()
sc = L.space space1 skipLineComment skipBlockComment

skipLineComment :: Parser ()
skipLineComment = string "//" >> void (takeWhileP (Just "character") (/= '\n'))

skipBlockComment :: Parser ()
skipBlockComment = string "/*" >> void (manyTill anySingle (symbol "*/"))

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: String -> Parser ()
symbol = void . L.symbol sc

typeName :: Parser Name
typeName = TypeName <$> nameString

varName :: Parser Name
varName = VarName <$> nameString

packageName :: Parser Name
packageName = PkgName <$> packageString

packageString :: Parser String
packageString = lexeme . try $ (:) <$> nameChar <*> many (nameChar <|> char '.')

nameString :: Parser String
nameString = lexeme . try $ (:) <$> nameChar <*> many nameChar

nameChar :: Parser Char
nameChar = alphaNumChar <|> char '_'

withSrc :: Parser a -> Parser (WithSrc a)
withSrc p = do
  start <- mkSrcPos <$> getSourcePos
  element <- p
  end <- mkSrcPos <$> getSourcePos
  return $ WithSrc (Just (start, end)) element
  where
    mkSrcPos s = (sourceName s, unPos $ sourceLine s, unPos $ sourceColumn s)
