{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}

module Parser (parsePackage, parsePackageHead, parseReplCommand) where

import Control.Monad.Combinators.Expr
import Control.Monad.Except (void, when, ExceptT, throwError)
import Data.Functor (($>))
import Data.Maybe (isNothing)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char

import qualified Data.List.NonEmpty as NE
import qualified Data.Text.Lazy as T
import qualified Text.Megaparsec.Char.Lexer as L

import Env
import Syntax

type Parser = Parsec Void String

type ParsedPackage = UPackage PhParse
type ParsedDecl = UDecl PhParse
type ParsedExpr = UExpr PhParse
type ParsedTypedVar = TypedVar PhParse
type ParsedMaybeTypedVar = MaybeTypedVar PhParse

-- === exception handling utils ===

throwNonLocatedParseError :: ParseErrorBundle String Void -> ExceptT Err IO b
throwNonLocatedParseError =
  throwError . Err ParseErr Nothing . T.pack . errorBundlePretty

-- === repl parsing utils ===

parseReplCommand :: String -> ExceptT Err IO Command
parseReplCommand s = case parse (replCommand <* eof) "repl" s of
  Left e -> throwNonLocatedParseError e
  Right com -> return com

replCommand :: Parser Command
replCommand =  loadPackageCommand
           <|> reloadPackageCommand
           <|> inspectModuleCommand
           <|> inspectPackageCommand
           <|> listPackageCommand
           <|> showMenuCommand

loadPackageCommand :: Parser Command
loadPackageCommand = do
  symbol "load"
  LoadPackage <$> (try sourceFileString <|> packageString)

reloadPackageCommand :: Parser Command
reloadPackageCommand = do
  symbol "reload"
  ReloadPackage <$> (try sourceFileString <|> packageString)

inspectModuleCommand :: Parser Command
inspectModuleCommand = do
  symbol "inspectm"
  InspectModule . ModName <$> nameString

inspectPackageCommand :: Parser Command
inspectPackageCommand = do
  symbol "inspectp"
  InspectPackage <$> packageName

listPackageCommand :: Parser Command
listPackageCommand = symbol "list" >> return ListPackages

showMenuCommand :: Parser Command
showMenuCommand = symbol "help" >> return ShowMenu

-- === package head parsing utils ===

-- TODO: find type for imports
parsePackageHead :: FilePath -> String -> ExceptT Err IO PackageHead
parsePackageHead filePath s =
  case parse (sc >> packageHead filePath s) filePath s of
    Left e -> throwNonLocatedParseError e
    Right ph -> return ph

packageHead :: FilePath -> String -> Parser PackageHead
packageHead filePath s = do
  pkgName <- keyword PackageKW *> packageName
  -- TODO: error out if package name is different from filepath?
  imports <- choice [ keyword ImportKW >>
                      (withSrc packageName `sepBy1` symbol ",")
                    , return []
                    ] <* semi
  return PackageHead { _packageHeadPath = filePath
                     , _packageHeadStr = s
                     , _packageHeadName = pkgName
                     , _packageHeadImports = imports
                     }

-- === package parsing utils ===

parsePackage :: FilePath -> String -> ExceptT Err IO ParsedPackage
parsePackage filePath s =
  case parse (sc >> package) filePath s of
    Left e -> throwNonLocatedParseError e
    Right pkg -> return pkg

package :: Parser ParsedPackage
package = annot package'
  where
    package' = do
      pkgName <- keyword PackageKW *> packageName
      deps <- choice [ keyword ImportKW >>
                       (annot (UPackageDep <$> packageName) `sepBy1` symbol ",")
                     , return []
                     ] <* semi
      decls <- manyTill (many semi *>
                         topLevelDecl <* many semi) eof
      return $ UPackage pkgName decls deps

topLevelDecl :: Parser (UTopLevelDecl PhParse)
topLevelDecl =  try (UModuleDecl <$> moduleDecl)
            <|> (UNamedRenamingDecl <$> renamingDecl)
            <|> (USatisfactionDecl <$> satisfaction)

moduleDecl :: Parser (UModule PhParse)
moduleDecl = do
  typ <- moduleType
  name <- ModName <$> nameString
  symbol "="
  annot $ try (moduleRef typ name) <|> inlineModule typ name

moduleType :: Parser UModuleType
moduleType = (keyword ConceptKW >> return Concept)
          <|> (keyword ImplementationKW >> return Implementation)
          <|> (keyword SignatureKW >> return Signature)
          <|> (keyword ProgramKW >> return Program)
          <|> (keyword ExternalKW >> return External)

moduleRef :: UModuleType -> Name -> Parser (UModule' PhParse)
moduleRef typ name = do
  refName <- ModName <$> nameString
  return $ RefModule typ name refName

inlineModule :: UModuleType -> Name -> Parser (UModule' PhParse)
inlineModule typ name = do
  declsAndDeps <- braces $ many (try (Left <$> declaration typ)
                                 <|> (Right <$> moduleDependency))
  let decls = [decl | (Left decl) <- declsAndDeps]
      deps  = [dep  | (Right dep) <- declsAndDeps]
  return $ UModule typ name decls deps

renamingDecl :: Parser (UNamedRenaming PhParse)
renamingDecl = annot renamingDecl'
  where
    renamingDecl' = do
      keyword RenamingKW
      name <- RenamingName <$> nameString
      symbol "="
      UNamedRenaming name <$> renamingBlock

satisfaction :: Parser (USatisfaction PhParse)
satisfaction = annot satisfaction'
  where
    -- TODO: fresh gen?
    anonName = GenName "__anonymous_module_name__"
    typ = Concept
    anonModule = annot $ moduleRef typ anonName <|> inlineModule typ anonName

    renamedModule = do
      modul <- anonModule
      renamingBlocks <- many renamingBlock
      return $ RenamedModule modul renamingBlocks

    satisfaction' = do
      -- TODO: add renamings
      keyword SatisfactionKW
      name <- SatName <$> nameString
      symbol "="
      initialModule <- renamedModule
      withModule <- optional $ keyword WithKW >> renamedModule
      modeledModule <- keyword ModelsKW >> renamedModule
      return $ USatisfaction name initialModule withModule modeledModule

declaration :: UModuleType -> Parser ParsedDecl
declaration mtyp = do
  -- TODO: we consider 'require' to be a no-op in this case.
  -- I do not believe that there are cases in which it is not. However, we
  -- might want to keep it as a part of the AST, so this might change.
  optional (keyword RequireKW) *> declaration' <* many semi
  where declaration' =  (TypeDecl <$> annot typeDecl)
                    <|> (CallableDecl <$> annot (callable mtyp))

typeDecl :: Parser (TypeDecl' PhParse)
typeDecl = do
  keyword TypeKW
  name <- typeName <* semi -- TODO: make expr
  return $ Type name

callable :: UModuleType -> Parser (CallableDecl' PhParse)
callable mtyp = do
  callableType <- (keyword AxiomKW >> return Axiom)
              <|> (keyword TheoremKW >> return Axiom)
              <|> (keyword FunctionKW >> return Function)
              <|> (keyword PredicateKW >> return Predicate)
              <|> (keyword ProcedureKW >> return Procedure)
  let nameCons = if callableType == Procedure then ProcName else FuncName
  name <- try symOpName <|> (nameCons <$> nameString)
  args <- case callableType of
      Procedure -> parens ((varMode >>= annVar) `sepBy` symbol ",")
      _ -> parens (annVar UObs `sepBy` symbol ",")
  retType <- case callableType of
      Function  -> symbol ":" *> typeName
      Predicate -> return Pred
      _         -> return Unit
  guard <- optional (keyword GuardKW >> expr)
  mBody <- optional (blockExpr
                <|> (symbol "=" *> (blockExpr <|> (expr <* semi))))
  when (isNothing mBody) semi
  let body = maybe (if mtyp == External then ExternalBody else EmptyBody)
                   MagnoliaBody mBody
  return $ Callable callableType name args retType guard body

annVar :: UVarMode -> Parser ParsedTypedVar
annVar mode = annot annVar'
  where
    annVar' = do
      name <- varName
      typAnn <- symbol ":" *> typeName
      return $ Var mode name typAnn

nameVar :: UVarMode -> Parser ParsedMaybeTypedVar
nameVar mode = annot ((\name -> Var mode name Nothing) <$> varName)

var :: UVarMode -> Parser ParsedMaybeTypedVar
var mode = try ((partialize <$$>) <$> annVar mode) <|> nameVar mode
  where partialize v = v { _varType = Just (_varType v) }

varMode :: Parser UVarMode
varMode = try (keyword ObsKW >> return UObs)
      <|> (keyword OutKW >> return UOut)
      <|> (keyword UpdKW >> return UUpd)

expr :: Parser ParsedExpr
expr = makeExprParser leafExpr ops
  where
    leafExpr = blockExpr
           <|> try ifExpr
           <|> try (callableCall FuncName)
           <|> try (keyword CallKW *> callableCall ProcName)
           <|> annot (keyword SkipKW >> return USkip)
           <|> annot (var UUnk >>= \v -> return $ UVar v)
           <|> parens expr

blockExpr :: Parser ParsedExpr
blockExpr = annot blockExpr'
  where
    blockExpr' = do
      block <- braces (many (exprStmt <* some semi))
      block' <- if null block then (:[]) <$> annot (return USkip)
                else return block
      return $ UBlockExpr (NE.fromList block')

ifExpr :: Parser ParsedExpr
ifExpr = annot ifExpr'
  where
    ifExpr' = do
      keyword IfKW
      cond <- expr
      keyword ThenKW
      bTrue <- expr
      keyword ElseKW
      UIf cond bTrue <$> expr

callableCall :: (String -> Name) -> Parser ParsedExpr
callableCall nameCons = annot callableCall'
  where
    callableCall' = do
      name <- try symOpName <|> (nameCons <$> nameString)
      args <- parens (expr `sepBy` symbol ",")
      ann <- optional (symbol ":" *> typeName)
      return $ UCall name args ann

moduleDependency :: Parser (UModuleDep PhParse)
moduleDependency = annot moduleDependency'
  where
    moduleDependency' = do
      choice [keyword UseKW, keyword RequireKW]
      (castToSignature, name) <-
        try (keyword SignatureKW *>
             ((True,) . ModName <$> parens nameString))
          <|> (False,) . ModName <$> nameString
      renamings <- many renamingBlock
      semi
      return $ UModuleDep name renamings castToSignature

renamingBlock :: Parser (URenamingBlock PhParse)
renamingBlock = annot renamingBlock'
  where
    renamingBlock' = URenamingBlock <$> brackets (renaming `sepBy` symbol ",")

renaming :: Parser (URenaming PhParse)
renaming = try inlineRenaming <|> annot (RefRenaming . RenamingName <$> nameString)

inlineRenaming :: Parser (URenaming PhParse)
inlineRenaming = annot inlineRenaming'
  where
    inlineRenaming' = do
      source <- try symOpName <|> (UnspecName <$> nameString)
      symbol "=>"
      target <- try symOpName <|> (UnspecName <$> nameString)
      return $ InlineRenaming (source, target)

exprStmt :: Parser ParsedExpr
exprStmt = assertStmt <|> try letStmt <|> expr

assertStmt :: Parser ParsedExpr
assertStmt = keyword AssertKW *> annot (UAssert <$> expr)

letStmt :: Parser ParsedExpr
letStmt = annot letStmt'
  where
    letStmt' = do
      keyword LetKW
      isConst <- option False (keyword ObsKW $> True)
      name <- varName
      ann <- optional (symbol ":" *> typeName)
      value <- optional (symbol "=" *> expr)
      let mode | isConst = UObs
               | isNothing value = UOut
               | otherwise = UUpd
      return $ ULet mode name ann value

ops :: [[Operator Parser ParsedExpr]]
ops = [ map unOp ["+", "-", "!", "~"]                -- unary ops
      , map binOp ["*", "/", "%"]                    -- mult ops
      , map binOp ["+", "-"]                         -- add ops
      , map binOp ["<<", ">>"]                       -- shift ops
      , [binOp ".."]                                 -- range
      , map binOp [ "<", ">", ">=", "<=", "=="
                  , "!=", "===", "!=="]              -- comparison ops
      , [binOp "&&"]                                 -- logical and
      , [binOp "||"]                                 -- logical or
      , [binOp "=>", binOp "<=>"]                    -- logical implication
      ]

unOp :: String -> Operator Parser ParsedExpr
unOp s = Prefix $ unOpCall <$> withSrc (symbol s)
  where
    unOpCall (WithSrc src _) e =
      Ann { _ann = src, _elem = UCall (FuncName (s <> "_")) [e] Nothing }


binOp :: String -> Operator Parser ParsedExpr
binOp s = InfixL $ binOpCall <$> withSrc (symbol s)
  where
    binOpCall :: WithSrc a -> ParsedExpr -> ParsedExpr -> ParsedExpr
    binOpCall (WithSrc src _) e1 e2 =
      Ann { _ann = src
          , _elem = UCall (FuncName ("_" <> s <> "_")) [e1, e2] Nothing
          }

-- === general parsing utils ===

braces :: Parser a -> Parser a
braces = between (symbol "{") (symbol "}")

brackets :: Parser a -> Parser a
brackets = between (symbol "[") (symbol "]")

parens :: Parser a -> Parser a
parens = between (symbol "(") (symbol ")")

-- TODO: reason about "==" and the need for "===" and "!==".
symOpName :: Parser Name
symOpName = choice $ map (try . mkSymOpNameParser) symOps
  where symOps = [ "+_", "-_", "!_", "~_"
                 , "_*_", "_/_", "_%_"
                 , "_+_", "_-_"
                 , "_<<_", "_>>_"
                 , "_.._"
                 , "_<_", "_>_", "_>=_", "_<=_", "_==_", "_!=_", "_===_", "_!==_"
                 , "_&&_"
                 , "_||_"
                 , "_=>_", "_<=>_"
                 ]
        mkSymOpNameParser s = do
          symbol s <* notFollowedBy nameChar
          return $ FuncName s

data Keyword = ConceptKW | ImplementationKW | ProgramKW | SignatureKW | ExternalKW
             | RenamingKW | SatisfactionKW
             | ModelsKW | WithKW
             | AxiomKW | FunctionKW | PredicateKW | ProcedureKW | TheoremKW
             | TypeKW
             | RequireKW | UseKW
             | ObsKW | OutKW | UpdKW
             | GuardKW
             | AssertKW | CallKW | IfKW | ThenKW | ElseKW | LetKW | SkipKW
             | PackageKW | ImportKW

keyword :: Keyword -> Parser ()
keyword kw = (lexeme . try) $ string s *> notFollowedBy nameChar
  where
    s = case kw of
      ConceptKW        -> "concept"
      ImplementationKW -> "implementation"
      ProgramKW        -> "program"
      SignatureKW      -> "signature"
      ExternalKW       -> "external"
      RenamingKW       -> "renaming"
      SatisfactionKW   -> "satisfaction"
      ModelsKW         -> "models"
      WithKW           -> "with"
      AxiomKW          -> "axiom"
      FunctionKW       -> "function"
      PredicateKW      -> "predicate"
      ProcedureKW      -> "procedure"
      TheoremKW        -> "theorem"
      TypeKW           -> "type"
      RequireKW        -> "require"
      UseKW            -> "use"
      ObsKW            -> "obs"
      OutKW            -> "out"
      UpdKW            -> "upd"
      GuardKW          -> "guard"
      AssertKW         -> "assert"
      CallKW           -> "call"
      IfKW             -> "if"
      ThenKW           -> "then"
      ElseKW           -> "else"
      LetKW            -> "var"
      SkipKW           -> "skip"
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

semi :: Parser ()
semi = symbol ";"

typeName :: Parser Name
typeName = TypeName <$> nameString

varName :: Parser Name
varName = VarName <$> nameString

packageName :: Parser Name
packageName = PkgName <$> packageString

packageString :: Parser String
packageString =
  lexeme . try $ (:) <$> nameChar <*> many (nameChar <|> char '.')

sourceFileString :: Parser String
sourceFileString =
  lexeme . try $ (:) <$> sourceFileChar <*> many sourceFileChar <> string ".mg"
  where
    sourceFileChar = nameChar <|> char '/'

nameString :: Parser String
nameString = lexeme . try $ (:) <$> nameChar <*> many nameChar

nameChar :: Parser Char
nameChar = alphaNumChar <|> char '_'

annot ::
  (XAnn PhParse e ~ SrcCtx) =>
  Parser (e PhParse) ->
  Parser (Ann PhParse e)
annot p = do
  (WithSrc src element) <- withSrc p
  return $ Ann { _ann = src, _elem = element }

withSrc ::
  Parser a ->
  Parser (WithSrc a)
withSrc p = do
  start <- mkSrcPos <$> getSourcePos
  element <- p
  end <- mkSrcPos <$> getSourcePos
  return $ WithSrc (Just (start, end)) element
  where
    mkSrcPos s = (sourceName s, unPos $ sourceLine s, unPos $ sourceColumn s)
