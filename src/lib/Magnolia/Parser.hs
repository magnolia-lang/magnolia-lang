{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}

module Magnolia.Parser (parsePackage, parsePackageHead, parseReplCommand) where

import Control.Monad.Combinators.Expr
import Control.Monad.Except (void, when)
import Data.Functor (($>))
import Data.Maybe (isJust, isNothing)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char

import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Text.Lazy as T
import qualified Text.Megaparsec.Char.Lexer as Lex

import Env
import Err
import Magnolia.Syntax
import Magnolia.Util
import Monad

type Parser = Parsec Void String

-- === exception handling utils ===

throwNonLocatedParseError
  :: ParseErrorBundle String Void -> MgMonad b
throwNonLocatedParseError =
  throwNonLocatedE ParseErr . T.pack . errorBundlePretty

-- === repl parsing utils ===

parseReplCommand :: String -> MgMonad Command
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

-- TODO: fix repl to use fully qualified names
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
parsePackageHead :: FilePath -> String -> MgMonad PackageHead
parsePackageHead filePath input =
  case parse (sc >> packageHead filePath input) filePath input of
    Left e -> throwNonLocatedParseError e
    Right ph -> return ph

packageHead :: FilePath -> String -> Parser PackageHead
packageHead filePath input = do
  (src, (pkgName, imports)) <- withSrc packageHead'
  return PackageHead { _packageHeadPath = filePath
                     , _packageHeadFileContent = input
                     , _packageHeadName = pkgName
                     , _packageHeadImports = imports
                     , _packageHeadSrcCtx = src
                    }
  where
    packageHead' = do
      pkgName <- keyword PackageKW *> fullyQualifiedName NSDirectory NSPackage
      -- TODO: error out if package name is different from filepath?
      imports <- choice
        [ keyword ImportKW >>
          (fullyQualifiedName NSDirectory NSPackage `sepBy1` symbol ",")
        , return []
        ] <* semi
      return (pkgName, imports)

-- === package parsing utils ===

parsePackage :: FilePath -> String -> MgMonad ParsedPackage
parsePackage filePath s =
  case parse (sc >> package) filePath s of
    Left e -> throwNonLocatedParseError e
    Right pkg -> return pkg

package :: Parser ParsedPackage
package = annot $ do
  pkgName <- keyword PackageKW *> packageName
  deps <- choice
    [ do keyword ImportKW ;
          annot (MPackageDep <$> fullyQualifiedName NSDirectory NSPackage)
                  `sepBy1` symbol ","
    , return []
    ] <* semi
  decls <- manyTill (many semi *> topLevelDecl <* many semi) eof
  return $ MPackage pkgName decls deps

topLevelDecl :: Parser ParsedTopLevelDecl
topLevelDecl =  try (MModuleDecl <$> moduleDecl)
            <|> (MNamedRenamingDecl <$> renamingDecl)
            <|> (MSatisfactionDecl <$> satisfaction)

moduleDecl :: Parser ParsedModule
moduleDecl = label "module" $ annot $ do
  typ <- moduleType
  name <- ModName <$> nameString
  symbol "="
  (src, mexternal) <- withSrc $ option Nothing (do
    keyword ExternalKW
    backend <- choice [ keyword CxxKW >> return Cxx
                      , keyword JavaScriptKW >> return JavaScript
                      , keyword PythonKW >> return Python]
    externalFqn <- fullyQualifiedName NSPackage NSModule
    return $ Just (backend, externalFqn))
  case mexternal of
    Nothing -> MModule typ name <$> moduleExpr
    Just (backend, externalFqn) ->
      MModule typ name . Ann src . MModuleExternal backend externalFqn <$>
        moduleExpr
  where
    moduleExpr = moduleDef <|> moduleCastedRef <|> moduleRef

moduleType :: Parser MModuleType
moduleType = (keyword ConceptKW >> return Concept)
          <|> (keyword ImplementationKW >> return Implementation)
          <|> (keyword SignatureKW >> return Signature)
          <|> (keyword ProgramKW >> return Program)

moduleRef :: Parser ParsedModuleExpr
moduleRef = annot $ do
  refName <- fullyQualifiedName NSPackage NSModule
  renamingBlocks <- many renamingBlock
  return $ MModuleRef refName renamingBlocks

moduleCastedRef :: Parser ParsedModuleExpr
moduleCastedRef = annot $ do
  keyword SignatureKW
  refName <- parens $ fullyQualifiedName NSPackage NSModule
  renamingBlocks <- many renamingBlock
  return $ MModuleAsSignature refName renamingBlocks

moduleDef :: Parser ParsedModuleExpr
moduleDef = annot $ do
  declsAndDeps <- braces $ many (do
    isExplicitlyRequired <- isJust <$> optional (keyword RequireKW)
    (Left <$> declaration isExplicitlyRequired) <|>
      (if isExplicitlyRequired
       then Right <$> dependency MModuleDepRequire
       else keyword UseKW >> Right <$> dependency MModuleDepUse))
  let decls = [decl | (Left decl) <- declsAndDeps]
      deps  = [dep  | (Right dep) <- declsAndDeps]
  renamingBlocks <- many renamingBlock
  pure $ MModuleDef decls deps renamingBlocks
  where
    declaration :: Bool -> Parser ParsedDecl
    declaration isExplicitlyRequired = label "declaration" $ do
      let modifiers = [Require | isExplicitlyRequired]
      -- TODO: allow requiring callables in externals as well.
      (MTypeDecl modifiers <$> typeDecl) <|>
        (MCallableDecl modifiers <$> callable) <* many semi

    dependency :: MModuleDepType -> Parser ParsedModuleDep
    dependency mmoduleDepType = label "module dependency" $ annot $ do
      moduleExpr <- moduleDef <|> moduleCastedRef <|> moduleRef
      semi
      pure $ MModuleDep mmoduleDepType moduleExpr

renamingDecl :: Parser ParsedNamedRenaming
renamingDecl = annot $ do
  keyword RenamingKW
  name <- RenamingName <$> nameString
  symbol "="
  MNamedRenaming name <$> renamingBlock

satisfaction :: Parser ParsedSatisfaction
satisfaction = annot $ do
  keyword SatisfactionKW
  name <- SatName <$> nameString
  symbol "="
  initialModule <- moduleExpr
  withModule <- optional $ keyword WithKW >> moduleExpr
  modeledModule <-
    choice [keyword ModelsKW, keyword ApproximatesKW] >> moduleExpr
  return $ MSatisfaction name initialModule withModule modeledModule
  where moduleExpr = moduleDef <|> moduleCastedRef <|> moduleRef

typeDecl :: Parser ParsedTypeDecl
typeDecl = annot $ do
  keyword TypeKW
  name <- typeName <* semi -- TODO: make expr
  return $ Type name

callable :: Parser ParsedCallableDecl
callable = annot $ do
  callableType <- (keyword AxiomKW >> return Axiom)
              <|> (keyword TheoremKW >> return Axiom)
              <|> (keyword FunctionKW >> return Function)
              <|> (keyword PredicateKW >> return Predicate)
              <|> (keyword ProcedureKW >> return Procedure)
  let nameCons = if callableType == Procedure then ProcName else FuncName
  name <- try symOpName <|> (nameCons <$> nameString)
  args <- case callableType of
      Procedure -> parens ((varMode >>= annVar) `sepBy` symbol ",")
      _ -> parens (annVar MObs `sepBy` symbol ",")
  retType <- case callableType of
      Function  -> symbol ":" *> typeName
      Predicate -> return Pred
      _         -> return Unit
  guard <- optional (keyword GuardKW >> expr)
  mBody <- optional (blockExpr
                <|> (symbol "=" *> (blockExpr <|> (expr <* semi))))
  when (isNothing mBody) semi
  let body = maybe EmptyBody MagnoliaBody mBody
  return $ Callable callableType name args retType guard body

annVar :: MVarMode -> Parser ParsedTypedVar
annVar mode = annot $ do
  name <- varName
  typAnn <- symbol ":" *> typeName
  return $ Var mode name typAnn

nameVar :: MVarMode -> Parser ParsedMaybeTypedVar
nameVar mode = annot ((\name -> Var mode name Nothing) <$> varName)

var :: MVarMode -> Parser ParsedMaybeTypedVar
var mode = try ((partialize <$$>) <$> annVar mode) <|> nameVar mode
  where partialize v = v { _varType = Just (_varType v) }

varMode :: Parser MVarMode
varMode = try (keyword ObsKW >> return MObs)
      <|> (keyword OutKW >> return MOut)
      <|> (keyword UpdKW >> return MUpd)

expr :: Parser ParsedExpr
expr = makeExprParser leafExpr ops
  where
    leafExpr = -- Statements
               assertStmt
           <|> letStmt
           <|> try assignStmt
               -- "Expressions"
           <|> blockExpr
           <|> ifExpr
           <|> try (callableCall FuncName)
           <|> try (keyword CallKW *> callableCall ProcName)
           <|> annot (keyword ValueKW >> MValue <$> expr)
           <|> annot (keyword SkipKW >> return MSkip)
           <|> annot (var MUnk >>= \v -> return $ MVar v)
           <|> parens expr

blockExpr :: Parser ParsedExpr
blockExpr = annot $ do
  block <- braces (many (expr <* some semi))
  block' <- if null block then (:[]) <$> annot (return MSkip)
            else return block
  let blockType = if any isValueExpr block' then MValueBlock
                  else MEffectfulBlock
  return $ MBlockExpr blockType (NE.fromList block')

-- TODO: re-habilitate the 'end' keyword, or force users to put brackets.
ifExpr :: Parser ParsedExpr
ifExpr = annot $ do
  keyword IfKW
  cond <- expr
  keyword ThenKW
  tbranch <- expr
  keyword ElseKW
  MIf cond tbranch <$> expr

callableCall :: (String -> Name) -> Parser ParsedExpr
callableCall nameCons = annot $ do
  name <- try symOpName <|> (nameCons <$> nameString)
  args <- parens (expr `sepBy` symbol ",")
  ann <- optional (symbol ":" *> typeName)
  return $ MCall name args ann

renamingBlock :: Parser ParsedRenamingBlock
renamingBlock = annot $ do
  (renamings, renamingBlockType) <-
        ((,PartialRenamingBlock) <$>
            doubleBrackets (renaming `sepBy` symbol ","))
    <|> ((,TotalRenamingBlock) <$>
            brackets (renaming `sepBy` symbol ","))
  return $ MRenamingBlock renamingBlockType renamings

renaming :: Parser ParsedRenaming
renaming = try inlineRenaming
        <|> annot (RefRenaming <$> fullyQualifiedName NSPackage NSRenaming)

inlineRenaming :: Parser ParsedRenaming
inlineRenaming = annot $ do
  source <- try symOpName <|> (UnspecName <$> nameString)
  symbol "=>"
  target <- try symOpName <|> (UnspecName <$> nameString)
  return $ InlineRenaming (source, target)

assertStmt :: Parser ParsedExpr
assertStmt = keyword AssertKW *> annot (MAssert <$> expr)

letStmt :: Parser ParsedExpr
letStmt = annot $ do
  keyword LetKW
  ~(Ann ann (Var declMode name mty)) <-
    option MUpd (keyword ObsKW $> MObs) >>= var
  value <- optional (symbol "=" *> expr)
  let mode = if isNothing value then MOut else declMode
  return $ MLet (Ann ann (Var mode name mty)) value

assignStmt :: Parser ParsedExpr
assignStmt = annot $ do
  lhsVarExpr <- annot (nameVar MUnk >>= \v -> return $ MVar v)
  symbol "="
  rhsExpr <- expr
  return $ MCall (ProcName "_=_") [lhsVarExpr, rhsExpr] Nothing

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
unOp s = Prefix $ unOpCall <$> withSrc (symOp s)
  where
    unOpCall (src, _) e =
      Ann { _ann = src, _elem = MCall (FuncName (s <> "_")) [e] Nothing }


binOp :: String -> Operator Parser ParsedExpr
binOp s = InfixL $ binOpCall <$> withSrc (symOp s)
  where
    binOpCall :: (SrcCtx, a) -> ParsedExpr -> ParsedExpr -> ParsedExpr
    binOpCall (src, _) e1 e2 =
      Ann { _ann = src
          , _elem = MCall (FuncName ("_" <> s <> "_")) [e1, e2] Nothing
          }

-- === general parsing utils ===

braces :: Parser a -> Parser a
braces = between (symbol "{") (symbol "}")

brackets :: Parser a -> Parser a
brackets = between (symbol "[") (symbol "]")

doubleBrackets :: Parser a -> Parser a
doubleBrackets = between (symbol "[[") (symbol "]]")

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

data Keyword = ConceptKW | ImplementationKW | ProgramKW | SignatureKW
             | ExternalKW
             | CxxKW | JavaScriptKW | PythonKW
             | RenamingKW | SatisfactionKW
             | ModelsKW | ApproximatesKW | WithKW
             | AxiomKW | FunctionKW | PredicateKW | ProcedureKW | TheoremKW
             | TypeKW
             | RequireKW | UseKW
             | ObsKW | OutKW | UpdKW
             | GuardKW
             | AssertKW | CallKW | IfKW | ThenKW | ElseKW | LetKW | SkipKW
             | ValueKW
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
      CxxKW            -> "C++"
      JavaScriptKW     -> "JavaScript"
      PythonKW         -> "Python"
      RenamingKW       -> "renaming"
      SatisfactionKW   -> "satisfaction"
      ModelsKW         -> "models"
      ApproximatesKW   -> "approximates"
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
      ValueKW          -> "value"
      PackageKW        -> "package"
      ImportKW         -> "imports"

sc :: Parser ()
sc = skipMany $
  choice [ hidden space1
         , hidden $ Lex.skipLineComment "//"
         , hidden $ Lex.skipBlockCommentNested "/*<" ">*/"
         , hidden $ Lex.skipBlockComment "/*" "*/"
         ]

-- skipLineComment :: Parser ()
-- skipLineComment = Lex.skipLineComment "//"

-- skipBlockComment :: Parser ()
-- skipBlockComment =
--   Lex.skipBlockCommentNested "/*<" ">*/" <> Lex.skipBlockComment "/*" "*/"

lexeme :: Parser a -> Parser a
lexeme = Lex.lexeme sc

symbol :: String -> Parser ()
symbol = void . Lex.symbol sc

symOp :: String -> Parser ()
symOp s = lexeme . try $ string s >> notFollowedBy symChar
  where
    symChar = choice $ map char "!%&*+-./<=>|~"

semi :: Parser ()
semi = symbol ";"

typeName :: Parser Name
typeName = TypeName <$> nameString

varName :: Parser Name
varName = VarName <$> nameString

packageName :: Parser Name
packageName = fromFullyQualifiedName <$>
  fullyQualifiedName NSDirectory NSPackage

-- TODO: remove
packageString :: Parser String
packageString = _name <$> packageName

fullyQualifiedName :: NameSpace -> NameSpace
                   -> Parser FullyQualifiedName
fullyQualifiedName scopeNs targetNs = do
  nameStrings <- stringParser `sepBy1` char '.'
  -- nameStrings *must* contain at least one element here. Therefore, we can
  -- safely call last.
  let targetName = Name targetNs (last nameStrings)
  case init nameStrings of
    [] -> return $ FullyQualifiedName Nothing targetName
    ss -> return $ FullyQualifiedName
      (Just $ Name scopeNs (L.intercalate "." ss)) targetName
  where
    -- When parsing a package path through directories, for convenience, we
    -- may want to allow hyphens as part of the names. We assume that directory
    -- and filenames do not start with the character '-' though, as this may
    -- lead to trouble when parsing command line options if one is not careful.
    allowHyphens = scopeNs == NSDirectory && targetNs == NSPackage
    stringParser = if allowHyphens
      then lexeme . try $ (:) <$> nameChar <*> many (nameChar <|> char '-')
      else nameString

-- TODO: deprecate repl
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
  Parser (Parsed e)
annot p = do
  (src, element) <- withSrc p
  return $ Ann { _ann = src, _elem = element }

withSrc ::
  Parser a ->
  Parser (SrcCtx, a)
withSrc p = do
  start <- mkSrcPos <$> getSourcePos
  element <- p
  end <- mkSrcPos <$> getSourcePos
  return (SrcCtx $ Just (start, end), element)
  where
    mkSrcPos s = (sourceName s, unPos $ sourceLine s, unPos $ sourceColumn s)
