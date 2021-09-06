{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wno-orphans #-}

module MgToXml (pprintListToXMLDocument, pprintToXML, toXML) where

import Control.Monad (join)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isJust)
import qualified Data.Text as T
import Data.Void (absurd)
import qualified Data.XML.Types as X
import Text.XML.Writer hiding (many)

import Err
import Magnolia.PPrint hiding (pprint)
import Magnolia.Syntax

many :: ToXML a => X.Name -> [a] -> XML
many name list = element name $ mapM_ toXML list

pprintToXML :: ToXML a => a -> IO ()
pprintToXML = pprint . document "root" . toXML

pprintListToXMLDocument :: ToXML a => X.Name -> [a] -> IO ()
pprintListToXMLDocument xmlName = pprint . document "root" . many xmlName

instance ToXML Err where
  toXML err@(Err _ errorLoc _ _) =
    element "error" (content (pshow err) >> toXML errorLoc)

instance ToXML TcPackage where
  toXML (Ann _ (MPackage name topLevelDecls packageDependencies)) =
    elementA "package" [ ("name", pshow name) ] (do
      many "top-level-decls" (join . map snd $ M.toList topLevelDecls)
      many "package-dependencies" packageDependencies)

instance ToXML TcPackageDep where
  toXML (Ann src (MPackageDep fqn)) =
    elementA "package-dependency" [ ("name", pshow fqn) ] src

instance ToXML TcTopLevelDecl where
  toXML tcTopLevelDecl =
    let elementName = "top-level-decl"
        hasType s = [("top-level-decl-type", s)]
    in case tcTopLevelDecl of
      MNamedRenamingDecl namedRenaming ->
        elementA elementName (hasType "named-renaming") namedRenaming
      MModuleDecl module' ->
        elementA elementName (hasType "module") module'
      MSatisfactionDecl satisfaction ->
        elementA elementName (hasType "satisfaction") satisfaction

instance ToXML TcNamedRenaming where
  toXML (Ann src (MNamedRenaming name renamingBlock)) =
    elementA "named-renaming" [ ("name", pshow name) ]
      (toXML renamingBlock >> toXML src)

instance ToXML TcSatisfaction where
  -- TODO: satisfactions are not checked for the moment, so we defer writing
  -- this instance
  toXML = undefined

instance ToXML TcModule where
  toXML (Ann src (MModule ty name tcModuleExpr)) =
    elementA "module" [ ("module-type", pshow ty)
                      , ("name", pshow name)
                      ] (toXML tcModuleExpr >> toXML src)

instance ToXML TcModuleExpr where
  toXML (Ann src tcModuleExpr) = case tcModuleExpr of
    MModuleRef v _ -> absurd v
    MModuleAsSignature v _ -> absurd v
    MModuleExternal _ _ v -> absurd v
    MModuleDef decls deps renamingBlocks ->
      elementA "module-expression" [] $ do
        many "declarations" (join . map snd $ M.toList decls)
        many "dependencies" deps
        many "renaming-blocks" renamingBlocks
        toXML src

instance ToXML TcModuleDep where
  toXML (Ann _ (MModuleDep depType depModuleExpr)) =
    elementA "module-dependency" [ ("dependency-type", pshow depType) ]
            depModuleExpr

instance ToXML TcRenamingBlock where
  toXML (Ann src (MRenamingBlock ty renamings)) =
    let tyString = case ty of
          PartialRenamingBlock -> "partial"
          TotalRenamingBlock -> "total"
    in elementA "renaming-block" [ ("renaming-block-type", tyString) ] (do
        many "renamings" renamings
        toXML src)

instance ToXML TcRenaming where
  toXML (Ann src renaming) = case renaming of
    RefRenaming v -> absurd v
    InlineRenaming (sourceName, targetName) ->
      elementA "renaming" [ ("source-name", pshow sourceName)
                          , ("target-name", pshow targetName)
                          ] src

instance ToXML TcDecl where -- TODO: output modifiers
  toXML decl = case decl of
    MCallableDecl modifiers cdecl -> toXML cdecl
    MTypeDecl modifiers tdecl -> toXML tdecl

instance ToXML TcTypeDecl where
  toXML (Ann (mconDeclO, absDeclOs) (Type tyName)) =
    elementA "type" [ ("name", pshow tyName)
                    ] (do many "abstract-origins" $ NE.toList absDeclOs
                          case mconDeclO of
                            Nothing -> pure ()
                            Just conDeclO -> toXML conDeclO)

instance ToXML TcCallableDecl where
  toXML (Ann (mconDeclO, absDeclOs)
             (Callable ctyp name args retTy mguard body)) =
    elementA "callable" [ ("callable-type", pshow ctyp)
                        , ("name", pshow name)
                        , ("return-type", pshow retTy)
                        ] (do many "arguments" args
                              case mguard of
                                Nothing -> pure ()
                                Just guard -> element "guard" guard
                              element "body" body
                              many "abstract-origins" $ NE.toList absDeclOs
                              case mconDeclO of
                                Nothing -> pure ()
                                Just conDeclO -> toXML conDeclO)

instance ToXML (CBody PhCheck) where
  toXML body = let elementName = "callable-body" in case body of
    ExternalBody _ -> elementA elementName (hasType "provided-externally") empty
    EmptyBody -> elementA elementName (hasType "not-provided") empty
    BuiltinBody -> elementA elementName (hasType "built-in") empty
    MagnoliaBody expr -> elementA elementName (hasType "provided-locally") expr
    where
      hasType ty = [("body-type", ty)]

instance ToXML AbstractDeclOrigin where
  toXML absDeclO = case absDeclO of
    AbstractLocalDecl declO -> element "abstract-origin" declO
    _ -> error $ "AbstractDeclOrigin should only wrap a local declaration. " <>
                 "This error should disappear once declOs are updated (TODO)"

instance ToXML ConcreteDeclOrigin where
  toXML conDeclO = let elementName = "concrete-origin" in
    case conDeclO of
      ConcreteMagnoliaDecl declO ->
        elementA elementName (comesFrom "magnolia") declO
      ConcreteExternalDecl declO extDeclDetails ->
        elementA elementName (comesFrom "external") $ do toXML declO
                                                         toXML extDeclDetails
      GeneratedBuiltin -> elementA elementName (comesFrom "built-in") empty
    where
      comesFrom s = [("origin", s)]

instance ToXML DeclOrigin where
  toXML declO = let elementName = "originates-from" in case declO of
    LocalDecl src ->
      elementA elementName [ ("is-local", "true") ] src
    ImportedDecl originName src ->
      elementA elementName [ ("is-local", "false")
                           , ("comes-from", pshow originName)
                           ] src

instance ToXML SrcCtx where
  toXML (SrcCtx mspan) = let elementName = "source-location" in case mspan of
      Nothing -> elementA elementName [ ("is-provided", "false") ] empty
      Just ((filename, startLine, startCol), (_, endLine, endCol)) ->
        elementA elementName [ ("is-provided", "true")
                             , ("filename", pshow filename)
                             , ("start-line", pshow startLine)
                             , ("start-col", pshow startCol)
                             , ("end-line", pshow endLine)
                             , ("end-col", pshow endCol)
                             ] empty

instance ToXML ExternalDeclDetails where
  toXML extDeclDetails =
    let backend = externalDeclBackend extDeclDetails
        declName = externalDeclElementName extDeclDetails
        filepath = externalDeclFilePath extDeclDetails
        moduleName = externalDeclModuleName extDeclDetails
    in elementA "external-decl-details"  [ ("backend", pshow backend)
                                         , ("decl-name", pshow declName)
                                         , ("filepath", T.pack filepath)
                                         , ("module-name", pshow moduleName)
                                         ] empty

instance ToXML TcExpr where
  toXML (Ann ty expr) = case expr of
    MVar v -> elementA "var-ref" tyAtt v
    MCall name args mexplicitCast ->
      elementA "call" (tyAtt <> [ ("name", pshow name)
                                , ("has-explicit-cast"
                                  , pshow $ isJust mexplicitCast
                                  )
                                ])
               (many "arg" args)
    MBlockExpr MValueBlock stmts ->
      elementA "value-block" tyAtt (many "statement" $ NE.toList stmts)
    MBlockExpr MEffectfulBlock stmts ->
      elementA "effectful-block" tyAtt (many "statement" $ NE.toList stmts)
    MValue e -> elementA "value" tyAtt $ element "expr" e
    MLet v me -> elementA "var-decl" tyAtt
      (do toXML v
          case me of
            Nothing -> pure ()
            Just e -> element "initial-value" e)
    MIf cond te fe -> elementA "if" tyAtt (do element "condition" $ toXML cond
                                              element "true-branch" $ toXML te
                                              element "false-branch" $ toXML fe)
    MAssert e -> elementA "assert" tyAtt e
    MSkip -> elementA "skip" tyAtt empty
    where
      tyAtt = [("type", pshow ty)]

instance ToXML (Ann PhCheck (MVar m)) where
  toXML (Ann ty (Var mode name _)) =
    elementA "var" [ ("mode", pshow mode)
                   , ("name", pshow name)
                   , ("type", pshow ty)
                   ] empty