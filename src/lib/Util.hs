{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Util (
    throwLocatedE, throwNonLocatedE,
    getModules, getNamedRenamings,
    moduleDecls, moduleDepNames,
    mkInlineRenamings, expandRenaming, expandRenamingBlock, checkRenamingBlock)
  where

import Control.Monad
import Control.Monad.Trans.Except
import qualified Data.Map as M
import qualified Data.Text.Lazy as T
import Data.Void

import Env
import PPrint
import Syntax

throwLocatedE :: Monad t => SrcCtx -> T.Text -> ExceptT Err t e
throwLocatedE src err = throwE $ WithSrc src err

throwNonLocatedE :: Monad t => T.Text -> ExceptT Err t e
throwNonLocatedE = throwLocatedE Nothing

-- === top level declarations manipulation ===

getModules :: [UTopLevelDecl p] -> [UModule p]
getModules = foldl extractModule []
  where
    extractModule :: [UModule p] -> UTopLevelDecl p -> [UModule p]
    extractModule acc topLevelDecl
      | UModuleDecl m <- topLevelDecl = m:acc
      | otherwise = acc

getNamedRenamings :: [UTopLevelDecl p] -> [UNamedRenaming p]
getNamedRenamings = foldl extractNamedRenaming []
  where
    extractNamedRenaming
      :: [UNamedRenaming p] -> UTopLevelDecl p -> [UNamedRenaming p]
    extractNamedRenaming acc topLevelDecl
      | UNamedRenamingDecl nr <- topLevelDecl = nr:acc
      | otherwise = acc

-- === modules manipulation ===

moduleDecls :: UModule PhCheck -> Env [UDecl PhCheck]
moduleDecls (Ann _ modul) = case modul of
  UModule _ _ decls _ -> decls
  RefModule _ _ v -> absurd v

moduleDepNames :: UModule PhParse -> [Name]
moduleDepNames (Ann _ modul) = case modul of
  UModule _ _ _ deps -> map nodeName deps
  RefModule _ _ refName -> [refName]

-- === renamings manipulation ===

-- TODO: coercion w/o RefRenaming type?
checkRenamingBlock
  :: Monad t
  => URenamingBlock PhParse
  -> ExceptT Err t (URenamingBlock PhCheck)
checkRenamingBlock (Ann blockSrc (URenamingBlock renamings)) = do
  checkedRenamings <- traverse checkInlineRenaming renamings
  return $ Ann blockSrc $ URenamingBlock checkedRenamings
  where
    checkInlineRenaming
      :: Monad t
      => URenaming PhParse
      -> ExceptT Err t (URenaming PhCheck)
    checkInlineRenaming (Ann src renaming) = case renaming of
      -- It seems that the below pattern matching can not be avoided, due to
      -- coercion concerns (https://gitlab.haskell.org/ghc/ghc/-/issues/15683).
      InlineRenaming r -> return $ Ann (src, LocalDecl) (InlineRenaming r)
      RefRenaming _ -> throwLocatedE src $ "Compiler bug: references left " <>
        "in renaming block at checking time."

expandRenamingBlock
  :: Monad t
  => Env [UNamedRenaming PhCheck]
  -> URenamingBlock PhParse
  -> ExceptT Err t (URenamingBlock PhCheck)
expandRenamingBlock env (Ann src (URenamingBlock renamings)) =
  Ann src . URenamingBlock <$>
    (foldl (<>) [] <$> mapM (expandRenaming env) renamings)

expandRenaming
  :: Monad t
  => Env [UNamedRenaming PhCheck]
  -> URenaming PhParse
  -> ExceptT Err t [URenaming PhCheck]
expandRenaming env (Ann src renaming) = case renaming of
  InlineRenaming ir -> return [Ann (src, LocalDecl) (InlineRenaming ir)]
  RefRenaming ref -> case M.lookup ref env of
    Nothing -> throwLocatedE src $ "Named renaming " <> pshow ref <>
                                   " does not exist in current scope."
    Just namedRenamings -> case namedRenamings of
      []  -> throwLocatedE src $ "Compiler bug in renaming expansion: " <>
                                 "no match but existing renaming name."
      _ -> do
        -- TODO: add disambiguation attempts
        when (length namedRenamings /= 1) $
          throwLocatedE src $ "Could not deduce named renaming instance " <>
                              "from '" <> pshow ref <> "'. Candidates " <>
                              "are: " <> pshow namedRenamings <> "."
        let Ann _ (UNamedRenaming _ (Ann _ (URenamingBlock renamings))) =
              head namedRenamings
        return renamings

mkInlineRenamings :: URenamingBlock PhCheck -> [InlineRenaming]
mkInlineRenamings (Ann _ (URenamingBlock renamings)) =
    map mkInlineRenaming renamings
  where
    mkInlineRenaming :: URenaming PhCheck -> InlineRenaming
    mkInlineRenaming (Ann _ renaming) = case renaming of
      InlineRenaming ir -> ir
      RefRenaming v -> absurd v