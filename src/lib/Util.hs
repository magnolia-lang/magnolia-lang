{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Util (
    throwLocatedE, throwNonLocatedE,
    mkPkgNameFromPath, mkPkgPathFromName, mkPkgPathFromStr, isPkgPath,
    mkInlineRenamings, expandRenaming, expandRenamingBlock, checkRenamingBlock)
  where

import Control.Monad
import Control.Monad.Trans.Except
import qualified Data.List as L
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

-- === package name manipulation ===

mkPkgNameFromPath :: String -> Name
mkPkgNameFromPath pkgPath = PkgName $ map (\c -> if c == '/' then '.' else c) $
  take (length pkgPath - 3) pkgPath

mkPkgPathFromName :: Name -> String
mkPkgPathFromName = mkPkgPathFromStr. _name

mkPkgPathFromStr :: String -> String
mkPkgPathFromStr pkgStr =
  map (\c -> if c == '.' then '/' else c) pkgStr <> ".mg"

isPkgPath :: String -> Bool
isPkgPath s = ".mg" `L.isSuffixOf` s

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