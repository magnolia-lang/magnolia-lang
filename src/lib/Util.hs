{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Util (
    throwLocatedE, throwNonLocatedE,
    mkPkgNameFromPath, mkPkgPathFromName, mkPkgPathFromStr, isPkgPath,
    mkInlineRenamings, expandRenaming, expandRenamingBlock, checkRenamingBlock,
    callableIsImplemented, replaceGuard)
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

throwLocatedE :: Monad t => ErrType -> SrcCtx -> T.Text -> ExceptT Err t e
throwLocatedE errType src err = throwE $ Err errType src err

throwNonLocatedE :: Monad t => ErrType -> T.Text -> ExceptT Err t e
throwNonLocatedE errType = throwLocatedE errType Nothing

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
      InlineRenaming r -> return $ Ann (LocalDecl src) (InlineRenaming r)
      RefRenaming _ -> throwLocatedE CompilerErr src $ "references left " <>
        "in renaming block at checking time"

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
  InlineRenaming ir -> return [Ann (LocalDecl src) (InlineRenaming ir)]
  RefRenaming ref -> case M.lookup ref env of
    Nothing -> throwLocatedE UnboundNamedRenamingErr src $ pshow ref
    Just namedRenamings -> case namedRenamings of
      []  -> throwLocatedE CompilerErr src
        "renaming name exists but is not bound in renaming expansion"
      _ -> do
        -- TODO: add disambiguation attempts
        when (length namedRenamings /= 1) $
          throwLocatedE AmbiguousNamedRenamingRefErr src $ pshow ref <>
            "'. Candidates are: " <> pshow namedRenamings
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

-- === declarations manipulation ===

-- UCallable UCallable Name [UVar p] UType (CGuard p) (CBody p)
-- TODO: avoid partiality
replaceGuard :: UDecl' p -> CGuard p -> UDecl' p
replaceGuard ~(UCallable callableType name args retType _ mbody) mguard =
  UCallable callableType name args retType mguard mbody

-- TODO: avoid partiality
callableIsImplemented :: UDecl p -> Bool
callableIsImplemented ~(Ann _ (UCallable _ _ _ _ _ mbody)) = case mbody of
  Nothing -> False
  _ -> True