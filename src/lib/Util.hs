{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Util (
    MgMonad, runMgMonad,
    foldMAccumErrors, foldMAccumErrorsAndFail, recover, recoverWithDefault,
    throwLocatedE, throwNonLocatedE,
    mkPkgNameFromPath, mkPkgPathFromName, mkPkgPathFromStr, isPkgPath,
    lookupTopLevelRef,
    mkInlineRenamings, expandRenaming, expandRenamingBlock, checkRenamingBlock,
    isLocalDecl,
    isValueExpr)
  where

import Control.Monad
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (lift)
import qualified Control.Monad.Trans.Except as E
import qualified Control.Monad.Trans.State as St
import qualified Data.List as L
import qualified Data.Map as M
import Data.Maybe (fromJust, isNothing)
import qualified Data.Set as S
import Data.Text.Prettyprint.Doc (Pretty)
import qualified Data.Text.Lazy as T
import Data.Void

import Env
import PPrint
import Syntax

-- === magnolia monad utils ===

newtype MgMonadT e s m a = MgMonadT { unMg :: E.ExceptT e (St.StateT s m) a }
                           deriving (Functor, Applicative, Monad)

instance MonadIO m => MonadIO (MgMonadT e s m) where
  liftIO = MgMonadT . liftIO

type MgMonad = MgMonadT () (S.Set Err) IO

runMgMonadT :: MgMonadT e s m a -> s -> m (Either e a, s)
runMgMonadT mgm s = (`St.runStateT` s) . E.runExceptT $ unMg mgm

runMgMonad :: MgMonad a -> IO (Either () a, S.Set Err)
runMgMonad = (`runMgMonadT` S.empty)

get :: Monad m => MgMonadT e s m s
get = MgMonadT (lift St.get)

modify :: Monad m => (s -> s) -> MgMonadT e s m ()
modify f = MgMonadT (lift (St.modify f))

throwE :: Monad m => e -> MgMonadT e s m a
throwE = MgMonadT . E.throwE

catchE
  :: Monad m => MgMonadT e s m a -> (e -> MgMonadT e s m a) -> MgMonadT e s m a
catchE me f = MgMonadT (unMg me `E.catchE` (unMg . f))

-- === error handling utils ===

foldMAccumErrors :: Foldable t => (b -> a -> MgMonad b) -> b -> t a -> MgMonad b
foldMAccumErrors f = foldM (\b a -> f b a `catchE` const (return b))

foldMAccumErrorsAndFail
  :: Foldable t => (b -> a -> MgMonad b) -> b -> t a -> MgMonad b
foldMAccumErrorsAndFail f initialValue elems = do
    foldMRes <- foldMAccumErrors f initialValue elems
    errs <- get
    maybe (return foldMRes) (const (throwE ())) (S.lookupMin errs)

recover :: Monoid b => (a -> MgMonad b) -> (a -> MgMonad b)
recover f a = f a `catchE` const (return mempty)

recoverWithDefault :: (a -> MgMonad b) -> b -> (a -> MgMonad b)
recoverWithDefault f b a = f a `catchE` const (return b)

throwLocatedE :: ErrType -> SrcCtx -> T.Text -> MgMonad b
throwLocatedE errType src err =
  modify (S.insert (Err errType src err)) >> throwE ()

throwNonLocatedE :: ErrType -> T.Text -> MgMonad b
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

-- === top-level manipulation ===

lookupTopLevelRef
  :: ( XAnn PhCheck e ~ DeclOrigin
     , Show (e PhCheck)
     , Pretty (e PhCheck)
     , NamedNode (e PhCheck)
     )
  => SrcCtx
  -> Env [Ann PhCheck e]
  -> FullyQualifiedName
  -> MgMonad (Ann PhCheck e)
lookupTopLevelRef src env ref@(FullyQualifiedName mscopeName targetName) =
  case M.lookup targetName env of
    Nothing -> throwLocatedE UnboundTopLevelErr src $ pshow ref
    Just matches ->
      if null matches then throwLocatedE CompilerErr src $
        "name " <> pshow targetName <> " exists but is not bound to anything"
      else do
        compatibleMatches <-
            if isNothing mscopeName then do
              -- There are two valid cases in which it is possible not to
              -- provide a scope name:
              -- (1) there is only one element in scope corresponding to the
              --     target name, and the code is therefore unambiguous; in
              --     this case, the name of the scope is here optional;
              -- (2) there is exactly one locally defined element in scope.
              --     In this case, the name of the scope *can not* be provided,
              --     because the surrounding package is not yet *in scope*.
              let localDecls = filter isLocalDecl matches
              if null localDecls then return matches
              else do
                unless (null $ tail localDecls) $
                  throwLocatedE CompilerErr src $
                    "there are several locally defined top-level constructs " <>
                    "with the name " <> pshow targetName <> " in scope"
                return localDecls
            else return $ filter (matchesImportScope (fromJust mscopeName))
                                  matches
        -- The reference is only valid when a match exists and is unambiguous,
        -- i.e. in cases when compatibleMatches contains exactly one element.
        when (null compatibleMatches) $ -- no match
          throwLocatedE UnboundTopLevelErr src $ pshow ref
        unless (null $ tail compatibleMatches) $ -- more than one match
          throwLocatedE AmbiguousTopLevelRefErr src $ pshow ref <>
            "'. Candidates are: " <> pshow (map mkFQName compatibleMatches)
        return $ head compatibleMatches
  where
    matchesImportScope
      :: XAnn PhCheck e ~ DeclOrigin => Name -> Ann PhCheck e -> Bool
    matchesImportScope scopeName (Ann declO _) = case declO of
      ImportedDecl (FullyQualifiedName (Just scopeName') _) _ ->
        scopeName == scopeName'
      _ -> False

    mkFQName
      :: (XAnn PhCheck e ~ DeclOrigin, NamedNode (e PhCheck))
      => Ann PhCheck e -> Name
    mkFQName (Ann declO node) = case declO of
      LocalDecl _ -> nodeName node
      ImportedDecl fqn _ -> fromFullyQualifiedName fqn

-- === renamings manipulation ===

-- TODO: coercion w/o RefRenaming type?
checkRenamingBlock
  :: MRenamingBlock PhParse
  -> MgMonad (MRenamingBlock PhCheck)
checkRenamingBlock (Ann blockSrc (MRenamingBlock renamings)) = do
  checkedRenamings <- traverse checkInlineRenaming renamings
  return $ Ann blockSrc $ MRenamingBlock checkedRenamings
  where
    checkInlineRenaming
      :: MRenaming PhParse
      -> MgMonad (MRenaming PhCheck)
    checkInlineRenaming (Ann src renaming) = case renaming of
      -- It seems that the below pattern matching can not be avoided, due to
      -- coercion concerns (https://gitlab.haskell.org/ghc/ghc/-/issues/15683).
      InlineRenaming r -> return $ Ann (LocalDecl src) (InlineRenaming r)
      RefRenaming _ -> throwLocatedE CompilerErr src $ "references left " <>
        "in renaming block at checking time"

expandRenamingBlock
  :: Env [MNamedRenaming PhCheck]
  -> MRenamingBlock PhParse
  -> MgMonad (MRenamingBlock PhCheck)
expandRenamingBlock env (Ann src (MRenamingBlock renamings)) =
  Ann src . MRenamingBlock <$>
    (foldl (<>) [] <$> mapM (expandRenaming env) renamings)

expandRenaming
  :: Env [MNamedRenaming PhCheck]
  -> MRenaming PhParse
  -> MgMonad [MRenaming PhCheck]
expandRenaming env (Ann src renaming) = case renaming of
  InlineRenaming ir -> return [Ann (LocalDecl src) (InlineRenaming ir)]
  RefRenaming ref -> do
    Ann _ (MNamedRenaming _ (Ann _ (MRenamingBlock renamings))) <-
      lookupTopLevelRef src env ref
    return renamings

mkInlineRenamings :: MRenamingBlock PhCheck -> [InlineRenaming]
mkInlineRenamings (Ann _ (MRenamingBlock renamings)) =
    map mkInlineRenaming renamings
  where
    mkInlineRenaming :: MRenaming PhCheck -> InlineRenaming
    mkInlineRenaming (Ann _ renaming) = case renaming of
      InlineRenaming ir -> ir
      RefRenaming v -> absurd v

-- === declaration manipulation ===

isLocalDecl :: XAnn p e ~ DeclOrigin => Ann p e -> Bool
isLocalDecl (Ann ann _) = case ann of LocalDecl _ -> True ; _ -> False

-- === expression manipulation ===

isValueExpr :: MExpr p -> Bool
isValueExpr (Ann _ e) = case e of
  MValue _ -> True
  MIf _ eTrue eFalse -> isValueExpr eTrue || isValueExpr eFalse
  _ -> False