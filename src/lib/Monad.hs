{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}

module Monad (
  -- * MgMonad utils
    MgMonad
  , runMgMonad
  -- ** MgMonad scope-related utils
  , enter
  , getParentPackageName
  , getParentModuleName
  -- ** MgMonad error-related utils
  , foldMAccumErrors
  , foldMAccumErrorsAndFail
  , recover
  , recoverWithDefault
  , throwLocatedE
  , throwNonLocatedE
  -- * Reexport error utils
  , Err (..)
  , ErrType (..)
  )
  where

import Control.Monad
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (lift)
import qualified Control.Monad.Trans.Except as E
import qualified Control.Monad.Trans.Reader as R
import qualified Control.Monad.Trans.State as St
import qualified Data.List as L
import qualified Data.Set as S
import qualified Data.Text as T

import Env
import Err

-- === magnolia monad utils ===

-- | A monad transformer wrapping the minimum stack of monads required within
-- the compiler.
newtype MgMonadT e r s m a =
  MgMonadT { unMgT :: E.ExceptT e (R.ReaderT r (St.StateT s m)) a }
  deriving (Functor, Applicative, Monad)

instance MonadIO m => MonadIO (MgMonadT e r s m) where
  liftIO = MgMonadT . liftIO

type MgMonad = MgMonadT () [Name] (S.Set Err) IO

runMgMonadT :: MgMonadT e r s m a -> r -> s -> m (Either e a, s)
runMgMonadT mgm r s =
  (`St.runStateT` s) . (`R.runReaderT` r) . E.runExceptT $ unMgT mgm

runMgMonad :: MgMonad a -> IO (Either () a, S.Set Err)
runMgMonad m = runMgMonadT m [] S.empty

-- | Retrieves the state within a MgMonadT.
get :: Monad m => MgMonadT e r s m s
get = MgMonadT (lift (lift St.get))

-- | Modifies the state within a MgMonadT using the provided function.
modify :: Monad m => (s -> s) -> MgMonadT e r s m ()
modify f = MgMonadT (lift (lift (St.modify f)))

-- | Throws an exception within a MgMonadT.
throwE :: Monad m => e -> MgMonadT e s r m a
throwE = MgMonadT . E.throwE

-- | Catches an exception within a MgMonadT.
catchE :: Monad m => MgMonadT e r s m a -> (e -> MgMonadT e r s m a)
       -> MgMonadT e r s m a
catchE me f = MgMonadT (unMgT me `E.catchE` (unMgT . f))

-- | Retrieves the environment within a MgMonadT.
ask :: Monad m => MgMonadT e r s m r
ask = MgMonadT (lift R.ask)

-- | Executes a computation in a modified environment within a MgMonadT.
local :: Monad m => (r -> r) -> MgMonadT e r s m a -> MgMonadT e r s m a
local f = MgMonadT . E.mapExceptT (R.local f) . unMgT

-- === error handling utils ===

-- | Accumulates exceptions for each element in the traversable when performing
-- a fold. Always succeeds.
foldMAccumErrors :: Foldable t => (b -> a -> MgMonad b) -> b -> t a -> MgMonad b
foldMAccumErrors f = foldM (\b a -> f b a `catchE` const (return b))

-- | Similar to foldMAccumErrors, except that if any exception is caught
-- during processing, an exception is thrown at the end of the fold.
foldMAccumErrorsAndFail
  :: Foldable t => (b -> a -> MgMonad b) -> b -> t a -> MgMonad b
foldMAccumErrorsAndFail f initialValue elems = do
    foldMRes <- foldMAccumErrors f initialValue elems
    errs <- get
    maybe (return foldMRes) (const (throwE ())) (S.lookupMin errs)

-- | Runs a computation and recovers by returning a (generated) default value
-- if an exception is caught.
recover :: Monoid b => (a -> MgMonad b) -> (a -> MgMonad b)
recover f a = f a `catchE` const (return mempty)

-- | Similar to recover, except that the default value to return is provided
-- by the caller.
recoverWithDefault :: (a -> MgMonad b) -> b -> (a -> MgMonad b)
recoverWithDefault f b a = f a `catchE` const (return b)

-- | Returns the list of the parent scopes associated with the computation.
-- The list is ordered from the outermost to the innermost scope.
parentScopes :: MgMonad [Name]
parentScopes = L.reverse <$> ask

-- | Returns the innermost package name associated with the computation. An
-- exception is thrown if it doesn't exist.
getParentPackageName :: MgMonad Name
getParentPackageName = getParentNSName NSPackage

-- | Returns the innermost module name associated with the computation. An
-- exception is thrown if it doesn't exist.
getParentModuleName :: MgMonad Name
getParentModuleName = getParentNSName NSModule

getParentNSName :: NameSpace -> MgMonad Name
getParentNSName ns = do
  ps <- ask
  case L.find ((== ns) . _namespace) ps of
    Nothing -> throwNonLocatedE CompilerErr $
      "attempted to query a parent " <> T.pack (show ns) <> " name but " <>
      "none exists"
    Just name -> return name

-- | Runs a computation within a child scope.
enter :: Name      -- ^ the name of the child scope
      -> MgMonad a -- ^ the computation
      -> MgMonad a
enter = local . (:)

-- | Throws an error with source information.
throwLocatedE :: ErrType -- ^ the type of error that was encountered
              -> SrcCtx  -- ^ the location of the error within the source
              -> T.Text  -- ^ the error text
              -> MgMonad b
throwLocatedE errType src err = do
  ps <- parentScopes
  modify (S.insert (Err errType src ps err)) >> throwE ()

-- | Like throwLocatedE, except that the source information is omitted.
throwNonLocatedE :: ErrType -> T.Text -> MgMonad b
throwNonLocatedE errType = throwLocatedE errType (SrcCtx Nothing)

