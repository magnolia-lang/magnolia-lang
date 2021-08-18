{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Magnolia.Util (
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
  -- * Package name manipulation utils
  , mkPkgNameFromPath
  , mkPkgPathFromName
  , mkPkgPathFromStr
  , isPkgPath
  -- * Dependencies-related utils
  , checkNoCycle
  -- * Top-level manipulation utils
  , lookupTopLevelRef
  , topSortTopLevelE
  , isLocalTopLevelDeclOrigin
   -- * Renaming manipulation utils
  , renamingBlockToInlineRenamings
  -- * Declaration manipulation utils
  -- * Expression manipulations utils
  , isValueExpr
  )
  where

import Control.Monad
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (lift)
import qualified Control.Monad.Trans.Except as E
import qualified Control.Monad.Trans.Reader as R
import qualified Control.Monad.Trans.State as St
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Text.Prettyprint.Doc (Pretty)
import qualified Data.Text.Lazy as T
import Data.Void (absurd)

import Env
import Magnolia.PPrint
import Magnolia.Syntax

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
      "attempted to query a parent " <> pshow ns <> " name but none exists"
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
  :: ( XAnn PhCheck e ~ TopLevelDeclOrigin
     , Show (e PhCheck)
     , Pretty (e PhCheck)
     , HasName (e PhCheck)
     )
  => SrcCtx -> Env [Tc e] -> FullyQualifiedName -> MgMonad (Tc e)
lookupTopLevelRef src env ref = case M.lookup (_targetName ref) env of
  Nothing -> throwLocatedE UnboundTopLevelErr src $ pshow ref
  Just [] -> throwLocatedE CompilerErr src $
    "name " <> pshow (_targetName ref) <> " exists but is not bound to anything"
  Just matches -> do
    localDecls <- filterM (isLocalTopLevelDeclOrigin . _ann) matches
    compatibleMatches <- case _scopeName ref of
      Nothing ->
        -- There are two valid cases in which it is possible not to
        -- provide a scope name:
        -- (1) there is only one element in scope corresponding to the
        --     target name, and the code is therefore unambiguous; in
        --     this case, the name of the scope is here optional;
        -- (2) there is exactly one locally defined element in scope.
        if null localDecls
        then return matches
        else do unless (null $ tail localDecls) $
                  throwLocatedE CompilerErr src $
                    "there are several locally defined top-level constructs " <>
                    "with the name " <> pshow (_targetName ref) <> " in scope"
                return localDecls
      Just scopeName -> do
        parentPkgName <- getParentPackageName
        if scopeName == parentPkgName
        then do
          when (null localDecls) $
            throwLocatedE UnboundTopLevelErr src $ pshow ref
          unless (null $ tail localDecls) $
            throwLocatedE CompilerErr src $
              "there are several locally defined top-level constructs " <>
              "with the name " <> pshow (_targetName ref) <> " in scope"
          return localDecls
        else let matchesImportScope (Ann (TopLevelDeclOrigin fqn _) _) =
                  Just scopeName == _scopeName fqn
             in return $ filter matchesImportScope matches
    -- The reference is only valid when a match exists and is unambiguous,
    -- i.e. in cases when compatibleMatches contains exactly one element.
    when (null compatibleMatches) $ -- no match
      throwLocatedE UnboundTopLevelErr src $ pshow ref
    unless (null $ tail compatibleMatches) $ -- more than one match
      throwLocatedE AmbiguousTopLevelRefErr src $ pshow ref <>
        "'. Candidates are: " <> T.intercalate ", " (
          map (pshow . getName . _ann) compatibleMatches)
    return $ head compatibleMatches

-- | Topologically sorts top-level named elements (i.e. named renamings
-- declarations, modules) based on their local dependencies.
topSortTopLevelE :: (HasDependencies a, HasName a) => Name -> [a] -> [G.SCC a]
topSortTopLevelE pkgName elems = G.stronglyConnComp
  [ ( e
    , getName e
    , map _targetName $ filter isLocalFullyQualifiedName (dependencies e)
    )
    | e <- elems
  ]
  where
    isLocalFullyQualifiedName :: FullyQualifiedName -> Bool
    isLocalFullyQualifiedName = maybe True (== pkgName) . _scopeName

-- | Throws an error if a strongly connected component is cyclic. Otherwise,
-- returns the vertex contained in the acyclic component.
checkNoCycle :: (HasSrcCtx a, HasName a) => G.SCC a -> MgMonad a
checkNoCycle (G.CyclicSCC vertices) =
  let cyclicErr = T.intercalate ", " $ map (pshow . getName) vertices in
  throwLocatedE CyclicErr (srcCtx (head vertices)) cyclicErr
checkNoCycle (G.AcyclicSCC vertex) = return vertex

-- | Checks whether the 'TopLevelDeclOrigin' passed as a parameter refers
-- to an element in the current parent package.
isLocalTopLevelDeclOrigin :: TopLevelDeclOrigin -> MgMonad Bool
isLocalTopLevelDeclOrigin (TopLevelDeclOrigin fqn _) =
  (== _scopeName fqn) . Just <$> getParentPackageName

-- === renamings manipulation ===

renamingBlockToInlineRenamings :: TcRenamingBlock -> [InlineRenaming]
renamingBlockToInlineRenamings (Ann _ (MRenamingBlock _ renamings)) =
    map mkInlineRenaming renamings
  where mkInlineRenaming (Ann _ r) = case r of InlineRenaming ir -> ir
                                               RefRenaming v -> absurd v

-- === expression manipulation ===

isValueExpr :: MExpr p -> Bool
isValueExpr (Ann _ e) = case e of
  MValue _ -> True
  MIf _ eTrue eFalse -> isValueExpr eTrue || isValueExpr eFalse
  _ -> False