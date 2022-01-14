{-# LANGUAGE OverloadedStrings #-}

module MgToUtil (
    mapMAccumErrors
  , mapMAccumErrorsAndFail
    -- * Requirement-related utils
  , Requirement (..)
  , gatherUniqueExternalRequirements
  , orderRequirements
  , orderedRequirements
    -- * Name-related utils
  , BoundNames
  , freshFunctionClassNameM
  , freshName
  , freshNameM
  , freshObjectNameM
    -- * Misc utils
  , checkCxxBackend
  , checkPyBackend
  )
  where

import Control.Monad.State
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import Data.Void (absurd)
import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Monad

-- | Not really a map, but a wrapper over 'foldMAccumErrors' that allows writing
-- code like
--
-- >>> mapM f t
--
-- on lists that accumulates errors within the 'MgMonad' and returns a list of
-- items.
mapMAccumErrors :: (a -> MgMonad b) -> [a] -> MgMonad [b]
mapMAccumErrors f = foldMAccumErrors (\acc a -> (acc <>) . (:[]) <$> f a) []

-- | Like 'mapMAccumErrors' but fails if any error was thrown by the time the
-- end of the list is reached.
mapMAccumErrorsAndFail :: (a -> MgMonad b) -> [a] -> MgMonad [b]
mapMAccumErrorsAndFail f = foldMAccumErrorsAndFail
  (\acc a -> (acc <>) . (:[]) <$> f a) []

-- === requirement-related utils ===

-- | A helper data type to wrap the requirements found in 'ExternalDeclDetails'.
data Requirement = Requirement { -- | The original required declaration
                                 _requiredDecl :: TcDecl
                                 -- | The declaration supposed to fullfill
                                 --   the requirement.
                               , _parameterDecl :: TcDecl
                               }
                   deriving (Eq, Ord)

-- | Defines an ordering on 'Requirement'. The convention is that type
-- declarations are lower than callable declarations, and elements of the
-- same type are ordered lexicographically on their names.
orderRequirements :: Requirement -> Requirement -> Ordering
orderRequirements (Requirement (MTypeDecl _ tcTy1) _)
                  (Requirement tcDecl2 _) = case tcDecl2 of
  MTypeDecl _ tcTy2 -> nodeName tcTy1 `compare` nodeName tcTy2
  MCallableDecl _ _ -> LT
orderRequirements (Requirement (MCallableDecl _ tcCallable1) _)
                  (Requirement tcDecl2 _) = case tcDecl2 of
  MTypeDecl _ _ -> GT
  MCallableDecl _ tcCallable2 ->
    nodeName tcCallable1 `compare` nodeName tcCallable2

-- | Produces a list of ordered requirements from an 'ExternalDeclDetails'.
-- See 'orderRequirements' for details on the ordering.
orderedRequirements :: ExternalDeclDetails -> [Requirement]
orderedRequirements extDeclDetails = L.sortBy orderRequirements $
  map (uncurry Requirement) $ M.toList $ externalDeclRequirements extDeclDetails

-- | Gathers the external requirements of a module expression as a list. The
-- resulting external requirements are represented as a pair of elements, the
-- first or which is the name of the external module, and the second one is
-- a list of arguments it requires. The elements in the resulting list are all
-- distinct.
gatherUniqueExternalRequirements :: TcModuleExpr -> [(Name, [Requirement])]
gatherUniqueExternalRequirements (Ann _ tcModuleExpr')
  | MModuleRef v _ <- tcModuleExpr' = absurd v
  | MModuleAsSignature v _ <- tcModuleExpr' = absurd v
  | MModuleExternal _ _ v <- tcModuleExpr' = absurd v
  | MModuleDef decls _ _ <- tcModuleExpr' =
    S.toList $ foldl accExtReqs S.empty (join $ M.elems decls)
  where
    accExtReqs :: S.Set (Name, [Requirement]) -> TcDecl
               -> S.Set (Name, [Requirement])
    accExtReqs acc (MTypeDecl _ (Ann (mconDeclO, _) _)) = case mconDeclO of
      Just (ConcreteExternalDecl _ extDeclDetails) ->
        S.insert ( externalDeclModuleName extDeclDetails
                 , orderedRequirements extDeclDetails
                 ) acc
      _ -> acc
    accExtReqs acc (MCallableDecl _ (Ann (mconDeclO, _) _)) = case mconDeclO of
      Just (ConcreteExternalDecl _ extDeclDetails) ->
        S.insert ( externalDeclModuleName extDeclDetails
                 , orderedRequirements extDeclDetails
                 ) acc
      _ -> acc

-- === name-related utils ===

type BoundNames = S.Set String

-- | Produces a fresh name based on an initial input name and a set of bound
-- strings. If the input name's String component is not in the set of bound
-- strings, it is returned as is.
freshName :: Name -> BoundNames -> Name
freshName name@(Name ns str) boundNs
  | str `S.member` boundNs = freshName' (0 :: Int)
  | otherwise = name
  where
    freshName' i | (str <> show i) `S.member` boundNs = freshName' (i + 1)
                 | otherwise = Name ns $ str <> show i

-- | Like 'freshName' but using a State monad.
freshNameM :: Monad m => Name -> StateT BoundNames m Name
freshNameM n = do
  env <- get
  let freeName = freshName n env
  modify (S.insert $ _name freeName)
  pure freeName

-- | Like 'freshNameM', following the naming convention for objects.
freshObjectNameM :: Monad m => Name -> StateT BoundNames m Name
freshObjectNameM n = freshNameM n { _name = "__" <> _name n }

-- | Like 'freshNameM', following the naming convention for function classes.
freshFunctionClassNameM :: Monad m => Name -> StateT BoundNames m Name
freshFunctionClassNameM n = freshNameM n { _name = "_" <> _name n }

-- === misc utils ===

-- | Checks whether some 'ExternalDeclDetails' correspond to a declaration with
-- the specificed backend. Throws an error if it is not the case.
checkBackend :: Backend
             -> SrcCtx -- ^ source information for error reporting
             -> T.Text -- ^ the type of declaration we check; this is
                       --   used verbatim in the error message
             -> ExternalDeclDetails
             -> MgMonad ()
checkBackend expectedBackend src prettyDeclTy extDeclDetails =
  let backend = externalDeclBackend extDeclDetails
      extStructName = externalDeclModuleName extDeclDetails
      extTyName = externalDeclElementName extDeclDetails
  in unless (backend == expectedBackend) $ throwLocatedE MiscErr src $
      "attempted to generate " <> pshow expectedBackend <> " code relying " <>
      "on external " <> prettyDeclTy <> " " <>
      pshow (FullyQualifiedName (Just extStructName) extTyName) <>
      " but it is declared as having a " <> pshow backend <> " backend"

-- | See 'checkBackend'.
checkCxxBackend :: SrcCtx -> T.Text -> ExternalDeclDetails -> MgMonad ()
checkCxxBackend = checkBackend Cxx

-- | See 'checkBackend'.
checkPyBackend :: SrcCtx -> T.Text -> ExternalDeclDetails -> MgMonad ()
checkPyBackend = checkBackend Python