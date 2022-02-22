{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Magnolia.Util (
  -- * Package name manipulation utils
    mkPkgNameFromPath
  , mkPkgPathFromName
  , mkPkgPathFromStr
  , isPkgPath
  -- * Dependencies-related utils
  , checkNoCycle
  -- * Top-level manipulation utils
  , lookupTopLevelRef
  , topSortTopLevelE
  -- * Renaming manipulation utils
  , renamingBlockToInlineRenamings
  -- * Declaration manipulation utils
  , isLocalDecl
  -- * Expression manipulations utils
  , isValueExpr
  -- * Generic utils
  , mapAccumM
  )
  where

import Control.Monad
import Control.Monad.Except (lift)
import qualified Control.Monad.Trans.State as ST
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M
import Data.Tuple (swap)

import Prettyprinter (Pretty)
import qualified Data.Text.Lazy as T
import Data.Void (absurd)

import Env
import Err
import Monad
import Magnolia.PPrint
import Magnolia.Syntax

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
     , HasName (e PhCheck)
     )
  => SrcCtx -> Env [Tc e] -> FullyQualifiedName -> MgMonad (Tc e)
lookupTopLevelRef src env ref = case M.lookup (_targetName ref) env of
  Nothing -> throwLocatedE UnboundTopLevelErr src $ pshow ref
  Just [] -> throwLocatedE CompilerErr src $
    "name " <> pshow (_targetName ref) <> " exists but is not bound to anything"
  Just matches -> do
    let localDecls = filter isLocalDecl matches
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
        else let matchesImportScope match = case _ann match of
                    ImportedDecl fqn _ -> Just scopeName == _scopeName fqn
                    _ -> False
             in return $ filter matchesImportScope matches
    -- The reference is only valid when a match exists and is unambiguous,
    -- i.e. in cases when compatibleMatches contains exactly one element.
    when (null compatibleMatches) $ -- no match
      throwLocatedE UnboundTopLevelErr src $ pshow ref
    parentPkgName <- getParentPackageName
    let toFQN n declO = case declO of
          LocalDecl {} -> FullyQualifiedName (Just parentPkgName) n
          ImportedDecl fqn _ -> fqn
    unless (null $ tail compatibleMatches) $ -- more than one match
      throwLocatedE AmbiguousTopLevelRefErr src $ pshow ref <>
        "'. Candidates are: " <> T.intercalate ", " (
          map (\m -> pshow $ toFQN (nodeName m) (_ann m)) compatibleMatches)
    return $ head compatibleMatches

-- | Topologically sorts top-level named elements (i.e. named renamings
-- declarations, modules) based on their local dependencies.
topSortTopLevelE :: (HasDependencies a, HasName a) => Name -> [a] -> [G.SCC a]
topSortTopLevelE pkgName elems = G.stronglyConnComp
  [ ( e
    , nodeName e
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
  let cyclicErr = T.intercalate ", " $ map (pshow . nodeName) vertices in
  throwLocatedE CyclicErr (srcCtx (head vertices)) cyclicErr
checkNoCycle (G.AcyclicSCC vertex) = return vertex

-- === renamings manipulation ===

renamingBlockToInlineRenamings :: TcRenamingBlock -> [InlineRenaming]
renamingBlockToInlineRenamings (Ann _ (MRenamingBlock _ renamings)) =
    map mkInlineRenaming renamings
  where mkInlineRenaming (Ann _ r) = case r of InlineRenaming ir -> ir
                                               RefRenaming v -> absurd v

-- === declaration manipulation ===

isLocalDecl :: XAnn p e ~ DeclOrigin => Ann p e -> Bool
isLocalDecl d = case _ann d of LocalDecl {} -> True ; _ -> False

-- === expression manipulation ===

isValueExpr :: MExpr p -> Bool
isValueExpr (Ann _ e) = case e of
  MValue _ -> True
  MIf _ eTrue eFalse -> isValueExpr eTrue || isValueExpr eFalse
  _ -> False

-- === generic utils ===

mapAccumM :: (Traversable t, Monad m)
          => (a -> b -> m (a, c)) -> a -> t b -> m (a, t c)
mapAccumM f a tb = swap <$> mapM go tb `ST.runStateT` a
  where go b = do s <- ST.get
                  (s', r) <- lift $ f s b
                  ST.put s'
                  return r