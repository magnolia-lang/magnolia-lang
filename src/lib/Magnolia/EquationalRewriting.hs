{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module Magnolia.EquationalRewriting (
    runOptimizer
  )
  where

import Control.Monad (foldM, join)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Except as E
import Control.Monad.State
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isNothing)
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import Data.Void (absurd)

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import Monad
import Python.Syntax (extractParentObjectName)
import Data.Foldable (traverse_)

-- TODO: define optimizer modes in the compiler

type Optimizer = TcModule

-- TODO: should we only optimize Programs here?

-- == optimizer IR ==

data GenericExprIR a = ExprIR'Var VarIR
                     | ExprIR'Call Name [a]
                     | ExprIR'ValueBlock (NE.NonEmpty a)
                     | ExprIR'EffectfulBlock (NE.NonEmpty a)
                     | ExprIR'Value a
                     | ExprIR'Let VarIR (Maybe a)
                     | ExprIR'If (GenericPredicateIR a) a a
                     | ExprIR'Assert (GenericPredicateIR a)
                     | ExprIR'Skip
                       deriving (Eq, Ord, Show)

data VarIR = VarIR MVarMode Name MType
             deriving (Eq, Ord, Show)

data GenericPredicateIR a =
    PredicateIR'And (GenericPredicateIR a) (GenericPredicateIR a)
  | PredicateIR'Or (GenericPredicateIR a) (GenericPredicateIR a)
  | PredicateIR'Implies (GenericPredicateIR a) (GenericPredicateIR a)
  | PredicateIR'Equivalent (GenericPredicateIR a) (GenericPredicateIR a)
  | PredicateIR'PredicateEquation (GenericPredicateIR a) (GenericPredicateIR a)
  | PredicateIR'Negation (GenericPredicateIR a)
  | PredicateIR'Atom (GenericPredicateIRAtom a)
    deriving (Eq, Ord, Show)

-- | 'GenericPredicateIRAtom's represent the subset of Magnolia expressions
-- corresponding to atomic predicates, that is to say equations, or calls to
-- custom predicates.
data GenericPredicateIRAtom a = PredicateIRAtom'Call Name [a]
                              | PredicateIRAtom'ExprEquation a a
                                deriving (Eq, Ord, Show)

-- | An IR to represent expressions during this equational rewriting phase. The
-- representation is introduced to manipulate predicates with more ease.
-- TODO: fix up TcExpr so that it is the opposite: predicate combinators will
-- not be represented as functions, unless passing through a STF/possibly code
-- generation. This should simplify some things.
-- TODO: round-tripping is not the identity right now, since the Checked AST
-- makes less assumptions than is possible. It would be better to round-trip
-- if possible, but again, we should modify the core checked AST for that.
data TypedExprIR = TypedExprIR MType ExprIR
                   deriving (Eq, Ord, Show)
type ExprIR = GenericExprIR TypedExprIR
type PredicateIR = GenericPredicateIR TypedExprIR
type PredicateIRAtom = GenericPredicateIRAtom TypedExprIR

data Equation = Equation { eqnVariables :: S.Set Name
                         , eqnSourceExpr :: TypedPattern
                         , eqnTargetExpr :: TypedPattern
                         }
                deriving (Eq, Show)

-- | Converts a 'TcExpr' into its equivalent 'TypedExprIR' representation.
toTypedExprIR :: TcExpr -> MgMonad TypedExprIR
toTypedExprIR (Ann inTy inTcExpr) = TypedExprIR inTy <$> go inTcExpr
  where
    go :: MExpr' PhCheck -> MgMonad ExprIR
    go tcExpr = case tcExpr of
      MVar (Ann ty (Var mode name _)) -> pure $ ExprIR'Var (VarIR mode name ty)
      -- TODO: this call could be a predicate, but we don't care (?)
      MCall name args _ -> ExprIR'Call name <$> mapM toTypedExprIR args
      MBlockExpr blockTy stmts -> do
        stmtsIR <- mapM toTypedExprIR stmts
        pure $ case blockTy of MValueBlock -> ExprIR'ValueBlock stmtsIR
                               MEffectfulBlock -> ExprIR'EffectfulBlock stmtsIR
      MValue expr -> ExprIR'Value <$> toTypedExprIR expr
      MLet (Ann ty (Var mode name _)) mexpr -> do
        let varIR = VarIR mode name ty
        mexprIR <- maybe (pure Nothing) ((Just <$>) . toTypedExprIR) mexpr
        pure $ ExprIR'Let varIR mexprIR
      MIf condExpr trueExpr falseExpr ->
        ExprIR'If <$> toPredicateIR condExpr <*> toTypedExprIR trueExpr
                                             <*> toTypedExprIR falseExpr
      MAssert expr -> ExprIR'Assert <$> toPredicateIR expr
      MSkip -> pure ExprIR'Skip

    toPredicateIR :: TcExpr -> MgMonad PredicateIR
    toPredicateIR (Ann ty _) | ty /= Pred =
      unexpected "non-predicate expression"
    toPredicateIR (Ann _ tcExpr) = case tcExpr of
      MVar {} -> unimplemented "variables"
      MCall name args _ -> case (name, map exprTy args) of
        (FuncName "_&&_", [Pred, Pred]) -> predicateCombinator
          PredicateIR'And args
        (FuncName "_||_", [Pred, Pred]) -> predicateCombinator
          PredicateIR'Or args
        (FuncName "_=>_", [Pred, Pred]) -> predicateCombinator
          PredicateIR'Implies args
        (FuncName "_<=>_", [Pred, Pred]) -> predicateCombinator
          PredicateIR'Equivalent args
        (FuncName "_==_", [Pred, Pred]) -> predicateCombinator
          PredicateIR'PredicateEquation args
        (FuncName "_!=_", [Pred, Pred]) -> predicateCombinator
          ((PredicateIR'Negation .) . PredicateIR'PredicateEquation)
          args
        (FuncName op, [lhsTy, rhsTy]) -> do
          ~[lhsIR, rhsIR] <- mapM toTypedExprIR args
          pure $
            if lhsTy == rhsTy
            then case op of
              "_==_" -> PredicateIR'Atom $
                PredicateIRAtom'ExprEquation lhsIR rhsIR
              "_!=_" -> PredicateIR'Negation
                (PredicateIR'Atom (PredicateIRAtom'ExprEquation lhsIR rhsIR))
              _ -> PredicateIR'Atom $ PredicateIRAtom'Call name [lhsIR, rhsIR]
            else PredicateIR'Atom $ PredicateIRAtom'Call name [lhsIR, rhsIR]
        _ -> do
          argsIR <- mapM toTypedExprIR args
          pure $ PredicateIR'Atom (PredicateIRAtom'Call name argsIR)
      MBlockExpr {} -> unimplemented "block expressions"
      MValue tcExpr' -> toPredicateIR tcExpr'
      MLet {} -> unexpected "let binding"
      -- TODO: implementation should not be too hard, just make one predicateIR
      -- for each branch, but then we need to return several PredicateIRs here
      -- :)
      MIf {} -> unimplemented "if expression"
      MAssert {} -> unexpected "assertion"
      MSkip -> unexpected "skip"

    predicateCombinator constructor ~[lhsPredicate, rhsPredicate] =
      constructor <$> toPredicateIR lhsPredicate <*> toPredicateIR rhsPredicate

    unexpected unexpectedSourceInfo = throwNonLocatedE CompilerErr $
      "unexpected " <> unexpectedSourceInfo <> " when building PredicateIR"
    unimplemented exprKind = throwNonLocatedE NotImplementedErr $
      "building PredicateIR from " <> exprKind <> " is not implemented"

-- | Converts a 'TypedExprIR' into an equivalent 'TcExpr' representation.
-- Note that converting back and forth does not necessarily result in a
-- roundtrip – e.g. expressions such as a != b would get transformed into
-- !(a == b).
fromTypedExprIR :: TypedExprIR -> TcExpr
fromTypedExprIR (TypedExprIR tyExprIR exprIR) = Ann tyExprIR $ case exprIR of
  ExprIR'Var (VarIR mode name ty) -> MVar (Ann ty (Var mode name (Just ty)))
  ExprIR'Call name args -> MCall name (map fromTypedExprIR args) (Just tyExprIR)
  ExprIR'ValueBlock stmts ->
    MBlockExpr MValueBlock (NE.map fromTypedExprIR stmts)
  ExprIR'EffectfulBlock stmts ->
    MBlockExpr MEffectfulBlock (NE.map fromTypedExprIR stmts)
  ExprIR'Value typedExprIR -> MValue (fromTypedExprIR typedExprIR)
  ExprIR'Let (VarIR mode name ty) mtypedExprIR ->
    MLet (Ann ty (Var mode name (Just ty))) (fromTypedExprIR <$> mtypedExprIR)
  ExprIR'If condExprIR trueExprIR falseExprIR ->
    MIf (fromPredicateIR condExprIR) (fromTypedExprIR trueExprIR)
                                     (fromTypedExprIR falseExprIR)
  ExprIR'Assert assertExpr -> MAssert $ fromPredicateIR assertExpr
  ExprIR'Skip -> MSkip
  where
    fromPredicateIR :: PredicateIR -> TcExpr
    fromPredicateIR predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> predicateCombinator "&&" lhs rhs
      PredicateIR'Or lhs rhs -> predicateCombinator "||" lhs rhs
      PredicateIR'Implies lhs rhs -> predicateCombinator "=>" lhs rhs
      PredicateIR'Equivalent lhs rhs -> predicateCombinator "<=>" lhs rhs
      PredicateIR'Negation predicateIR' -> Ann Pred $
        MCall (FuncName "!_") [fromPredicateIR predicateIR'] (Just Pred)
      PredicateIR'PredicateEquation lhs rhs -> predicateCombinator "==" lhs rhs
      PredicateIR'Atom atom -> fromPredicateIRAtom atom

    fromPredicateIRAtom :: PredicateIRAtom -> TcExpr
    fromPredicateIRAtom predicateIRAtom = Ann Pred $ case predicateIRAtom of
      PredicateIRAtom'Call name args ->
        MCall name (map fromTypedExprIR args) (Just Pred)
      PredicateIRAtom'ExprEquation lhs rhs ->
        MCall (FuncName "_==_") [fromTypedExprIR lhs, fromTypedExprIR rhs]
              (Just Pred)

    predicateCombinator combinator lhs rhs = Ann Pred $
      MCall (FuncName ("_" <> combinator <> "_"))
            [fromPredicateIR lhs, fromPredicateIR rhs] (Just Pred)

-- == equational rewriting ==

data TypedPattern = TypedPattern MType Pattern
                    deriving (Eq, Show)
data Pattern = Pattern'Wildcard Name
             | Pattern'Expr (GenericExprIR TypedPattern)
               deriving (Eq, Show)
type PredicatePattern = GenericPredicateIR TypedPattern
type PredicateAtomPattern = GenericPredicateIRAtom TypedPattern

-- | Checks whether a typed expression matches a typed expression pattern.
-- Returns a map from the universally quantified variable names in the
-- expression pattern (or wildcards) to the concrete expressions corresponding
-- to them in the typed expression – if the operation succeeds. Returns Nothing
-- otherwise.
matchesPattern :: TypedExprIR -> TypedPattern
               -> Maybe (M.Map Name TypedExprIR)
matchesPattern (TypedExprIR tyExprIR _) (TypedPattern tyPattern _)
  | tyExprIR /= tyPattern = Nothing
matchesPattern inTypedExprIR inPattern =
  case runState (E.runExceptT (matchesPattern' inTypedExprIR inPattern))
                M.empty of
    (Left _, _) -> Nothing
    (Right _, matchMap) -> Just matchMap
  where
    matchesPattern' :: TypedExprIR
                    -> TypedPattern
                    -> ExceptT () (State (M.Map Name TypedExprIR)) ()
    matchesPattern' (TypedExprIR tyExprIR _) (TypedPattern tyPattern _)
      | tyExprIR /= tyPattern = patternDoesNotMatch
    matchesPattern' typedExprIR (TypedPattern _ pattern) = case pattern of
      Pattern'Wildcard holeName -> bind holeName typedExprIR
      Pattern'Expr patExprIR -> do
        let TypedExprIR _ exprIR = typedExprIR
        case (patExprIR, exprIR) of
          -- TODO: exception? we do not expect Vars in pattern
          --(ExprIR'Var patVarIR, ExprIR'Var varIR) -> assert $ patVarIR == varIR
          (ExprIR'Call patName patArgs, ExprIR'Call exprName exprArgs) -> do
            assert $ patName == exprName
            assert $ length patArgs == length exprArgs
            traverse_ (uncurry matchesPattern') (zip exprArgs patArgs)
          (ExprIR'ValueBlock patStmts, ExprIR'ValueBlock exprStmts) -> do
             assert $ length patStmts == length exprStmts
             traverse_ (uncurry matchesPattern')
                       (zip (NE.toList exprStmts) (NE.toList patStmts))
          (ExprIR'EffectfulBlock patStmts,
           ExprIR'EffectfulBlock exprStmts) -> do
             assert $ length patStmts == length exprStmts
             traverse_ (uncurry matchesPattern')
                       (zip (NE.toList exprStmts) (NE.toList patStmts))
          (ExprIR'Value pat, ExprIR'Value valueExprIR) ->
            matchesPattern' valueExprIR pat
          --(ExprIR'Let patVarIR mpat, ExprIR'Let varIR exprIR) -> undefined
          (ExprIR'If patCond patTrue patFalse,
           ExprIR'If condExprIR trueExprIR falseExprIR) -> do
            matchesPredPattern condExprIR patCond
            matchesPattern' trueExprIR patTrue
            matchesPattern' falseExprIR patFalse
          (ExprIR'Assert patPredicateIR, ExprIR'Assert predicateIR) ->
            matchesPredPattern predicateIR patPredicateIR
          (ExprIR'Skip, ExprIR'Skip) -> patternMatches
          _ -> patternDoesNotMatch

    matchesPredPattern :: PredicateIR -> PredicatePattern
                       -> ExceptT () (State (M.Map Name TypedExprIR)) ()
    matchesPredPattern predicateIR predicatePattern =
      case (predicatePattern, predicateIR) of
        (PredicateIR'And patLhs patRhs, PredicateIR'And exprLhs exprRhs) -> do
          matchesPredPattern exprLhs patLhs
          matchesPredPattern exprRhs patRhs
        (PredicateIR'Or patLhs patRhs, PredicateIR'Or exprLhs exprRhs) -> do
          matchesPredPattern exprLhs patLhs
          matchesPredPattern exprRhs patRhs
        (PredicateIR'Implies patLhs patRhs,
         PredicateIR'Implies exprLhs exprRhs) -> do
          matchesPredPattern exprLhs patLhs
          matchesPredPattern exprRhs patRhs
        (PredicateIR'Equivalent patLhs patRhs,
         PredicateIR'Equivalent exprLhs exprRhs) -> do
          matchesPredPattern exprLhs patLhs
          matchesPredPattern exprRhs patRhs
        (PredicateIR'PredicateEquation patLhs patRhs,
         PredicateIR'PredicateEquation exprLhs exprRhs) -> do
          matchesPredPattern exprLhs patLhs
          matchesPredPattern exprRhs patRhs
        (PredicateIR'Negation predicatePattern',
         PredicateIR'Negation predicateIR') ->
           matchesPredPattern predicateIR' predicatePattern'
        (PredicateIR'Atom predicateAtomPattern,
         PredicateIR'Atom predicateIRAtom) ->
           matchesPredAtomPattern predicateIRAtom predicateAtomPattern
        _ -> patternDoesNotMatch

    matchesPredAtomPattern :: PredicateIRAtom -> PredicateAtomPattern
                           -> ExceptT () (State (M.Map Name TypedExprIR)) ()
    matchesPredAtomPattern predicateIRAtom predicateAtomPattern =
      case (predicateAtomPattern, predicateIRAtom) of
        (PredicateIRAtom'Call patName patArgs,
         PredicateIRAtom'Call exprName exprArgs) -> do
           assert $ patName == exprName
           assert $ length patArgs == length exprArgs
           traverse_ (uncurry matchesPattern') (zip exprArgs patArgs)
        (PredicateIRAtom'ExprEquation patLhs patRhs,
         PredicateIRAtom'ExprEquation exprLhs exprRhs) -> do
           matchesPattern' exprLhs patLhs
           matchesPattern' exprRhs patRhs
        _ -> patternDoesNotMatch

    bind :: Name -> TypedExprIR
         -> ExceptT () (State (M.Map Name TypedExprIR)) ()
    bind holeName typedExprIR = do
      bindings <- lift get
      case M.lookup holeName bindings of
        Nothing -> lift $ put $ M.insert holeName typedExprIR bindings
        Just typedExprIR' -> if typedExprIR == typedExprIR'
                             then patternMatches
                             else patternDoesNotMatch

    assert cond = if cond then pure () else throwE ()
    patternMatches = pure ()
    patternDoesNotMatch = throwE ()

-- | Transforms a typed expression into a typed pattern.
toPattern :: TypedExprIR -> TypedPattern
toPattern typedExprIR =
  toPattern' typedExprIR `evalState` (M.empty, 0)
  where
    toPattern' :: TypedExprIR -> State (M.Map Name Name, Int) TypedPattern
    toPattern' (TypedExprIR inTyExprIR inExprIR) =
      TypedPattern inTyExprIR <$> (case inExprIR of
        ExprIR'Var (VarIR _ name _) -> toWildcard name
        ExprIR'Call name args -> pe . ExprIR'Call name <$> mapM toPattern' args
        ExprIR'ValueBlock stmts -> pe . ExprIR'ValueBlock <$>
          mapM toPattern' stmts
        ExprIR'EffectfulBlock stmts -> pe . ExprIR'EffectfulBlock <$>
          mapM toPattern' stmts
        ExprIR'Value exprIR -> pe . ExprIR'Value <$> toPattern' exprIR
        ExprIR'Let (VarIR mode name ty) mexprIR ->
          error "let stmts in toPattern are not implemented"
        ExprIR'If condIR trueExprIR falseExprIR -> do
          pe <$> (ExprIR'If <$> toPredicatePattern condIR
                            <*> toPattern' trueExprIR
                            <*> toPattern' falseExprIR)
        ExprIR'Assert predicateIR ->
          pe . ExprIR'Assert <$> toPredicatePattern predicateIR
        ExprIR'Skip -> pure $ pe ExprIR'Skip)

    pe = Pattern'Expr

    toPredicatePattern :: PredicateIR
                       -> State (M.Map Name Name, Int) PredicatePattern
    toPredicatePattern predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> predicateCombinator PredicateIR'And lhs rhs
      PredicateIR'Or lhs rhs -> predicateCombinator PredicateIR'Or lhs rhs
      PredicateIR'Implies lhs rhs ->
        predicateCombinator PredicateIR'Implies lhs rhs
      PredicateIR'Equivalent lhs rhs ->
        predicateCombinator PredicateIR'Equivalent lhs rhs
      PredicateIR'PredicateEquation lhs rhs ->
        predicateCombinator PredicateIR'PredicateEquation lhs rhs
      PredicateIR'Negation predicateIR' -> PredicateIR'Negation <$>
        toPredicatePattern predicateIR'
      PredicateIR'Atom predicateIRAtom -> PredicateIR'Atom <$>
        toPredicateAtomPattern predicateIRAtom

    toPredicateAtomPattern :: PredicateIRAtom
                           -> State (M.Map Name Name, Int) PredicateAtomPattern
    toPredicateAtomPattern predicateIRAtom = case predicateIRAtom of
      PredicateIRAtom'Call name args -> PredicateIRAtom'Call name <$>
        mapM toPattern' args
      PredicateIRAtom'ExprEquation lhs rhs -> PredicateIRAtom'ExprEquation <$>
        toPattern' lhs <*> toPattern' rhs

    predicateCombinator constructor lhsPredicate rhsPredicate =
      constructor <$> toPredicatePattern lhsPredicate
                  <*> toPredicatePattern rhsPredicate

    toWildcard :: Name -> State (M.Map Name Name, Int) Pattern
    toWildcard name = do
      (bindings, nextId) <- get
      case M.lookup name bindings of
        Just name' -> pure $ Pattern'Wildcard name'
        Nothing -> do
          let name' = GenName ("?" <> show nextId)
              newBindings = M.insert name name' bindings
          put (newBindings, nextId + 1)
          pure $ Pattern'Wildcard name'

-- | We represent a condition as the intersection of a set of predicate
-- expressions.
type Condition = S.Set PredicateIRAtom

-- TODO: how to deal with variables in predicates?
data Constraint = Constraint'Equational Equation
                | Constraint'ConditionalEquational Condition Equation
                  deriving (Eq, Show)

constraintEquation :: Constraint -> Equation
constraintEquation constraint = case constraint of
  Constraint'Equational equation -> equation
  Constraint'ConditionalEquational _ equation -> equation

type FactDB = S.Set PredicateIRAtom
type ConstraintDB = M.Map TypedExprIR Constraint

-- | Runs an optimizer one time over one module.
runOptimizer :: Optimizer -> TcModule -> MgMonad TcModule
runOptimizer (Ann _ optimizer) (Ann tgtDeclO tgtModule) = case optimizer of
  MModule Concept optimizerName optimizerModuleExpr -> enter optimizerName $ do
    -- 1. gather axioms
    axioms <- gatherAxioms optimizerModuleExpr
    -- 2. inline expressions within axioms (TODO)
    inlinedAxioms <- mapM (\(AnonymousAxiom v typedExprIR) ->
      AnonymousAxiom v <$> inlineTypedExprIR typedExprIR) axioms
    -- 3. gather directed rewrite rules
    constraintDb <- join <$> mapM gatherConstraints inlinedAxioms
    -- 4. build some kind of assertion scope
    let constraintDb = M.fromList $ map
          (\c -> (fst $ canonicalize (eqnSourceExpr (constraintEquation c)), c))
          constraints
    liftIO $ mapM_ (pprint . fromTypedExprIR) $ M.keys constraintDb
    -- 5. traverse each expression in the target module to rewrite it (while
    -- elaborating the scope)
    let ~(MModule tgtModuleTy tgtModuleName (Ann src tgtModuleExpr)) = tgtModule
    resultModuleExpr <- Ann src <$> case tgtModuleExpr of
      MModuleDef decls deps renamings -> do
        decls' <- mapM (traverse $ rewriteDecl constraintDb) decls
        pure $ MModuleDef decls' deps renamings
      MModuleRef v _ -> absurd v
      MModuleAsSignature v _ -> absurd v
      MModuleExternal {} -> pure tgtModuleExpr

    liftIO $ pprint resultModuleExpr
    -- 6: wrap it up
    -- comment out to avoid IDE suggestion
    pure $ Ann tgtDeclO (MModule tgtModuleTy tgtModuleName resultModuleExpr)
  _ -> undefined


-- | Performs equational rewriting on a Magnolia declaration.
rewriteDecl :: ConstraintDB -> TcDecl -> MgMonad TcDecl
rewriteDecl constraintDb tcDecl = case tcDecl of
  MTypeDecl {} -> pure tcDecl
  MCallableDecl mods (Ann callableAnn callableDecl) ->
    case _callableBody callableDecl of
      MagnoliaBody tcExpr -> do
        tcExprIR <- toTypedExprIR tcExpr
        liftIO $ pprint $ "rewriting " <> pshow (nodeName tcDecl) <> " now"
        tcExprIR' <- oneRewritePass constraintDb tcExprIR
        let tcExpr' = fromTypedExprIR tcExprIR'
            callableDecl' = callableDecl {_callableBody = MagnoliaBody tcExpr'}
        pure (MCallableDecl mods (Ann callableAnn callableDecl'))
      _ -> pure tcDecl

oneRewritePass :: ConstraintDB -> TypedExprIR -> MgMonad TypedExprIR
oneRewritePass constraintDb = oneRewritePass' S.empty
  where
    oneRewritePass' :: FactDB -> TypedExprIR -> MgMonad TypedExprIR
    oneRewritePass' factDb typedExprIR = do
      liftIO $ pprint $ fromTypedExprIR $ fst (canonicalize typedExprIR)
      case M.lookup (fst $ canonicalize typedExprIR) constraintDb of
        Nothing -> oneRewritePassExprIR factDb typedExprIR
        Just constraint -> case constraint of
          Constraint'Equational eqn -> do
            liftIO $ pprint $ "firing rule " <> show eqn
            rewrite eqn typedExprIR
          Constraint'ConditionalEquational {} ->
            throwNonLocatedE NotImplementedErr
              "conditional equational rewriting"

    oneRewritePassExprIR :: FactDB -> TypedExprIR -> MgMonad TypedExprIR
    oneRewritePassExprIR factDb (TypedExprIR tyExprIR exprIR) =
      TypedExprIR tyExprIR <$> (case exprIR of
        ExprIR'Var varIR -> pure $ ExprIR'Var varIR
        ExprIR'Call name args -> ExprIR'Call name <$>
          mapM (oneRewritePassExprIR factDb) args
        ExprIR'ValueBlock ne -> undefined
        ExprIR'EffectfulBlock ne -> undefined
        ExprIR'Value x0 -> undefined
        ExprIR'Let vi ma -> undefined
        ExprIR'If pi x0 x1 -> undefined
        ExprIR'Assert pi -> undefined
        ExprIR'Skip -> undefined)

-- | Performs an equational rewrite on the expression passed as a paramter.
-- Note that if the source equation does not match the expression, an exception
-- is raised.
rewrite :: Equation -> TypedExprIR -> MgMonad TypedExprIR
rewrite eqn inTypedExprIR = do
  let (canonicalEqnSourceExpr, nameMapEqn) = canonicalize $ eqnSourceExpr eqn
      (canonicalInExpr, nameMapInExpr)  = canonicalize inTypedExprIR
  if canonicalEqnSourceExpr == canonicalInExpr
  then do let renamingMap = M.map (\canonicalName ->
                fromJust $ M.lookup canonicalName nameMapInExpr) nameMapEqn
          alphaRenameTypedExprIR renamingMap (eqnTargetExpr eqn)
  else throwNonLocatedE CompilerErr
    "rewrite should only be called with an equation matching the expression"
  where fromJust ~(Just v) = v

alphaRenameTypedExprIR :: M.Map Name Name -> TypedExprIR -> MgMonad TypedExprIR
alphaRenameTypedExprIR nameMap = alphaRename'
  where
    alphaRename' :: TypedExprIR -> MgMonad TypedExprIR
    alphaRename' (TypedExprIR inTyExprIR inExprIR) =
      TypedExprIR inTyExprIR <$> (case inExprIR of
        ExprIR'Var (VarIR mode name ty) -> do
          name' <- alphaRenameName name
          pure $ ExprIR'Var (VarIR mode name' ty)
        ExprIR'Call name args -> ExprIR'Call name <$> mapM alphaRename' args
        ExprIR'ValueBlock stmts -> ExprIR'ValueBlock <$>
          mapM alphaRename' stmts
        ExprIR'EffectfulBlock stmts -> ExprIR'EffectfulBlock <$>
          mapM alphaRename' stmts
        ExprIR'Value exprIR -> ExprIR'Value <$> alphaRename' exprIR
        ExprIR'Let (VarIR mode name ty) mexprIR -> do
          newVar <- (\n -> VarIR mode n ty) <$> alphaRenameName name
          mexprIR' <- case alphaRename' <$> mexprIR of
            Nothing -> pure Nothing
            Just stateComp -> Just <$> stateComp
          pure $ ExprIR'Let newVar mexprIR'
        ExprIR'If condIR trueExprIR falseExprIR -> do
          ExprIR'If <$> alphaRenamePredicateIR condIR
                    <*> alphaRename' trueExprIR <*> alphaRename' falseExprIR
        ExprIR'Assert predicateIR ->
          ExprIR'Assert <$> alphaRenamePredicateIR predicateIR
        ExprIR'Skip -> pure ExprIR'Skip)

    alphaRenamePredicateIR :: PredicateIR
                           -> MgMonad PredicateIR
    alphaRenamePredicateIR predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> predicateCombinator PredicateIR'And lhs rhs
      PredicateIR'Or lhs rhs -> predicateCombinator PredicateIR'Or lhs rhs
      PredicateIR'Implies lhs rhs ->
        predicateCombinator PredicateIR'Implies lhs rhs
      PredicateIR'Equivalent lhs rhs ->
        predicateCombinator PredicateIR'Equivalent lhs rhs
      PredicateIR'PredicateEquation lhs rhs ->
        predicateCombinator PredicateIR'PredicateEquation lhs rhs
      PredicateIR'Negation predicateIR' -> PredicateIR'Negation <$>
        alphaRenamePredicateIR predicateIR'
      PredicateIR'Atom predicateIRAtom -> PredicateIR'Atom <$>
        alphaRenamePredicateIRAtom predicateIRAtom

    alphaRenamePredicateIRAtom :: PredicateIRAtom
                               -> MgMonad PredicateIRAtom
    alphaRenamePredicateIRAtom predicateIRAtom = case predicateIRAtom of
      PredicateIRAtom'Call name args -> PredicateIRAtom'Call name <$>
        mapM alphaRename' args
      PredicateIRAtom'ExprEquation lhs rhs -> PredicateIRAtom'ExprEquation <$>
        alphaRename' lhs <*> alphaRename' rhs

    predicateCombinator constructor lhsPredicate rhsPredicate =
      constructor <$> alphaRenamePredicateIR lhsPredicate
                  <*> alphaRenamePredicateIR rhsPredicate

    alphaRenameName :: Name -> MgMonad Name
    alphaRenameName name = do
      case M.lookup name nameMap of
        Nothing -> throwNonLocatedE MiscErr $ "no name match for " <>
          pshow name
        Just name' -> pure name'

-- Note: the strategy for identifying if a rule fits a pattern is to have a
-- map which contains for each rewrite rule its
-- see https://www.microsoft.com/en-us/research/uploads/prod/2021/02/hashing-modulo-alpha.pdf

-- | Canonicalize variable names (alpha renaming).
canonicalize :: TypedExprIR -> (TypedExprIR, M.Map Name Name)
canonicalize typedExprIR =
  let (typedExprIR', (nameMap, _)) =
        canonicalize' typedExprIR `runState` (M.empty, 0)
  in (typedExprIR', nameMap)
  where
    canonicalize' :: TypedExprIR -> State (M.Map Name Name, Int) TypedExprIR
    canonicalize' (TypedExprIR inTyExprIR inExprIR) =
      TypedExprIR inTyExprIR <$> (case inExprIR of
        ExprIR'Var (VarIR mode name ty) -> do
          name' <- canonicalizeName name
          pure $ ExprIR'Var (VarIR mode name' ty)
        ExprIR'Call name args -> ExprIR'Call name <$> mapM canonicalize' args
        ExprIR'ValueBlock stmts -> ExprIR'ValueBlock <$>
          mapM canonicalize' stmts
        ExprIR'EffectfulBlock stmts -> ExprIR'EffectfulBlock <$>
          mapM canonicalize' stmts
        ExprIR'Value exprIR -> ExprIR'Value <$> canonicalize' exprIR
        ExprIR'Let (VarIR mode name ty) mexprIR -> do
          newVar <- (\n -> VarIR mode n ty) <$> canonicalizeName name
          mexprIR' <- case canonicalize' <$> mexprIR of
            Nothing -> pure Nothing
            Just stateComp -> Just <$> stateComp
          pure $ ExprIR'Let newVar mexprIR'
        ExprIR'If condIR trueExprIR falseExprIR -> do
          ExprIR'If <$> canonicalizePredicateIR condIR
                    <*> canonicalize' trueExprIR <*> canonicalize' falseExprIR
        ExprIR'Assert predicateIR ->
          ExprIR'Assert <$> canonicalizePredicateIR predicateIR
        ExprIR'Skip -> pure ExprIR'Skip)

    canonicalizePredicateIR :: PredicateIR
                            -> State (M.Map Name Name, Int) PredicateIR
    canonicalizePredicateIR predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> predicateCombinator PredicateIR'And lhs rhs
      PredicateIR'Or lhs rhs -> predicateCombinator PredicateIR'Or lhs rhs
      PredicateIR'Implies lhs rhs ->
        predicateCombinator PredicateIR'Implies lhs rhs
      PredicateIR'Equivalent lhs rhs ->
        predicateCombinator PredicateIR'Equivalent lhs rhs
      PredicateIR'PredicateEquation lhs rhs ->
        predicateCombinator PredicateIR'PredicateEquation lhs rhs
      PredicateIR'Negation predicateIR' -> PredicateIR'Negation <$>
        canonicalizePredicateIR predicateIR'
      PredicateIR'Atom predicateIRAtom -> PredicateIR'Atom <$>
        canonicalizePredicateIRAtom predicateIRAtom

    canonicalizePredicateIRAtom :: PredicateIRAtom
                                -> State (M.Map Name Name, Int) PredicateIRAtom
    canonicalizePredicateIRAtom predicateIRAtom = case predicateIRAtom of
      PredicateIRAtom'Call name args -> PredicateIRAtom'Call name <$>
        mapM canonicalize' args
      PredicateIRAtom'ExprEquation lhs rhs -> PredicateIRAtom'ExprEquation <$>
        canonicalize' lhs <*> canonicalize' rhs

    predicateCombinator constructor lhsPredicate rhsPredicate =
      constructor <$> canonicalizePredicateIR lhsPredicate
                  <*> canonicalizePredicateIR rhsPredicate

    canonicalizeName :: Name -> State (M.Map Name Name, Int) Name
    canonicalizeName name = do
      (bindings, nextId) <- get
      case M.lookup name bindings of
        Just name' -> pure name'
        Nothing -> do
          let name' = GenName ("?" <> show nextId)
              newBindings = M.insert name name' bindings
          put (newBindings, nextId + 1)
          pure name'

data AnonymousAxiom = AnonymousAxiom { axiomVariables :: S.Set Name
                                     , axiomBody :: TypedExprIR
                                     }
                      deriving (Eq, Show)

-- | Extracts a list of constraints from an axiom. Every assertion involving an
-- equation found within the body of the axiom produces a (possibly conditional)
-- constraint. To ensure that this operation works correctly, the parameter's
-- expression must have been inlined beforehand.
gatherConstraints :: AnonymousAxiom -> MgMonad [Constraint]
gatherConstraints (AnonymousAxiom variables typedExprIR) = go typedExprIR
  where
    go :: TypedExprIR -> MgMonad [Constraint]
    go (TypedExprIR tyExprIR exprIR) = case exprIR of
      ExprIR'Var {} -> noConstraint
      ExprIR'Call {} -> noConstraint
      ExprIR'ValueBlock stmtsIR -> join <$> traverse go (NE.toList stmtsIR)
      ExprIR'EffectfulBlock stmtsIR -> join <$> traverse go (NE.toList stmtsIR)
      ExprIR'Value valueExprIR -> go valueExprIR
      -- TODO(bchetioui): we ignore the right hand side block here, but
      -- I am not sure if this is correct.
      ExprIR'Let {} -> noConstraint -- Do we have to do something here?
      ExprIR'If {} -> -- Conditional equations with condition
        -- Add condExpr to conditions for trueExpr
        -- Should we catch assertions in condition too?
        throwNonLocatedE CompilerErr
          "can not yet gather constraints in conditionals"
      ExprIR'Assert predicateIR -> unfoldPredicateIR predicateIR
      ExprIR'Skip -> noConstraint

    unfoldPredicateIR :: PredicateIR
                      -> MgMonad [Constraint]
    unfoldPredicateIR predicateIR = case predicateIR of
      -- Distribute conditions
      PredicateIR'And lhs rhs -> (<>) <$> unfoldPredicateIR lhs <*>
        unfoldPredicateIR rhs
      PredicateIR'Or {} -> pure [] -- TODO: not sure what to do here
      -- Implication produces a conditional equation
      PredicateIR'Implies lhs rhs -> do
        lhsConditions <- gatherConditions lhs
        rhsConstraints <- unfoldPredicateIR rhs
        pure $ foldl (\constraints condition ->
            map (addConditionToConstraint condition) constraints)
            rhsConstraints lhsConditions
      -- Equivalence is treated as a bidirectional implication
      PredicateIR'Equivalent lhs rhs ->
        (<>) <$> unfoldPredicateIR (PredicateIR'Implies lhs rhs) <*>
                 unfoldPredicateIR (PredicateIR'Implies rhs lhs)
      -- Negation
      PredicateIR'Negation {} -> unimplemented "constraints involving negation"
      -- Predicate equation (see related comment in gatherConditions)
      PredicateIR'PredicateEquation {} ->
        unimplemented "constraints involving predicate equations"
      PredicateIR'Atom atom -> case atom of
        -- This is a fact that holds, but not a rewriting rule. What do we
        -- do? TODO: collect facts somewhere
        PredicateIRAtom'Call {} -> pure []
        PredicateIRAtom'ExprEquation lhs rhs ->
          pure [Constraint'Equational (toEquation lhs rhs)] --undefined
        --PredicateIRAtom'PredicateEquation pi pi' -> undefined
        --PredicateIRAtom'Negation pi -> undefined

    -- We call this when we are looking at a condition (e.g. the left of an
    -- implication). Constructs a list of sets of predicate atoms, such that
    -- each set represents an equivalent condition.
    gatherConditions :: PredicateIR -> MgMonad [S.Set PredicateIRAtom]
    gatherConditions predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> do
        lhsConditions <- gatherConditions lhs
        rhsConditions <- gatherConditions rhs
        let unionRhs s = map (s `S.union`) rhsConditions
        pure $ join $ map unionRhs lhsConditions
      PredicateIR'Or lhs rhs -> do
        (<>) <$> gatherConditions lhs <*> gatherConditions rhs
      -- The below case would imply a condition like (x => y) => z.
      -- For the moment, we decide not to handle it, because this is not
      -- an atomic condition.
      PredicateIR'Implies {} ->
        unimplemented "constraints with nested implications"
      PredicateIR'Equivalent {} ->
        unimplemented "constraints with nested equivalences"
      -- TODO: should negation be an atom, in the end? Otherwise, we can not
      -- really implement it here. Perhaps there should be another subset of
      -- operations for conditions. OTOH, !p can be understood as p => False,
      -- therefore, handling implications should be enough (but this also
      -- requires more machinery...). Let's think about it!
      PredicateIR'Negation {} ->
        unimplemented "constraints involving negation"
      PredicateIR'PredicateEquation {} ->
        -- Here, we could output something like lhs && rhs || !lhs && !rhs.
        -- Unfortunately, we do not support negation, so we can really only
        -- implement half of this (lhs && rhs). It is thus easy enough to
        -- enable partial support, but for now, we tag this as unimplemented.
        unimplemented "constraints involving predicate equations"
      PredicateIR'Atom predicateAtom -> pure [S.singleton predicateAtom]

    addConditionToConstraint :: Condition -> Constraint -> Constraint
    addConditionToConstraint condition constraint = case constraint of
      Constraint'Equational equation ->
        Constraint'ConditionalEquational condition equation
      Constraint'ConditionalEquational condition' equation ->
        Constraint'ConditionalEquational (condition `S.union` condition')
                                         equation

    unimplemented msg = throwNonLocatedE NotImplementedErr $
      msg <> " in construction of rewriting constraints"
    noConstraint = pure []


-- TODO: union-find to implement rewriting

-- An intersection gives one rewriting rule with two atomic constraints; a
-- disjunction gives two rewriting rules with an atomic constraint.


-- | Gathers the axioms within a type checked module expression.
gatherAxioms :: TcModuleExpr -> MgMonad [AnonymousAxiom]
gatherAxioms tcModuleExpr = do
  let callableDecls = getCallableDecls $
        join (M.elems (moduleExprDecls tcModuleExpr))
  mapM makeAxiom (filter ((== Axiom) . _callableType . _elem) callableDecls)
  where
    makeAxiom (Ann (_, absDeclOs) tcAxiomDecl) =
      case _callableBody tcAxiomDecl of
        MagnoliaBody tcBodyExpr -> do
          axiomExprIR <- toTypedExprIR tcBodyExpr
          pure $ AnonymousAxiom
            (S.fromList $ map (_varName . _elem) (_callableArgs tcAxiomDecl))
            axiomExprIR
        _ -> throwNonLocatedE CompilerErr $
          pshow (_callableType tcAxiomDecl) <> " " <>
          pshow (nodeName tcAxiomDecl) <> " (declared at " <>
          pshow (srcCtx $ NE.head absDeclOs) <> ")"

-- | Inlines an expression, i.e. replace all variables by their content as much
-- as possible. Note that for our purposes, inlining does *not* unfold function
-- calls. TODO: add another type for inlining?
-- TODO: this will not need to be an MgMonad if we can deal with procedures
-- properly.
inlineTypedExprIR :: TypedExprIR -> MgMonad TypedExprIR
inlineTypedExprIR inTypedExprIR = snd <$> go M.empty inTypedExprIR
  where
    go :: M.Map Name TypedExprIR -> TypedExprIR
       -> MgMonad (M.Map Name TypedExprIR, TypedExprIR)
    go bindings (TypedExprIR inTyExprIR inExprIR) = case inExprIR of
      ExprIR'Var (VarIR _ name _) -> case M.lookup name bindings of
        -- The only case in which the variable can not be found is when it is
        -- bound outside the expression, i.e. when it is a parameter, and thus
        -- universally quantified.
        Nothing -> ret bindings inExprIR
        Just (TypedExprIR _ exprIR) -> ret bindings exprIR
      ExprIR'Call name args -> do
        args' <- mapM ((snd <$>) . go bindings) args
        case name of
          -- If what we are calling is a procedure, then upd and out arguments
          -- are modified by the call, and we need to produce a valid expression
          -- for each of them. This could be achieved through functionalization
          -- (see 'Interfacing Concepts: Why Declaration Style Shouldn't
          -- Matter').
          -- However, functionalization requires the procedures to be
          -- implemented as several functions as well. Such implementations can
          -- be automatically generated for Magnolia procedures, but not for
          -- external ones.
          -- TODO: implement, or explicitly ignore
          ProcName _ -> throwNonLocatedE NotImplementedErr
            "inlining involving procedure calls"
          _ -> ret bindings (ExprIR'Call name args')
      ExprIR'ValueBlock stmts -> do
        -- In principle, bindings can be discarded when exiting value blocks.
        -- They should not be able to modify existing variables.
        (_, stmts') <- mapAccumM go bindings stmts
        ret bindings (ExprIR'ValueBlock stmts')
      ExprIR'EffectfulBlock stmts -> do
        (bindings', stmts') <- mapAccumM go bindings stmts
        -- Here, bindings' may contain more variables than bindings. We delete
        -- the variables introduced in the effectful block, as they now ran out
        -- of scope.
        let bindings'' = restrictVariables (M.keys bindings) bindings'
        ret bindings'' (ExprIR'ValueBlock stmts')
      ExprIR'Value exprIR -> do
        exprIR' <- snd <$> go bindings exprIR
        ret bindings (ExprIR'Value exprIR')
      ExprIR'Let (VarIR mode name ty) mexpr -> case mexpr of
        Nothing -> ret bindings inExprIR
        Just exprIR -> do
          (_, exprIR') <- go bindings exprIR
          let bindings' = M.insert name exprIR' bindings
          ret bindings'
              (ExprIR'Let (VarIR mode name ty) (Just exprIR'))
      -- TODO: WIP here
      ExprIR'If condPredIR trueExprIR falseExprIR -> do
        -- Cond can discard bindings
        inlineCond <- inlinePredicateIR bindings condPredIR
        (trueBindings, inlineTrueExprIR) <- go bindings trueExprIR
        (falseBindings, inlineFalseExprIR) <- go bindings falseExprIR

        let restrictedTrueBindings =
              restrictVariables (M.keys bindings) trueBindings
            restrictedFalseBindings =
              restrictVariables (M.keys bindings) falseBindings
        finalBindings <- M.fromList <$> mapM (\(k, v1) ->
          case M.lookup k restrictedFalseBindings of
            Nothing -> throwNonLocatedE CompilerErr
              "binding disappeared when inlining false branch"
            Just v2 -> pure (k, joinBranches inlineCond v1 v2))
            (M.toList restrictedTrueBindings)
        ret finalBindings
            (ExprIR'If inlineCond inlineTrueExprIR inlineFalseExprIR)
      ExprIR'Assert predicateIR -> do
        predicateIR' <- inlinePredicateIR bindings predicateIR
        ret bindings (ExprIR'Assert predicateIR')
      ExprIR'Skip -> ret bindings ExprIR'Skip
      where
        ret :: M.Map Name TypedExprIR -> ExprIR
            -> MgMonad (M.Map Name TypedExprIR, TypedExprIR)
        ret bindings' exprIR = pure (bindings', TypedExprIR inTyExprIR exprIR)

    joinBranches :: PredicateIR -> TypedExprIR -> TypedExprIR -> TypedExprIR
    joinBranches cond branch1@(TypedExprIR ty _) branch2 =
      if branch1 == branch2
      then branch1
      else TypedExprIR ty $ ExprIR'If cond branch1 branch2

    restrictVariables :: [Name] -> M.Map Name TypedExprIR
                      -> M.Map Name TypedExprIR
    restrictVariables variablesToKeep bindings =
      let foldFn m v = case M.lookup v bindings of Nothing -> m
                                                   Just e -> M.insert v e m
      in foldl foldFn M.empty variablesToKeep

    inlinePredicateIR :: M.Map Name TypedExprIR -> PredicateIR
                      -> MgMonad PredicateIR
    inlinePredicateIR bindings predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> ipc PredicateIR'And lhs rhs
      PredicateIR'Or lhs rhs -> ipc PredicateIR'Or lhs rhs
      PredicateIR'Implies lhs rhs -> ipc PredicateIR'Implies lhs rhs
      PredicateIR'Equivalent lhs rhs -> ipc PredicateIR'Equivalent lhs rhs
      PredicateIR'PredicateEquation lhs rhs ->
        ipc PredicateIR'PredicateEquation lhs rhs
      PredicateIR'Negation predicateIR' ->
        inlinePredicateIR bindings predicateIR'
      PredicateIR'Atom predicateIRAtom -> PredicateIR'Atom <$>
        inlinePredicateAtom bindings predicateIRAtom
      where ipc = inlinePredicateCombinator bindings

    inlinePredicateCombinator bindings cons lhs rhs = do
      cons <$> inlinePredicateIR bindings lhs <*> inlinePredicateIR bindings rhs

    inlinePredicateAtom :: M.Map Name TypedExprIR -> PredicateIRAtom
                        -> MgMonad PredicateIRAtom
    inlinePredicateAtom bindings predicateIRAtom =
      let inlineExprIR' = (snd <$>) . go bindings in case predicateIRAtom of
      PredicateIRAtom'Call name args -> do
        inlineArgs <- mapM inlineExprIR' args
        pure $ PredicateIRAtom'Call name inlineArgs
      PredicateIRAtom'ExprEquation lhs rhs ->
        PredicateIRAtom'ExprEquation <$> inlineExprIR' lhs <*> inlineExprIR' rhs

    -- TODO: with a proper analysis here, we could know which parameters are
    -- obs, and which are not. In this case, we are unfortunately creating
    -- a block for even obs variables (which are not modified).
    -- Functionalizing can essentially be implemented, for a single variable,
    -- as an assignment of a value block to the variable such that this value
    -- block copies each variable, calls the procedure with the copies as
    -- arguments, and returns the updated value of the component.
    -- the copy as an argument,
    functionalize :: S.Set Name -> ExprIR -> ExprIR
    functionalize bindings (ExprIR'Call (ProcName n) args) =
      let
      in undefined

-- | Extracts the type annotation from a 'TcExpr'.
exprTy :: TcExpr -> MType
exprTy (Ann ty _) = ty

-- TODO: expand variables before converting to equation.

-- | Takes two expressions e1 and e2 and creates an equation out of them. It is
-- assumed that the expressions have been inlined as much as possible, i.e. the
-- only variables left within the expression should be variables that can not be
-- deconstructed (e.g. axiom parameters).
toEquation :: TypedExprIR -> TypedExprIR -> Equation
toEquation srcExprIR tgtExprIR =
  -- TODO: theoretically, the source expression should capture all the variables
  -- in most cases – and probably in all useful cases. For now, we still gather
  -- variables in tgtExpr, and let error handling reject the tricky cases later.
  let variables = allVariables srcExprIR `S.union` allVariables tgtExprIR
  in Equation variables srcExprIR tgtExprIR
  where
    -- Gathers all the variable names within an expression.
    allVariables :: TypedExprIR -> S.Set Name
    allVariables (TypedExprIR _ exprIR) = case exprIR of
      ExprIR'Var (VarIR _ varName _) -> S.singleton varName
      ExprIR'Call _ args ->
        foldl (\s a -> s `S.union` allVariables a) S.empty args
      ExprIR'ValueBlock stmts ->
        foldl (\s a -> s `S.union` allVariables a) S.empty stmts
      ExprIR'EffectfulBlock stmts ->
        foldl (\s a -> s `S.union` allVariables a) S.empty stmts
      ExprIR'Value typedExprIR -> allVariables typedExprIR
      ExprIR'Let (VarIR _ name _) mtypedExprIR -> error "to do"
        --S.insert name (maybe S.empty allVariables mtypedExprIR)
      ExprIR'If condExprIR trueExprIR falseExprIR ->
        allVariablesPred condExprIR `S.union`
        allVariables trueExprIR `S.union` allVariables falseExprIR
      ExprIR'Assert predicateIR -> allVariablesPred predicateIR
      ExprIR'Skip -> S.empty

    allVariablesPred :: PredicateIR -> S.Set Name
    allVariablesPred predicateIR = case predicateIR of
      PredicateIR'And lhs rhs ->
        allVariablesPred lhs `S.union` allVariablesPred rhs
      PredicateIR'Or lhs rhs ->
        allVariablesPred lhs `S.union` allVariablesPred rhs
      PredicateIR'Implies lhs rhs ->
        allVariablesPred lhs `S.union` allVariablesPred rhs
      PredicateIR'Equivalent lhs rhs ->
        allVariablesPred lhs `S.union` allVariablesPred rhs
      PredicateIR'PredicateEquation lhs rhs ->
        allVariablesPred lhs `S.union` allVariablesPred rhs
      PredicateIR'Negation predicateIR' -> allVariablesPred predicateIR'
      PredicateIR'Atom predicateIRAtom -> allVariablesPredAtom predicateIRAtom

    allVariablesPredAtom :: PredicateIRAtom -> S.Set Name
    allVariablesPredAtom predicateIRAtom = case predicateIRAtom of
      PredicateIRAtom'Call _ args ->
        foldl (\s arg -> allVariables arg `S.union` s) S.empty args
      PredicateIRAtom'ExprEquation lhs rhs ->
        allVariables lhs `S.union` allVariables rhs

-- Note: Strategy for rewriting
-- We have two kinds of rewritings: rewritings based on conditional equational
-- constraints, and rewritings based on simple equational constraints. Both
-- kinds of rewritings are "context-aware". For example, suppose we want to
-- apply the following rule:
--
-- x == constant() => f(x, y) == g(y);   (1)
--
-- We need to somehow build a context as we traverse expressions if we want to
-- apply the rule in an expression like the one below:
--
-- { var a = constant();                 (2)
--   value f(a, a);
-- }
--
-- How do we do that? TODO: figure it out.
--

-- rewrite :: Equation -> Expr -> Expr
-- rewrite equation expr | sourceExpr equation == expr = targetExpr equation
-- rewrite equation expr = case expr of
--   Atom _ -> expr
--   Pair expr1 expr2 -> Pair (rewrite equation expr1) (rewrite equation expr2)


-- eqXZ = Equation (Atom "x") (Atom "z")
-- program = Pair (Atom "x") (Pair (Atom "y") (Atom "x"))

-- main :: IO ()
-- main = print $ rewrite eqXZ program

-- Note: Strategy for axiom pattern matching
-- Axioms may rely on function arguments with specific values, specific
-- repetitions of function arguments (e.g. f(a, b, a) has only two distinct
-- arguments), or arguments for which only the type is known.
--
-- We need a way to represent the following constraint:
--
-- { var b = constant();
--   value f(a, b, a) == a;
-- }
--
-- This could look like: Forall a:T. f(a, constant(), a) == a