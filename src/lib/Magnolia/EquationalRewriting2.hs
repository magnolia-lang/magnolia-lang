{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module Magnolia.EquationalRewriting2 (
    runGenerator
  , runOptimizer
  )
  where

import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Except as E
import Control.Monad.State
import qualified Data.List as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Void (absurd)

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import Monad
import Data.Foldable (traverse_)

-- TODO: define optimizer modes in the compiler

type Optimizer = TcModule
type CallableGenerator = TcModule

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
                         , eqnSourcePat :: TypedPattern
                         , eqnTargetPat :: TypedPattern
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
                    deriving (Eq, Ord, Show)
data Pattern = Pattern'Wildcard Name
             | Pattern'Expr (GenericExprIR TypedPattern)
               deriving (Eq, Ord, Show)
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

-- | Transforms a typed expression into a typed pattern. The set of variable
-- names passed to 'toPattern' denotes those variables that are universally
-- quantified within the typed expression IR.
toPattern :: S.Set Name -> TypedExprIR -> TypedPattern
toPattern argVariables typedExprIR =
  toPattern' typedExprIR `evalState` (M.empty, 0)
  where
    toPattern' :: TypedExprIR -> State (M.Map Name Name, Int) TypedPattern
    toPattern' (TypedExprIR inTyExprIR inExprIR) =
      TypedPattern inTyExprIR <$> (case inExprIR of
        ExprIR'Var (VarIR mode name ty) -> if name `S.member` argVariables
          then toWildcard name
          else pure $ pe $ ExprIR'Var (VarIR mode name ty)
        ExprIR'Call name args -> pe . ExprIR'Call name <$> mapM toPattern' args
        ExprIR'ValueBlock stmts -> pe . ExprIR'ValueBlock <$>
          mapM toPattern' stmts
        ExprIR'EffectfulBlock stmts -> pe . ExprIR'EffectfulBlock <$>
          mapM toPattern' stmts
        ExprIR'Value exprIR -> pe . ExprIR'Value <$> toPattern' exprIR
        ExprIR'Let (VarIR mode name ty) mexprIR ->
          -- TODO: again, this is dangerous, who knows what will happen...
          pe . ExprIR'Let (VarIR mode name ty) <$>
            (case mexprIR of Nothing -> pure Nothing
                             Just exprIR' -> Just <$> toPattern' exprIR')
          --error "let stmts in toPattern are not implemented"
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

-- | Instantiates a typed pattern into a typed expression, by replacing the
-- wildcards in the pattern with their match in a map of bindings.
-- TODO: forbid equations with variables not instantiated in lhs present in rhs
-- as that would prevent instantiating here.
instantiatePattern :: TypedPattern -> M.Map Name TypedExprIR
                   -> MgMonad TypedExprIR
instantiatePattern (TypedPattern ty pattern) bindings = case pattern of
  Pattern'Wildcard holeName -> case M.lookup holeName bindings of
    Nothing -> throwNonLocatedE CompilerErr $
      "could not find a binding for hole " <> pshow holeName <> " when " <>
      "instantiating a pattern"
    Just typedExprIR -> pure typedExprIR
  Pattern'Expr exprIR -> TypedExprIR ty <$> case exprIR of
    ExprIR'Var varIR -> pure $ ExprIR'Var varIR
    ExprIR'Call name patArgs -> ExprIR'Call name <$> mapM go patArgs
    ExprIR'ValueBlock patStmts -> ExprIR'ValueBlock <$> mapM go patStmts
    ExprIR'EffectfulBlock patStmts -> ExprIR'EffectfulBlock <$> mapM go patStmts
    ExprIR'Value pat -> ExprIR'Value <$> go pat
    ExprIR'Let varName mpat -> ExprIR'Let varName <$> case mpat of
      Nothing -> pure Nothing
      Just pat -> Just <$> go pat
    ExprIR'If patCond patTrue patFalse ->
      ExprIR'If <$> instantiatePredicatePattern patCond
                <*> go patTrue
                <*> go patFalse
    ExprIR'Assert patPredicate -> ExprIR'Assert <$>
      instantiatePredicatePattern patPredicate
    ExprIR'Skip -> pure ExprIR'Skip
  where
    go pat = instantiatePattern pat bindings

    instantiatePredicatePattern :: PredicatePattern -> MgMonad PredicateIR
    instantiatePredicatePattern patPredicate = case patPredicate of
      PredicateIR'And patLhs patRhs ->
        predCombinator PredicateIR'And patLhs patRhs
      PredicateIR'Or patLhs patRhs ->
        predCombinator PredicateIR'Or patLhs patRhs
      PredicateIR'Implies patLhs patRhs ->
        predCombinator PredicateIR'Implies patLhs patRhs
      PredicateIR'Equivalent patLhs patRhs ->
        predCombinator PredicateIR'Equivalent patLhs patRhs
      PredicateIR'PredicateEquation patLhs patRhs ->
        predCombinator PredicateIR'PredicateEquation patLhs patRhs
      PredicateIR'Negation patPredicate' -> PredicateIR'Negation <$>
        instantiatePredicatePattern patPredicate'
      PredicateIR'Atom patPredicateAtom -> PredicateIR'Atom <$>
        instantiatePredicateAtomPattern patPredicateAtom

    instantiatePredicateAtomPattern :: PredicateAtomPattern
                                    -> MgMonad PredicateIRAtom
    instantiatePredicateAtomPattern patPredicateAtom = case patPredicateAtom of
      PredicateIRAtom'Call name patArgs -> PredicateIRAtom'Call name <$>
        mapM go patArgs
      PredicateIRAtom'ExprEquation patLhs patRhs ->
        PredicateIRAtom'ExprEquation <$> go patLhs <*> go patRhs

    predCombinator :: (PredicateIR -> PredicateIR -> PredicateIR)
                   -> PredicatePattern -> PredicatePattern
                   -> MgMonad PredicateIR
    predCombinator constructor patLhs patRhs = constructor <$>
      instantiatePredicatePattern patLhs <*> instantiatePredicatePattern patRhs

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
type ConstraintDB = [Constraint]

-- | Runs a callable generator within one context module.
runGenerator :: TcModuleExpr -> TcModuleExpr -> MgMonad [TcDecl]
runGenerator generatorModuleExpr (Ann _ ctxModuleExpr') = do
  -- 1. gather axioms
  axioms <- gatherAxioms generatorModuleExpr
  -- 2. inline expressions within axioms (TODO)
  inlinedAxioms <- mapM (\(AnonymousAxiom v typedExprIR) ->
    AnonymousAxiom v <$> inlineTypedExprIR typedExprIR) axioms
  -- 3. gather directed rewrite rules
  constraintDb <- join <$> mapM gatherConstraints inlinedAxioms
  liftIO $ mapM_ print constraintDb
  -- 4. traverse each constraint to create a new callable
  x <- case ctxModuleExpr' of
    MModuleDef contextDecls _ _ -> do
      mapM (generateCallable contextDecls generatorModuleExpr)
            constraintDb
    MModuleRef v _ -> absurd v
    MModuleAsSignature v _ -> absurd v
    MModuleTransform _ v -> absurd v
    MModuleExternal _ _ v -> absurd v
  liftIO $ pprint x
  pure x
  where
    generateCallable :: M.Map Name [TcDecl] -> TcModuleExpr -> Constraint
                     -> MgMonad TcDecl
    generateCallable contextDecls generatorExpr constraint = case constraint of
      Constraint'ConditionalEquational {} -> throwNonLocatedE NotImplementedErr
        "conditional equational support in generators"
      Constraint'Equational (Equation _ sourcePat targetPat) ->
        case sourcePat of
          (TypedPattern tyPattern (Pattern'Expr
              (ExprIR'Call callableName patArgs))) -> do
            let extractType (TypedPattern tyPat _) = tyPat
                generatorCallableDecls = M.map getCallableDecls $
                  moduleExprDecls generatorExpr
            callableDecl <- findCallable generatorCallableDecls callableName
              (map extractType patArgs) tyPattern
            defineCallable contextDecls sourcePat callableDecl targetPat
          _ -> undefined

    findCallable :: M.Map Name [TcCallableDecl]
                 -> Name -> [MType] -> MType -> MgMonad TcCallableDecl
    findCallable callableDecls name argTypes returnType = do
      let signature (Ann _ (Callable _ _ args returnType' _ _)) =
            (map (_varType . _elem) args, returnType')
          matches = filter ((== (argTypes, returnType)) . signature) $
            M.findWithDefault [] name callableDecls
      case matches of
        [] -> undefined -- TODO: throw error
        [tcCallableDecl] -> pure tcCallableDecl
        _ -> undefined -- TODO: throw error


-- | Runs an optimizer maxSteps times over one module.
runOptimizer :: TcModuleExpr -> Int -> TcModuleExpr -> MgMonad TcModuleExpr
runOptimizer optimizerModuleExpr maxSteps (Ann src tgtModuleExpr) = do
  -- 1. gather axioms
  axioms <- gatherAxioms optimizerModuleExpr
  -- 2. inline expressions within axioms (TODO)
  inlinedAxioms <- mapM (\(AnonymousAxiom v typedExprIR) ->
    AnonymousAxiom v <$> inlineTypedExprIR typedExprIR) axioms
  -- 3. gather directed rewrite rules
  constraintDb <- join <$> mapM gatherConstraints inlinedAxioms
  liftIO $ mapM_ print constraintDb
  --liftIO $ mapM_ (pprint . fromTypedExprIR) $ M.keys constraintDb
  -- 4. traverse each expression in the target module to rewrite it (while
  -- elaborating the scope)
  resultModuleExpr <- Ann src <$> case tgtModuleExpr of
    MModuleDef decls deps renamings -> do
      decls' <- mapM (traverse $ rewriteDecl constraintDb) decls
      pure $ MModuleDef decls' deps renamings
    MModuleRef v _ -> absurd v
    MModuleAsSignature v _ -> absurd v
    MModuleTransform _ v -> absurd v
    MModuleExternal {} -> pure tgtModuleExpr
  liftIO $ pprint resultModuleExpr
  -- 6: wrap it up
  pure resultModuleExpr
  where
    -- TODO: do better, because this can infinite loop if not terminating
    rewriteDecl :: ConstraintDB -> TcDecl -> MgMonad TcDecl
    rewriteDecl = rewriteDeclNTimes maxSteps

    rewriteDeclNTimes :: Int -> ConstraintDB -> TcDecl -> MgMonad TcDecl
    rewriteDeclNTimes n constraintDb tcDecl
      | n <= 0 = pure tcDecl
      | otherwise = oneRewriteDeclPass constraintDb tcDecl >>= \res ->
          if res == tcDecl -- No more progress
          then pure tcDecl
          else rewriteDeclNTimes (n - 1) constraintDb res


-- | Performs equational rewriting on a Magnolia declaration.
oneRewriteDeclPass :: ConstraintDB -> TcDecl -> MgMonad TcDecl
oneRewriteDeclPass constraintDb tcDecl = case tcDecl of
  MTypeDecl {} -> pure tcDecl
  MCallableDecl mods (Ann callableAnn callableDecl) ->
    case _callableBody callableDecl of
      MagnoliaBody tcExpr -> do
        tcExprIR <- toTypedExprIR tcExpr
        --liftIO $ pprint $ "rewriting " <> pshow (nodeName tcDecl) <> " now"
        tcExprIR' <- oneRewritePass constraintDb tcExprIR
        let tcExpr' = fromTypedExprIR tcExprIR'
            callableDecl' = callableDecl {_callableBody = MagnoliaBody tcExpr'}
        pure (MCallableDecl mods (Ann callableAnn callableDecl'))
      _ -> pure tcDecl

oneRewritePass :: ConstraintDB -> TypedExprIR -> MgMonad TypedExprIR
oneRewritePass = oneRewritePass' S.empty
  where
    oneRewritePass' :: FactDB -> ConstraintDB -> TypedExprIR
                    -> MgMonad TypedExprIR
    oneRewritePass' factDb constraintDb typedExprIR = do
      let oneRuleRewritePass' c = oneRuleRewritePass factDb c typedExprIR
      allRewrites <- mapM oneRuleRewritePass' constraintDb
      -- TODO: add random here? Or how to choose which rule to fire when
      -- fixed point?
      pure $ head $ filter (/= typedExprIR) allRewrites <> [typedExprIR]

    oneRuleRewritePass :: FactDB -> Constraint -> TypedExprIR
                       -> MgMonad TypedExprIR
    oneRuleRewritePass factDb constraint typedExprIR = do
      let sourcePat = eqnSourcePat $ constraintEquation constraint
          targetPat = eqnTargetPat $ constraintEquation constraint
      case typedExprIR `matchesPattern` sourcePat of
        Nothing -> oneRuleRewritePassExprIR factDb constraint typedExprIR
        Just bindings -> case constraint of
          Constraint'Equational eqn -> do
            --liftIO $ pprint $ "firing rule " <> show eqn
            instantiatePattern targetPat bindings
          Constraint'ConditionalEquational {} ->
            throwNonLocatedE NotImplementedErr
              "conditional equational rewriting"

    oneRuleRewritePassExprIR :: FactDB -> Constraint -> TypedExprIR
                             -> MgMonad TypedExprIR
    oneRuleRewritePassExprIR factDb constraint (TypedExprIR tyExprIR exprIR) =
      let go = oneRuleRewritePass factDb constraint in
      TypedExprIR tyExprIR <$> (case exprIR of
        ExprIR'Var varIR -> pure $ ExprIR'Var varIR
        ExprIR'Call name args -> ExprIR'Call name <$> mapM go args
        -- TODO: traverse and enrich factDb
        ExprIR'ValueBlock stmts -> ExprIR'ValueBlock <$> mapM go stmts
        -- TODO: traverse and enrich factDb
        ExprIR'EffectfulBlock stmts -> ExprIR'EffectfulBlock <$> mapM go stmts
        ExprIR'Value tyExprIR' -> ExprIR'Value <$> go tyExprIR'
        ExprIR'Let varIR mtyExprIR' -> ExprIR'Let varIR <$> case mtyExprIR' of
          Nothing -> pure Nothing
          Just tyExprIR' -> Just <$> go tyExprIR'
        ExprIR'If pi x0 x1 -> undefined
        ExprIR'Assert pi -> undefined
        ExprIR'Skip -> undefined)
    -- TODO: keep iterating here

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
          (:[]) . Constraint'Equational <$> toEquation variables lhs rhs --undefined
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
inlineTypedExprIR = flip inlineTypedExprIRWith M.empty

inlineTypedExprIRWith :: TypedExprIR -> M.Map Name TypedExprIR
                      -> MgMonad TypedExprIR
inlineTypedExprIRWith inTypedExprIR inBindings = snd <$>
  go inBindings inTypedExprIR
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
          ProcName _ -> do
            -- TODO: this is mega dangerous, and not correct in all
            -- cases. This has to be safeguarded before being
            -- merged, so that only well-behaved cases work out.
            -- TODO: we invalidate all the variables that are passed as
            -- arguments to the procedure, as a precaution. This is not 100%
            -- correct, we should rely on actually identifying which ones are
            -- modified, but this is best effort for the moment.
            -- Rewritings that use procedures will only work in some simple
            -- cases.
            let bindings' = foldl invalidateProcedureArgBinding bindings args
            ret bindings' (ExprIR'Call name args)
            --throwNonLocatedE NotImplementedErr
            --"inlining involving procedure calls"
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
        -- TODO: add bindings here.
        ret bindings (ExprIR'Assert predicateIR')
      ExprIR'Skip -> ret bindings ExprIR'Skip
      where
        ret :: M.Map Name TypedExprIR -> ExprIR
            -> MgMonad (M.Map Name TypedExprIR, TypedExprIR)
        ret bindings' exprIR = pure (bindings', TypedExprIR inTyExprIR exprIR)

        invalidateProcedureArgBinding :: M.Map Name TypedExprIR -> TypedExprIR
                                      -> M.Map Name TypedExprIR
        invalidateProcedureArgBinding bindings' (TypedExprIR _ exprIR) =
          case exprIR of
            ExprIR'Var (VarIR _ name _) -> M.delete name bindings'
            _ -> bindings'

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
-- only variables left within the expression should be universally quantified
-- variables (e.g. axiom parameters – where the axioms are unguarded), or
-- variables declared as part of the expression (if the expression is a block).
-- The set of variables passed to 'toEquation' corresponds to the universally
-- quantified variables that may appear within the expression.
toEquation :: S.Set Name -> TypedExprIR -> TypedExprIR -> MgMonad Equation
toEquation argVariables srcExprIR tgtExprIR = do
  let srcVariables = allVariables srcExprIR
      tgtVariables = allVariables tgtExprIR
      equationExpr = TypedExprIR Pred $ ExprIR'Assert $ PredicateIR'Atom $
        PredicateIRAtom'ExprEquation srcExprIR tgtExprIR
      (TypedPattern Pred
        (Pattern'Expr
          (ExprIR'Assert
            (PredicateIR'Atom
              (PredicateIRAtom'ExprEquation srcPat tgtPat))))) =
                toPattern argVariables equationExpr
  -- The set of universally quantified variables in the target pattern must be
  -- a subset of the set of universally quantified variables in the source
  -- pattern.
  unless (tgtVariables `S.isSubsetOf` srcVariables) $
    throwNonLocatedE MiscErr $ "could not convert rule " <>
      pshow (fromTypedExprIR equationExpr) <> " to pattern: the target " <>
      "expression uses variables not present in the source expression"
  pure $ Equation srcVariables srcPat tgtPat
  where
    -- Gathers all the variable names within a pattern.
    allVariables :: TypedExprIR -> S.Set Name
    allVariables (TypedExprIR _ exprIR) = case exprIR of
      ExprIR'Var {} -> S.empty
      ExprIR'Call _ args ->
        foldl (\s a -> s `S.union` allVariables a) S.empty args
      ExprIR'ValueBlock stmts ->
        foldl (\s a -> s `S.union` allVariables a) S.empty stmts
      ExprIR'EffectfulBlock stmts ->
        foldl (\s a -> s `S.union` allVariables a) S.empty stmts
      ExprIR'Value pat -> allVariables pat
      ExprIR'Let _ mpat -> maybe S.empty allVariables mpat
        --S.insert name (maybe S.empty allVariables mtypedExprIR)
      ExprIR'If cond trueExprIR falseExprIR ->
        allVariablesPred cond `S.union`
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
      PredicateIR'Atom predicateIRAtom ->
        allVariablesPredAtom predicateIRAtom

    allVariablesPredAtom :: PredicateIRAtom -> S.Set Name
    allVariablesPredAtom predicateIRAtom = case predicateIRAtom of
      PredicateIRAtom'Call _ args ->
        foldl (\s arg -> allVariables arg `S.union` s) S.empty args
      PredicateIRAtom'ExprEquation lhs rhs ->
        allVariables lhs `S.union` allVariables rhs

defineCallable :: M.Map Name [TcDecl]
               -> TypedPattern
               -- ^ the pattern corresponding
               -> TcCallableDecl
               -- ^ the definition of the callable within the rewrite module
               -> TypedPattern
               -> MgMonad TcDecl
defineCallable tcDecls
    (TypedPattern tyPattern (Pattern'Expr (ExprIR'Call callableName patArgs)))
    (Ann (mconDeclO, absDeclOs)
         callable@(Callable _ defName defArgs defRetTy _ EmptyBody))
    rhsPattern | defRetTy == tyPattern && defName == callableName &&
                 defArgTypes == patArgTypes = do
  let wildcardArgs = S.fromList $ filter isWildcard patArgs
      defArgVars = map
        (\(Ann _ (Var m n t)) -> TypedExprIR t (ExprIR'Var (VarIR m n t)))
        defArgs
  unless (S.size wildcardArgs == length patArgs) $
    throwNonLocatedE CompilerErr "WIP defineCallable"
  patVarNames <- mapM getWildcardName patArgs
  let bindings = M.fromList $ zip patVarNames defArgVars
  bodyExpr <- fromTypedExprIR <$> (instantiatePattern rhsPattern bindings >>=
    unfoldCalls tcDecls)
  -- TODO: GeneratedBuiltin here is a hack. We should define a new constructor
  -- and do things properly.
  let declOs = case mconDeclO of Nothing -> (Just GeneratedBuiltin, absDeclOs)
                                 _ -> (mconDeclO, absDeclOs)
  pure $ MCallableDecl []
    (Ann declOs callable { _callableBody = MagnoliaBody bodyExpr })
  where
    isWildcard (TypedPattern _ (Pattern'Wildcard _)) = True
    isWildcard _ = False

    getWildcardName (TypedPattern _ (Pattern'Wildcard n)) = pure n
    getWildcardName _ = throwNonLocatedE CompilerErr
      "WIP defineCallable getWildcardName"

    defArgTypes = map (_varType . _elem) defArgs
    patArgTypes = map (\(TypedPattern patTy _) -> patTy) patArgs
-- TODO: throw proper errors
defineCallable _ _ _ _ = error "invalid call to defineCallable"


-- | Unfolds all calls within an expression.
-- TODO: perform alpha renaming of variables inside to avoid name clashes.
unfoldCalls :: M.Map Name [TcDecl] -> TypedExprIR -> MgMonad TypedExprIR
unfoldCalls tcDecls inTypedExprIR@(TypedExprIR inTyExprIR inExprIR) =
  case inExprIR of
    ExprIR'Var {} -> pure inTypedExprIR
    ExprIR'Call callableName args -> do
      let argTypes = map (\(TypedExprIR ty _) -> ty) args
      unfoldedArgs <- mapM unfoldCalls' args
      findCallable callableName argTypes inTyExprIR >>=
        flip unfoldWith unfoldedArgs >>= ret
    ExprIR'ValueBlock stmts ->
      mapM unfoldCalls' stmts >>= ret . ExprIR'ValueBlock
    ExprIR'EffectfulBlock stmts ->
      mapM unfoldCalls' stmts >>= ret . ExprIR'EffectfulBlock
    ExprIR'Value typedExprIR ->
      unfoldCalls' typedExprIR >>= ret . ExprIR'Value
    ExprIR'Let varName mtypedExprIR ->
      maybe (pure inTypedExprIR)
            (unfoldCalls' >=> ret . ExprIR'Let varName . Just) mtypedExprIR
    ExprIR'If cond trueExprIR falseExprIR -> do
      ExprIR'If <$> unfoldCallsPred cond <*> unfoldCalls' trueExprIR
                <*> unfoldCalls' falseExprIR >>= ret
    ExprIR'Assert predicateIR -> unfoldCallsPred predicateIR >>=
      ret . ExprIR'Assert
    ExprIR'Skip -> ret ExprIR'Skip
  where
    callableDecls = M.map getCallableDecls tcDecls
    findCallable :: Name -> [MType] -> MType -> MgMonad TcCallableDecl
    findCallable name argTypes returnType = do
      let signature (Ann _ (Callable _ _ args returnType' _ _)) =
            (map (_varType . _elem) args, returnType')
          matches = filter ((== (argTypes, returnType)) . signature) $
            M.findWithDefault [] name callableDecls
      case matches of
        -- TODO: prototypes should be inserted
        [] -> throwNonLocatedE CompilerErr $
          "could not find " <> pshow name <> pshow argTypes-- TODO: throw error
        [tcCallableDecl] -> pure tcCallableDecl
        _ -> undefined -- TODO: throw error

    ret :: ExprIR -> MgMonad TypedExprIR
    ret = pure . TypedExprIR inTyExprIR

    unfoldCalls' = unfoldCalls tcDecls

    unfoldCallsPred :: PredicateIR -> MgMonad PredicateIR
    unfoldCallsPred predicateIR = case predicateIR of
      PredicateIR'And lhs rhs -> PredicateIR'And <$>
        unfoldCallsPred lhs <*> unfoldCallsPred rhs
      PredicateIR'Or lhs rhs -> PredicateIR'Or <$>
        unfoldCallsPred lhs <*> unfoldCallsPred rhs
      PredicateIR'Implies lhs rhs -> PredicateIR'Implies <$>
        unfoldCallsPred lhs <*> unfoldCallsPred rhs
      PredicateIR'Equivalent lhs rhs -> PredicateIR'Equivalent <$>
        unfoldCallsPred lhs <*> unfoldCallsPred rhs
      PredicateIR'PredicateEquation lhs rhs -> PredicateIR'PredicateEquation <$>
        unfoldCallsPred lhs <*> unfoldCallsPred rhs
      PredicateIR'Negation predicateIR' -> PredicateIR'Negation <$>
        unfoldCallsPred predicateIR'
      PredicateIR'Atom predicateIRAtom -> PredicateIR'Atom <$>
        unfoldCallsPredAtom predicateIRAtom

    unfoldCallsPredAtom :: PredicateIRAtom -> MgMonad PredicateIRAtom
    unfoldCallsPredAtom predicateIRAtom = case predicateIRAtom of
      -- TODO: here, we should also unwrap predicate calls, but this doesn't
      -- fit well here atm. Let's see how to do it later.
      PredicateIRAtom'Call name args -> pure $
        PredicateIRAtom'Call name args
      PredicateIRAtom'ExprEquation lhs rhs -> PredicateIRAtom'ExprEquation <$>
        unfoldCalls' lhs <*> unfoldCalls' rhs

    -- TODO: need to do alpha renaming inside here to avoid shadowing problems!
    -- But, we do not have time for the paper. This is WIP.
    unfoldWith :: TcCallableDecl -> [TypedExprIR] -> MgMonad ExprIR
    unfoldWith (Ann _ (Callable _ name argVars _ _ body)) args = case body of
      MagnoliaBody tcExpr -> do
        typedExprIR <- toTypedExprIR tcExpr
        let argBindings = M.fromList $ zip (map nodeName argVars) args
        (TypedExprIR _ exprIR) <- inlineTypedExprIRWith typedExprIR argBindings
        pure exprIR
      -- In other cases, there is nothing to unfold.
      _ -> pure $ ExprIR'Call name args

-- TODO: this works but we should change it later.
-- mergeModuleDecls :: TcModule -> TcModule -> MgMonad (M.Map Name [TcDecl])
-- mergeModuleDecls module1 module2 = enter (PkgName "dummy") $
--   moduleDecls <$> checkModule topLevelEnv compositeModule
--   where
--     topLevelEnv = M.insert (nodeName module1) [MModuleDecl module1]
--       (M.insert (nodeName module2) [MModuleDecl module2] M.empty)

--     compositeModule :: ParsedModule
--     compositeModule = Ann (SrcCtx Nothing) (MModule Implementation
--       (ModName "#dummy#") (Ann (SrcCtx Nothing) (MModuleDef [] deps [])))

--     mkRef :: TcModule -> ParsedModuleExpr
--     mkRef tcModule@(Ann _ (MModule moduleTy _ _)) = Ann (SrcCtx Nothing) $
--       case moduleTy of
--         Concept -> MModuleAsSignature
--           (FullyQualifiedName Nothing (nodeName tcModule)) []
--         _ -> MModuleRef (FullyQualifiedName Nothing (nodeName tcModule)) []

--     deps :: [ParsedModuleDep]
--     deps = [ Ann (SrcCtx Nothing) $
--               MModuleDep MModuleDepUse
--                 (mkRef module1)
--            , Ann (SrcCtx Nothing) $
--               MModuleDep MModuleDepUse
--                 (mkRef module2)
--            ]

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