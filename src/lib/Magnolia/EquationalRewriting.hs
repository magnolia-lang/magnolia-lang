{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module Magnolia.EquationalRewriting (
    runOptimizer
  )
  where

import Control.Monad (foldM, join)
import Control.Monad.IO.Class (liftIO)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isNothing)
import qualified Data.Set as S
import qualified Data.Text.Lazy as T

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import Monad

-- TODO: define optimizer modes in the compiler

type Optimizer = TcModule

-- TODO: should we only optimize Programs here?

-- == optimizer IR ==

-- | An IR to represent expressions during this equational rewriting phase. The
-- representation is introduced to manipulate predicates with more ease.
-- TODO: fix up TcExpr so that it is the opposite: predicate combinatores will
-- not be represented as functions, unless passing through a STF/possibly code
-- generation. This should simplify some things.
-- TODO: round-tripping is not the identity right now, since the Checked AST
-- makes less assumptions than is possible. It would be better to round-trip
-- if possible, but again, we should modify the core checked AST for that.
type TypedExprIR = (MType, ExprIR)
data ExprIR = ExprIR'Var VarIR
            | ExprIR'Call Name [TypedExprIR]
            | ExprIR'ValueBlock (NE.NonEmpty TypedExprIR)
            | ExprIR'EffectfulBlock (NE.NonEmpty TypedExprIR)
            | ExprIR'Value TypedExprIR
            | ExprIR'Let VarIR (Maybe TypedExprIR)
            | ExprIR'If PredicateIR TypedExprIR TypedExprIR
            | ExprIR'Assert PredicateIR
            | ExprIR'Skip
              deriving (Eq, Ord, Show)

data VarIR = VarIR MVarMode Name MType
             deriving (Eq, Ord, Show)

data PredicateIR = PredicateIR'And PredicateIR PredicateIR
                 | PredicateIR'Or PredicateIR PredicateIR
                 | PredicateIR'Implies PredicateIR PredicateIR
                 | PredicateIR'Equivalent PredicateIR PredicateIR
                 | PredicateIR'PredicateEquation PredicateIR PredicateIR
                 | PredicateIR'Negation PredicateIR
                 | PredicateIR'Atom PredicateIRAtom
                   deriving (Eq, Ord, Show)

data Equation = Equation { eqnVariables :: S.Set Name
                         , eqnSourceExpr :: TypedExprIR
                         , eqnTargetExpr :: TypedExprIR
                         }
                deriving (Eq, Show)

-- | 'PredicateIRAtom's represent the subset of Magnolia expressions
-- corresponding to atomic predicates, that is to say equations, or calls to
-- custom predicates.
data PredicateIRAtom = PredicateIRAtom'Call Name [TypedExprIR]
                     | PredicateIRAtom'ExprEquation TypedExprIR TypedExprIR
                       deriving (Eq, Ord, Show)

-- | Converts a 'TcExpr' into its equivalent 'TypedExprIR' representation.
toTypedExprIR :: TcExpr -> MgMonad TypedExprIR
toTypedExprIR (Ann inTy inTcExpr) = (inTy,) <$> go inTcExpr
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
fromTypedExprIR (tyExprIR, exprIR) = Ann tyExprIR $ case exprIR of
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

newtype Fact = Fact PredicateIRAtom

-- | Runs an optimizer one time over one module.
runOptimizer :: Optimizer -> TcModule -> MgMonad TcModule
runOptimizer (Ann _ optimizer) (Ann tgtDeclO tgtModule) = case optimizer of
  MModule Concept optimizerName optimizerModuleExpr -> enter optimizerName $ do
    -- 1. gather axioms
    axioms <- gatherAxioms optimizerModuleExpr
    --liftIO $ print axioms
    -- 2. inline expressions within axioms (TODO)
    inlinedAxioms <- mapM (\(AnonymousAxiom v typedExprIR) ->
      AnonymousAxiom v <$> inlineTypedExprIR typedExprIR) axioms
    liftIO $ mapM_ (\(AnonymousAxiom _ e) -> pprint (fromTypedExprIR e)) inlinedAxioms
    undefined
    -- 3. gather directed rewrite rules
    constraints <- join <$> mapM gatherConstraints axioms
    -- == debug ==
    let printable = map (\c -> (S.toList $ eqnVariables $ constraintEquation c, fromTypedExprIR . eqnSourceExpr . constraintEquation $ c, fromTypedExprIR . eqnTargetExpr . constraintEquation $ c)) constraints
    liftIO $ mapM_ pprint printable
    --liftIO $ print constraints
    -- ==/ debug ==
    -- 4. build some kind of assertion scope
    equationalScope <- undefined
    -- 5. traverse each expression in the target module to rewrite it
    let ~(MModule tgtModuleTy tgtModuleName tgtModuleExpr) = tgtModule
    resultModuleExpr <- undefined
    -- 6: wrap it up
    -- comment out to avoid IDE suggestion
    --pure $ Ann tgtDeclO (MModule tgtModuleTy tgtModuleName resultModuleExpr)
    undefined
  _ -> undefined

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
    go (tyExprIR, exprIR) = case exprIR of
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
    go bindings (inTyExprIR, inExprIR) = case inExprIR of
      ExprIR'Var (VarIR _ name _) -> case M.lookup name bindings of
        -- The only case in which the variable can not be found is when it is
        -- bound outside the expression, i.e. when it is a parameter, and thus
        -- universally quantified.
        Nothing -> ret bindings inExprIR
        Just (_, exprIR) -> ret bindings exprIR
      ExprIR'Call name args -> do
        case name of
          ProcName _ -> throwNonLocatedE NotImplementedErr
            "inlining of procedures has not been implemented"
          _ -> pure ()
        args' <- mapM ((snd <$>) . go bindings) args
        ret bindings (ExprIR'Call name args')
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
        ret bindings' exprIR = pure (bindings', (inTyExprIR, exprIR))

    joinBranches :: PredicateIR -> TypedExprIR -> TypedExprIR -> TypedExprIR
    joinBranches cond branch1@(ty, _) branch2 =
      if branch1 == branch2
      then branch1
      else (ty, ExprIR'If cond branch1 branch2)

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
    allVariables (_, exprIR) = case exprIR of
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