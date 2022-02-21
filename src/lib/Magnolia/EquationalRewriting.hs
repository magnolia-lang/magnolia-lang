{-# LANGUAGE OverloadedStrings #-}

module Magnolia.EquationalRewriting (
    runOptimizer
  )
  where

import Control.Monad (foldM, join)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import Data.Maybe (isNothing)
import qualified Data.Set as S
import qualified Data.Text.Lazy as T

import Env
import Magnolia.PPrint
import Magnolia.Syntax
import Monad

-- TODO: define optimizer modes in the compiler

type Optimizer = TcModule

-- TODO: should we only optimize Programs here?

-- | Runs an optimizer one time over one module.
runOptimizer :: Optimizer -> TcModule -> MgMonad TcModule
runOptimizer (Ann _ optimizer) (Ann tgtDeclO tgtModule) = case optimizer of
  MModule Concept optimizerName optimizerModuleExpr -> enter optimizerName $ do
    -- 1. gather axioms
    axioms <- gatherAxioms optimizerModuleExpr
    -- 2. inline expressions within axioms (TODO)
    -- 3. gather assertions (directed rewrite rules)
    constraints <- join <$> mapM gatherConstraints axioms
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
                                     , axiomBody :: TcExpr
                                     }

-- | Extracts a list of constraints from an axiom. Every assertion involving an
-- equation found within the body of the axiom produces a (possibly conditional)
-- constraint. To ensure that this operation works correctly, the parameter's
-- expression must have been inlined beforehand.
gatherConstraints :: AnonymousAxiom -> MgMonad [Constraint]
gatherConstraints (AnonymousAxiom variables bodyExpr) = go bodyExpr
  where
    go :: TcExpr -> MgMonad [Constraint]
    go expr = case _elem expr of
      MVar _ -> noConstraint
      MCall {} -> noConstraint
      MBlockExpr _ statements -> join <$> traverse go (NE.toList statements)
      MValue valueExpr -> go valueExpr
      -- TODO(bchetioui): we do not ignore the right hand side block here, but
      -- I am not sure if this is correct.
      MLet {} -> noConstraint -- Do we have to do something here?
      MIf condExpr trueExpr falseExpr -> do
        undefined -- Conditional equations with condition
      MAssert assertExpr -> undefined -- Equation logic
      MSkip -> noConstraint


    noConstraint = pure []

-- | Transforms a predicate expression into a 'PredicateAtom'. 'toPredicateAtom'
-- expects its expression parameter to have been fully inlined.
-- TODO: right now, we also assume it does not contain "&&", "||", "=>", and
-- "<=>". Is this a reasonable assumption to make?
toPredicateAtom :: TcExpr -> MgMonad PredicateAtom
-- Ensure tcExpr is a predicate
toPredicateAtom tcExpr | exprTy tcExpr /= Pred =
  throwNonLocatedE CompilerErr $ "expected predicate when " <>
    "building predicate atom, but " <> pshow tcExpr <> " has type " <>
    pshow (exprTy tcExpr)
toPredicateAtom tcExpr = case _elem tcExpr of
  MVar {} -> throwNonLocatedE CompilerErr $ "expected fully inlined " <>
    "expression when building predicate atom, but got variable " <> pshow tcExpr
  MCall name args _ -> case (name, map exprTy args) of
    (FuncName "_==_", [Pred, Pred]) -> let ~[lhsExpr, rhsExpr] = args in
      pure $ PredicateAtom'Equation lhsExpr rhsExpr
    (FuncName "_!=_", [Pred, Pred]) -> let ~[lhsExpr, rhsExpr] = args in
      pure $ PredicateAtom'Negation (PredicateAtom'Equation lhsExpr rhsExpr)
    (FuncName "!_", [Pred]) -> let negExpr = head args in
      PredicateAtom'Negation <$> toPredicateAtom negExpr
    -- TODO(bchetioui): what happens here with implication, equivalence, or, and?
    _ -> pure $ PredicateAtom'Call name args
  MBlockExpr {} -> unimplemented "block expressions"
  MValue valueExpr -> toPredicateAtom valueExpr
  MLet {} -> unexpected $ "let binding " <> pshow tcExpr
  MIf {} -> unimplemented "if expressions"
  MAssert {} -> unexpected $ "assertion " <> pshow tcExpr
  MSkip -> unexpected "skip expression"
  where
    unexpected unexpectedSourceInfo = throwNonLocatedE CompilerErr $
      "unexpected " <> unexpectedSourceInfo <> " when building predicate atom"
    unimplemented exprKind = throwNonLocatedE NotImplementedErr $
      "building predicate atoms from " <> exprKind <> " is not implemented"




-- An intersection gives one rewriting rule with two atomic constraints; a
-- disjunction gives two rewriting rules with an atomic constraint.

-- -- | 'PredicateAtom's represent the subset of Magnolia expressions corresponding
-- -- to atomic predicates, that is to say equations, calls to custom predicates,
-- -- or the negation of a predicate.
-- data PredicateAtom = PredicateAtom'Call Name [TcExpr]
--                    | PredicateAtom'Equation TcExpr TcExpr
--                    | PredicateAtom'Negation PredicateAtom


--PredicateAtom'Implication PredicateAtom PredicateAtom


-- data Equation = Equation { eqnVariables :: S.Set Name
--                          , eqnSourceExpr :: TcExpr
--                          , eqnTargetExpr :: TcExpr
--                          }

-- -- | We represent a condition as the intersection of a set of predicate
-- -- expressions.
-- type Condition = S.Set TcExpr

-- data Constraint = Constraint'Equational Equation
--                 | Constraint'ConditionalEquational Condition Equation


-- | Gathers the axioms within a type checked module expression.
gatherAxioms :: TcModuleExpr -> MgMonad [AnonymousAxiom]
gatherAxioms tcModuleExpr = do
  let callableDecls = getCallableDecls $
        join (M.elems (moduleExprDecls tcModuleExpr))
  mapM makeAxiom (filter ((== Axiom) . _callableType . _elem) callableDecls)
  where
    makeAxiom (Ann (_, absDeclOs) tcAxiomDecl) =
      case _callableBody tcAxiomDecl of
        MagnoliaBody tcBodyExpr -> pure $ AnonymousAxiom
          (S.fromList $ map (_varName . _elem) (_callableArgs tcAxiomDecl))
          tcBodyExpr
        _ -> throwNonLocatedE CompilerErr $
          pshow (_callableType tcAxiomDecl) <> " " <>
          pshow (nodeName tcAxiomDecl) <> " (declared at " <>
          pshow (srcCtx $ NE.head absDeclOs) <> ")"

-- | Inlines an expression, i.e. replace all variables by their content as much
-- as possible.
inlineExpr :: undefined
inlineExpr = undefined



-- | Extracts the type annotation from a 'TcExpr'.
exprTy :: TcExpr -> MType
exprTy (Ann ty _) = ty

-- TODO: expand variables before converting to equation.

-- | Takes two expressions e1 and e2 and creates an equation out of them. It is
-- assumed that the expressions have been inlined as much as possible, i.e. the
-- only variables left within the expression should be variables that can not be
-- deconstructed (e.g. axiom parameters).
toEquation :: TcExpr -> TcExpr -> Equation
toEquation srcExpr tgtExpr =
  -- TODO: theoretically, the source expression should capture all the variables
  -- in most cases – and probably in all useful cases. For now, we still gather
  -- variables in tgtExpr, and let error handling reject the tricky cases later.
  let variables = allVariables srcExpr `S.union` allVariables tgtExpr
  in Equation variables srcExpr tgtExpr
  where
    -- Gathers all the variable names within an expression.
    allVariables :: TcExpr -> S.Set Name
    allVariables (Ann _ expr) = case expr of
      MVar (Ann _ v) -> S.singleton $ _varName v
      MCall _ args _ -> foldl (\s a -> s `S.union` allVariables a) S.empty args
      MBlockExpr _ stmts ->
        foldl (\s a -> s `S.union` allVariables a) S.empty stmts
      MValue expr' -> allVariables expr'
      MLet (Ann _ v) mexpr' ->
        S.insert (_varName v) (maybe S.empty allVariables mexpr')
      MIf condExpr trueExpr falseExpr -> allVariables condExpr `S.union`
        allVariables trueExpr `S.union` allVariables falseExpr
      MAssert expr' -> allVariables expr'
      MSkip -> S.empty


data Equation = Equation { eqnVariables :: S.Set Name
                         , eqnSourceExpr :: TcExpr
                         , eqnTargetExpr :: TcExpr
                         }

-- | 'PredicateAtom's represent the subset of Magnolia expressions corresponding
-- to atomic predicates, that is to say equations, calls to custom predicates,
-- or the negation of a predicate.
data PredicateAtom = PredicateAtom'Call Name [TcExpr]
                   | PredicateAtom'Equation TcExpr TcExpr
                   | PredicateAtom'Negation PredicateAtom

-- | We represent a condition as the intersection of a set of predicate
-- expressions.
type Condition = S.Set PredicateAtom

-- TODO: how to deal with variables in predicates?
data Constraint = Constraint'Equational Equation
                | Constraint'ConditionalEquational Condition Equation

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