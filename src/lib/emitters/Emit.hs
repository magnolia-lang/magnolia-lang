module Emit (Emitter (..)) where

import Syntax

class Emitter a where
  emitGlobalEnv :: GlobalEnv -> a
  emitPackage :: UPackage -> a
  emitModule :: UModule -> a
  emitDecl :: UDecl -> a
  emitExpr :: UExpr -> a
