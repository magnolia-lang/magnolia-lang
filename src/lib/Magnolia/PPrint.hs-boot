{-# OPTIONS_GHC -Wno-orphans #-}

module Magnolia.PPrint where

import Data.Text.Prettyprint.Doc (Pretty)
import Data.Text.Lazy as T (Text)

import Backend
import Env

pshow :: Pretty a => a -> T.Text
instance Pretty FullyQualifiedName
instance Pretty Backend