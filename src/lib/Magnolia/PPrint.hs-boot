{-# OPTIONS_GHC -Wno-orphans #-}

module Magnolia.PPrint where

import Data.Text as T (Text)
import Prettyprinter (Pretty)

import Backend
import Env

pshow :: Pretty a => a -> T.Text
instance Pretty FullyQualifiedName
instance Pretty Backend