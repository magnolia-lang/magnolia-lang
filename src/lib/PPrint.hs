module PPrint (pprint, pshow) where

import Text.Pretty.Simple (pPrint, pShow)

import qualified Data.Text.Lazy as T

pprint :: Show a => a -> IO ()
pprint = pPrint

pshow :: Show a => a -> T.Text
pshow = T.pack . show
