module Backend (
    Backend (..)
  )
  where

-- | The backends for which Magnolia can generate code.
data Backend = Cxx | JavaScript | Python
               deriving (Eq, Ord, Show)