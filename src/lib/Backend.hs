module Backend (
    Backend (..)
  )
  where

-- | The backends for which Magnolia can generate code.
data Backend = Cxx | Cuda | JavaScript | Python
               deriving (Eq, Ord, Show)