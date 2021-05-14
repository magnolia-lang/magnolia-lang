{-# LANGUAGE PatternSynonyms #-}

module Env (
  Env, FullyQualifiedName (..), Name (..), NameSpace (..),
  fromFullyQualifiedName,
  pattern FuncName, pattern GenName, pattern ModName, pattern PkgName,
  pattern ProcName, pattern RenamingName, pattern SatName, pattern TypeName,
  pattern UnspecName, pattern VarName)
  where

import qualified Data.Map as M

-- TODO: make this an actual newtype with different utils?
type Env a = M.Map Name a
data Name = Name { _namespace :: NameSpace
                 , _name :: String
                 }
            deriving (Eq, Ord, Show)

data FullyQualifiedName = FullyQualifiedName { _scopeName :: Maybe Name
                                             , _targetName :: Name
                                             }
                        deriving (Eq, Ord, Show)

fromFullyQualifiedName :: FullyQualifiedName -> Name
fromFullyQualifiedName (FullyQualifiedName scopeName targetName) =
  Name (_namespace targetName)
       (maybe "" ((<> ".") . _name) scopeName <> _name targetName)

-- TODO: align instances of Eq, Ord for soundness.

data NameSpace = NSDirectory
               | NSFunction
               | NSGenerated
               | NSModule
               | NSPackage
               | NSProcedure
               | NSRenaming
               | NSSatisfaction
               | NSType
               | NSUnspecified
               | NSVariable
                 deriving (Eq, Ord, Show)

pattern FuncName :: String -> Name
pattern FuncName s = Name NSFunction s

pattern GenName :: String -> Name
pattern GenName s = Name NSGenerated s

pattern ModName :: String -> Name
pattern ModName s = Name NSModule s

pattern PkgName :: String -> Name
pattern PkgName s = Name NSPackage s

pattern ProcName :: String -> Name
pattern ProcName s = Name NSProcedure s

pattern RenamingName :: String -> Name
pattern RenamingName s = Name NSRenaming s

pattern SatName :: String -> Name
pattern SatName s = Name NSSatisfaction s

pattern TypeName :: String -> Name
pattern TypeName s = Name NSType s

pattern UnspecName :: String -> Name
pattern UnspecName s = Name NSUnspecified s

pattern VarName :: String -> Name
pattern VarName s = Name NSVariable s
