from collections import namedtuple

def int32_utils():
    class Int32: # unsigned int32
        val: int

        def __init__(self, val):
            self.val = val % 2**32

        def __add__(self, i2):
            self.val = (self.val + i2.val) % 2**32
            return self

        def __lt__(self, i2):
            return self.val < i2.val

        def __repr__(self):
            return f"{self.__class__.__name__}({self.val})"

        def mutate(self, i2):
            self.val = i2.val

    def one():
        return Int32(1)

    def add(a: Int32, b: Int32) -> Int32:
        return a + b

    def isLowerThan(a: Int32, b: Int32) -> bool:
        return a < b

    int32_utils_tuple = namedtuple('int32_utils',
                                   ['Int32', 'add', 'isLowerThan', 'one'])
    return int32_utils_tuple(Int32, add, isLowerThan, one)
