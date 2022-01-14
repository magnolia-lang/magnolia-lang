from collections import namedtuple

def int16_utils():
    class Int16: # unsigned int16
        val: int
        def __init__(self, val):
            self.val = val % 2**16

        def __add__(self, i2):
            self.val = (self.val + i2.val) % 2**16
            return self

        def __lt__(self, i2):
            return self.val < i2.val

        def __repr__(self):
            return f"{self.__class__.__name__}({self.val})"

        def mutate(self, i2):
            self.val = i2.val

    def one():
        return Int16(1)

    def add(a: Int16, b: Int16) -> Int16:
        return a + b

    def isLowerThan(a: Int16, b: Int16) -> bool:
        return a < b

    int16_utils_tuple = namedtuple('int16_utils',
                                   ['Int16', 'add', 'isLowerThan', 'one'])
    return int16_utils_tuple(Int16, add, isLowerThan, one)
