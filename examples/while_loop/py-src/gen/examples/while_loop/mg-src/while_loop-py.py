from base import int16_utils
from base import int32_utils
from base import while_ops
from collections import namedtuple
import functools
import multiple_dispatch


def IterationProgram():
    overload = functools.partial(multiple_dispatch.overload, {})
    __int32_utils = int32_utils()
    __int16_utils = int16_utils()
    Int32 = __int32_utils.Int32
    @overload(Int32, Int32, return_type=Int32)
    def add(a, b):
        return __int32_utils.add(a, b)

    @overload(Int32, Int32, return_type=None)
    def increment(counter, bound):
        counter.mutate(add(counter, one(return_type=Int32), return_type=Int32))

    @overload(Int32, Int32, return_type=bool)
    def isLowerThan(a, b):
        return __int32_utils.isLowerThan(a, b)

    __while_ops0 = while_ops(Int32, Int32, increment, isLowerThan)
    @overload(return_type=Int32)
    def one():
        return __int32_utils.one()

    @overload(Int32, Int32, return_type=None)
    def repeat(s, c):
        return __while_ops0.repeat(s, c)

    Int16 = __int16_utils.Int16
    __while_ops = while_ops(Int16, Int16, increment, isLowerThan)
    @overload(Int16, Int16, return_type=Int16)
    def add(a, b):
        return __int16_utils.add(a, b)

    @overload(Int16, Int16, return_type=None)
    def increment(counter, bound):
        counter.mutate(add(counter, one(return_type=Int16), return_type=Int16))

    @overload(Int16, Int16, return_type=bool)
    def isLowerThan(a, b):
        return __int16_utils.isLowerThan(a, b)

    @overload(return_type=Int16)
    def one():
        return __int16_utils.one()

    @overload(Int16, Int16, return_type=None)
    def repeat(s, c):
        return __while_ops.repeat(s, c)

    __namedtuple = namedtuple("IterationProgram", ["Int16", "Int32", "add", "increment", "isLowerThan", "one", "repeat"])
    return __namedtuple(Int16, Int32, add, increment, isLowerThan, one, repeat)
