# Implementation of overloaded functions in Python adapted from the example
# given by Guido van Rossum at https://www.artima.com/weblogs/viewpost.jsp?thread=101605.
# This file must (at the moment) be provided at the root of the project.
from collections import defaultdict, namedtuple

FunctionAndReturnType = namedtuple('FunctionAndReturnType',
                                   ['function', 'return_type'])

class OverloadedFunction:
    def __init__(self, name):
        self.name = name
        self.dispatch_map = defaultdict(list)

    def __call__(self, *args, return_type=None):
        arg_types = tuple(arg.__class__ for arg in args)

        def _find_matches():
            all_matches = self.dispatch_map.get(arg_types)
            filtered_matches = (
                all_matches
                if return_type is None or all_matches is None
                else list(filter(lambda m: m.return_type == return_type,
                                 all_matches)))
            return filtered_matches

        pretty_args = str(tuple(map(lambda at: at.__name__, arg_types)))

        matches = _find_matches()
        fun = None
        if matches is None or len(matches) == 0:
            raise TypeError(
                f"couldn't find a match for function {self.name} with "
                f"argument types {pretty_args}")
        elif len(matches) == 1:
            fun = matches[0].function
        else: # len(matches) > 1
            print(matches)
            raise TypeError(
                 "couldn't disambiguate between several matches in call to "
                f"function {self.name} with argument types {pretty_args}: "
                f"got {matches}")
        
        return fun(*args)

    def register(self, function, arg_types, return_type):
        self.dispatch_map[arg_types].append(
            FunctionAndReturnType(function, return_type))

# TODO: function name is not enough here to differentiate between various
#       implementations. We will perhaps need to pass a scope name parameter
#       as well, to namespace things properly. I think the better solution
#       is to define a map for overloading in each module locally. That way,
#       the different modules do not conflict with each other.
def overload(_overloaded_functions_dict, *types, return_type):
    def register(function):
        name = function.__name__
        overloaded_func = _overloaded_functions_dict.get(name)
        if overloaded_func is None:
            overloaded_func = OverloadedFunction(name)
            _overloaded_functions_dict[name] = overloaded_func
        overloaded_func.register(function, types, return_type=return_type)
        return overloaded_func
    return register
