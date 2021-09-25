import importlib
while_loop = importlib.import_module('gen.examples.while_loop.mg-src.while_loop-py')

IterationProgram = while_loop.IterationProgram()

def while_loop_test():
    #p = IterationProgram()
    for cls in [IterationProgram.Int16, IterationProgram.Int32]:
        state, bound = cls(0), cls(65537)
        IterationProgram.repeat(state, bound)
        print(f'With class {cls.__name__}:', state)

if __name__ == '__main__':
    while_loop_test()
