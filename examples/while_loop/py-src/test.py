from while_loop import *

def main():
    for cls in [Int16, Int32]:
        state, bound = cls(0), cls(65537)
        repeat(state, bound)
        print(f'With class {cls.__name__}:', state)

if __name__ == '__main__':
    main()
