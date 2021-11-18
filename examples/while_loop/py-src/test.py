from gen_while_loop import IterationProgram

def main():
    p = IterationProgram()
    for cls in [p.Int16, p.Int32]:
        state, bound = cls(0), cls(65537)
        p.repeat(state, bound)
        print(f'With class {cls.__name__}:', state)

if __name__ == '__main__':
    main()
