import sys

def while_loop_code(path_to_struct, nb_ctx, nb_state):
    ctx_vars = [f'Context{i}' for i in range(1, nb_ctx + 1)]
    state_vars = [f'State{i}' for i in range(1, nb_state + 1)]

    struct_name = f'while_loop{nb_ctx}_{nb_state}'

    def magnolia_code():
        module = []
        p = module.append
        p(f'implementation WhileLoop{nb_ctx}_{nb_state} = ')
        p(f'\texternal C++ {path_to_struct}.{struct_name}' + ' {')
        for v in ctx_vars + state_vars:
            p(f'\t\trequire type {v};')
        p('')
        cond_params = ', '.join(f'{v.lower()}: {v}'
                                for v in ctx_vars + state_vars)
        p(f'\t\trequire predicate cond({cond_params});')
        proc_params = ', '.join([f'obs {v.lower()}: {v}' for v in ctx_vars] + \
                                [f'upd {v.lower()}: {v}' for v in state_vars])
        p(f'\t\trequire procedure body({proc_params});')
        p(f'\t\tprocedure repeat({proc_params});')
        p('};')
        return '\n'.join(module)

    def cpp_code():
        struct = []
        p = struct.append
        p('template <typename ' + \
          ', typename '.join(f'_{v}' for v in ctx_vars + state_vars) + \
          ', class _body, class _cond>')
        p(f'struct {struct_name}' + ' {')
        for v in ctx_vars + state_vars:
            p(f'\ttypedef _{v} {v};')
        p('')
        p('\t_body body;')
        p('\t_cond cond;')
        p('\tvoid repeat(' + \
          ', '.join([f'const {v} &{v.lower()}' for v in ctx_vars] + \
                    [f'{v} &{v.lower()}' for v in state_vars]) + ') {')
        args = ', '.join(f'{v.lower()}' for v in ctx_vars + state_vars)
        p(f'\t\twhile (cond({args})) body({args});')
        p('\t}')
        p('};')
        return '\n'.join(struct)

    return '\n\n'.join(['/* paste in Magnolia file */',
                        magnolia_code(),
                        '/* paste in C++ file */',
                        cpp_code()])


if __name__ == '__main__':
    assert len(sys.argv) == 4, (
        'Usage: {} <path/to/struct/file> <nb context types> <nb state types>'.format(sys.argv[0]))
    path_to_struct, nb_ctx, nb_state = sys.argv[1:]
    nb_ctx, nb_state = int(nb_ctx), int(nb_state)
    assert nb_ctx > 0 and nb_state > 0
    print(while_loop_code(path_to_struct, nb_ctx, nb_state))