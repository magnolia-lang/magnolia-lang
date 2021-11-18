from collections import namedtuple

def while_ops(_Context, _State, _cond, _body):
    def body(state, context):
        return _body(state, context)
    
    def cond(state, context):
        return _cond(state, context)

    def repeat(state, context):
        while cond(state, context):
            body(state, context)

    while_ops_tuple = namedtuple('while_ops',
                                 ['State', 'Context', 'body', 'cond', 'repeat'])
    return while_ops_tuple(_State, _Context, body, cond, repeat)
