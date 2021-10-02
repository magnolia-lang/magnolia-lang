package examples.bgl_v2.mg-src.while_loop;

concept WhileLoop = {
    require type Context;
    require type State;

    require predicate cond(s: State, c: Context);
    require procedure step(upd s: State, obs c: Context);
    procedure repeat(upd s: State, obs c: Context);

    axiom whileLoopBehavior(s: State, c: Context) {
        var mutableState = s;
        if cond(s, c) then {
            // if condition holds, then one loop iteration occurs
            var mutableState1 = s;
            var mutableState2 = s;
            call repeat(mutableState1, c);
            call step(mutableState2, c);
            assert mutableState1 == mutableState2;
        }
        else {
            // otherwise, the state shouldn't change
            var mutableState1 = s;
            call repeat(mutableState1, c);
            assert mutableState1 == s;
        };
    }
};
