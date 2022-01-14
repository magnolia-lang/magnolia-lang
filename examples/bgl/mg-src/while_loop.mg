package examples.bgl.mg-src.while_loop;

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

concept WhileLoop4_3 = {
    require type Context1;
    require type Context2;
    require type Context3;
    require type State1;
    require type State2;
    require type State3;
    require type State4;

    require predicate cond(s1: State1, s2: State2, s3: State3, s4: State4, ctx1: Context1, ctx2: Context2, ctx3: Context3);
    require procedure step(upd s1: State1, upd s2: State2, upd s3: State3, upd s4: State4, obs ctx1: Context1, obs ctx2: Context2, obs ctx3: Context3);
    procedure repeat(upd s1: State1, upd s2: State2, upd s3: State3, upd s4: State4, obs ctx1: Context1, obs ctx2: Context2, obs ctx3: Context3);
};

concept WhileLoop3 = {
    require type Context;
    require type State1;
    require type State2;
    require type State3;

    require predicate cond(s1: State1, s2: State2, s3: State3, ctx: Context);
    require procedure step(upd s1: State1, upd s2: State2, upd s3: State3, obs ctx: Context);
    procedure repeat(upd s1: State1, upd s2: State2, upd s3: State3, obs ctx: Context);
};

concept Mergeable = {
    type T;

    function merge(t1: T, t2: T): T;
};

concept DistributedWhileLoop4_2 = {

};
