package examples.bgl_v2.mg-src.for_loop;

concept ForIteratorLoop = {
    require type Context;
    require type State;
    require type Iterator;

    require procedure iterNext(upd itr: Iterator);
    require procedure step(obs itr: Iterator,
                           obs endItr: Iterator, // maybe not necessary?
                           upd state: State,
                           obs ctx: Context);
    
    procedure forLoopRepeat(obs itr: Iterator,
                            obs endItr: Iterator,
                            upd state: State,
                            obs ctx: Context);

    axiom forIterationEnd(itr: Iterator, endItr: Iterator,
                          state: State, ctx: Context) {
        var mut_state = state;

        if itr == endItr
        then {
            call forLoopRepeat(itr, endItr, mut_state, ctx);
            assert mut_state == state;
        } else skip;
    }
}


concept ForIteratorLoop3_2 = {
    require type Context1;
    require type Context2;
    require type State1;
    require type State2;
    require type State3;

    require type Iterator;

    require procedure iterNext(upd itr: Iterator);
    require procedure step(obs itr: Iterator,
                           obs endItr: Iterator,
                           upd s1: State1,
                           upd s2: State2,
                           upd s3: State3,
                           obs ctx1: Context1,
                           obs ctx2: Context2);

    procedure forLoopRepeat(obs itr: Iterator,
                            obs endItr: Iterator,
                            upd s1: State1,
                            upd s2: State2,
                            upd s3: State3,
                            obs ctx1: Context1,
                            obs ctx2: Context2);
}

concept ForIteratorLoop3_1 = {
    require type Context1;
    require type State1;
    require type State2;
    require type State3;

    require type Iterator;

    require procedure iterNext(upd itr: Iterator);
    require procedure step(obs itr: Iterator,
                           obs endItr: Iterator,
                           upd s1: State1,
                           upd s2: State2,
                           upd s3: State3,
                           obs ctx1: Context1);

    procedure forLoopRepeat(obs itr: Iterator,
                            obs endItr: Iterator,
                            upd s1: State1,
                            upd s2: State2,
                            upd s3: State3,
                            obs ctx1: Context1);
}
