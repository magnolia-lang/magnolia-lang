package examples.moa.mg-src.while-loops;

implementation WhileLoop7_2 =
	external C++ base.while_loop7_2 {
		require type Context1;
		require type Context2;
		require type Context3;
		require type Context4;
		require type Context5;
		require type Context6;
		require type Context7;
		require type State1;
		require type State2;

		require predicate cond(context1: Context1, context2: Context2, context3: Context3, context4: Context4, context5: Context5, context6: Context6, context7: Context7, state1: State1, state2: State2);
		require procedure body(obs context1: Context1, obs context2: Context2, obs context3: Context3, obs context4: Context4, obs context5: Context5, obs context6: Context6, obs context7: Context7, upd state1: State1, upd state2: State2);
		procedure repeat(obs context1: Context1, obs context2: Context2, obs context3: Context3, obs context4: Context4, obs context5: Context5, obs context6: Context6, obs context7: Context7, upd state1: State1, upd state2: State2);
};

implementation WhileLoop6_2 =
	external C++ base.while_loop6_2 {
		require type Context1;
		require type Context2;
		require type Context3;
		require type Context4;
		require type Context5;
		require type Context6;
		require type State1;
		require type State2;

		require predicate cond(context1: Context1, context2: Context2, context3: Context3, context4: Context4, context5: Context5, context6: Context6, state1: State1, state2: State2);
		require procedure body(obs context1: Context1, obs context2: Context2, obs context3: Context3, obs context4: Context4, obs context5: Context5, obs context6: Context6, upd state1: State1, upd state2: State2);
		procedure repeat(obs context1: Context1, obs context2: Context2, obs context3: Context3, obs context4: Context4, obs context5: Context5, obs context6: Context6, upd state1: State1, upd state2: State2);
};

implementation WhileLoop5_2 =
	external C++ base.while_loop5_2 {
		require type Context1;
		require type Context2;
		require type Context3;
		require type Context4;
		require type Context5;
		require type State1;
		require type State2;

		require predicate cond(context1: Context1, context2: Context2, context3: Context3, context4: Context4, context5: Context5, state1: State1, state2: State2);
		require procedure body(obs context1: Context1, obs context2: Context2, obs context3: Context3, obs context4: Context4, obs context5: Context5, upd state1: State1, upd state2: State2);
		procedure repeat(obs context1: Context1, obs context2: Context2, obs context3: Context3, obs context4: Context4, obs context5: Context5, upd state1: State1, upd state2: State2);
};

implementation WhileLoop2_3 =
	external C++ base.while_loop2_3 {
		require type Context1;
		require type Context2;
		require type State1;
		require type State2;
		require type State3;

		require predicate cond(context1: Context1, context2: Context2, state1: State1, state2: State2, state3: State3);
		require procedure body(obs context1: Context1, obs context2: Context2, upd state1: State1, upd state2: State2, upd state3: State3);
		procedure repeat(obs context1: Context1, obs context2: Context2, upd state1: State1, upd state2: State2, upd state3: State3);
};

implementation WhileLoop2_2 =
	external C++ base.while_loop2_2 {
		require type Context1;
		require type Context2;
		require type State1;
		require type State2;

		require predicate cond(context1: Context1, context2: Context2, state1: State1, state2: State2);
		require procedure body(obs context1: Context1, obs context2: Context2, upd state1: State1, upd state2: State2);
		procedure repeat(obs context1: Context1, obs context2: Context2, upd state1: State1, upd state2: State2);
};


implementation WhileLoop1_2 =
	external C++ base.while_loop1_2 {
		require type Context1;
		require type State1;
		require type State2;

		require predicate cond(context1: Context1, state1: State1, state2: State2);
		require procedure body(obs context1: Context1, upd state1: State1, upd state2: State2);
		procedure repeat(obs context1: Context1, upd state1: State1, upd state2: State2);
};

implementation WhileLoop1_1 =
	external C++ base.while_loop1_1 {
		require type Context1;
		require type State1;

		require predicate cond(context1: Context1, state1: State1);
		require procedure body(obs context1: Context1, upd state1: State1);
		procedure repeat(obs context1: Context1, upd state1: State1);
};