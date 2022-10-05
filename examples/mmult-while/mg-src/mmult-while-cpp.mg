package examples.mmult-while.mg-src.mmult-while-cpp;

/*
* MatMult example using loops
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

signature IntegerOps = {
    type Integer;

    function zero(): Integer;
    function one(): Integer;
    function add(a: Integer, b: Integer): Integer;
    function mult(a: Integer, b: Integer): Integer;

    predicate isLowerThan(a: Integer, b: Integer);

}

implementation BasicOps = external C++ base.matrix {

    require type Integer;

    type Matrix;
    type Index;
    type Shape;
    type Size;

    // access
    procedure set(obs m: Matrix, obs i: Index, obs j: Index, obs e: Integer);
    function get(m: Matrix, i: Index, j: Index): Integer;

    // Matrix creation
    function create_matrix(i: Size, j: Size): Matrix;
    function test_matrix(): Matrix;
    function zeros(i: Size, j: Size): Matrix;

    // IO
    procedure print_matrix(obs m: Matrix);
    procedure print_shape(obs m: Matrix);
    procedure print_number(obs e: Integer);

    // Util
    function indexToInteger(i: Index): Integer;
    function integerToIndex(i: Integer): Index;

}

//***********matrix mult***************************
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


implementation MatMult = {

    use BasicOps;

    require function add(a: Integer, b: Integer): Integer;
    require function mult(a: Integer, b: Integer): Integer;
    require function one(): Integer;
    require predicate isLowerThan(a: Integer, b: Integer);

    predicate isLowerThanOuter(m1: Matrix,
                               m2: Matrix,
                               outerBound: Integer, b: Integer, c: Integer, d: Integer,
                               e: Integer, outerCounter: Integer, mres: Matrix) {
        value isLowerThan(outerCounter,outerBound);
    }

    predicate isLowerThanMid(m1: Matrix,
                             m2: Matrix,
                             middleBound: Integer, b: Integer, c: Integer, d: Integer, middleCounter: Integer, mres: Matrix) {
        value isLowerThan(middleCounter,middleBound);
    }

    predicate isLowerThanInner(m1: Matrix,
                               m2: Matrix,
                               innerBound: Integer, b: Integer, c: Integer, innerCounter: Integer, mres : Matrix) {
        value isLowerThan(innerCounter,innerBound);
    }


    procedure innerLoop(obs m1: Matrix,
                        obs m2: Matrix,
                        obs innerBound: Integer,
                        obs outerCounter: Integer,
                        obs middleCounter: Integer,
                        upd innerCounter: Integer,
                        upd mres: Matrix) = {

        var x = integerToIndex(outerCounter);
        var y = integerToIndex(middleCounter);
        var z = integerToIndex(innerCounter);

        var m1_elem = get(m1, x, z);
        var m2_elem = get(m2, z, y);

        call set(mres, x, y,
                add(get(mres, x, y), mult(m1_elem, m2_elem)));

        innerCounter = add(innerCounter, one());
    }

    use WhileLoop5_2[Context1 => Matrix,
                     Context2 => Matrix,
                     Context3 => Integer,
                     Context4 => Integer,
                     Context5 => Integer,
                     State1 => Integer,
                     State2 => Matrix,
                     cond => isLowerThanInner,
                     body => innerLoop,
                     repeat => repeatInner];


    procedure middleLoop(obs m1: Matrix,
                         obs m2: Matrix,
                         obs middleBound: Integer,
                         obs innerBound: Integer,
                         obs innerCounter: Integer,
                         obs outerCounter: Integer,
                         upd middleCounter: Integer,
                         upd mres: Matrix) {

        var innerCounter_upd = value innerCounter;

        call repeatInner(m1, m2,
                         innerBound, outerCounter, middleCounter, innerCounter_upd,
                         mres);
        middleCounter = add(middleCounter, one());
    }

    use WhileLoop6_2 [Context1 => Matrix,
                      Context2 => Matrix,
                      Context3 => Integer,
                      Context4 => Integer,
                      Context5 => Integer,
                      Context6 => Integer,
                      State1 => Integer,
                      State2 => Matrix,
                      cond => isLowerThanMid,
                      body => middleLoop,
                      repeat => repeatMiddle];

    procedure outerLoop(obs m1: Matrix,
                        obs m2: Matrix,
                        obs outerBound: Integer,
                        obs middleBound: Integer,
                        obs innerBound: Integer,
                        obs middleCounter: Integer,
                        obs innerCounter: Integer,
                        upd outerCounter: Integer,
                        upd mres: Matrix) = {

        var middleCounter_upd = value middleCounter;

        call repeatMiddle(m1, m2,
                         middleBound, innerBound,
                         innerCounter, outerCounter, middleCounter_upd, mres);
        outerCounter = add(outerCounter, one());
    }

    use WhileLoop7_2[Context1 => Matrix,
                     Context2 => Matrix,
                     Context3 => Integer,
                     Context4 => Integer,
                     Context5 => Integer,
                     Context6 => Integer,
                     Context7 => Integer,
                     State1 => Integer,
                     State2 => Matrix,
                     cond => isLowerThanOuter,
                     body => outerLoop];

}

implementation Int32Utils = external C++ base.int32_utils
    IntegerOps[Integer => Int32];

program ArrayProgram = {

    use Int32Utils;
    use MatMult[Integer => Int32];

}