package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.while-loops;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

signature forall_ops = {

    type Context;
    type ContextContainer; // "variadics"
    type IndexedContainer;
    type A;
    type B;

    procedure forall_ix(obs ctx: ContextContainer,
                        obs ixContainer: IndexedContainer,
                        out ixContainer2: IndexedContainer);

    function forall_ix_action(ctx: ContextContainer, a: A): B;
}

implementation BasicOps = external C++ base.matrix {

    require type Integer;

    type Matrix;
    type Index;
    type Shape;
    type Size;

    // access
    procedure set(obs m: Matrix, obs i: Integer, obs j: Integer, obs e: Integer);
    function get(m: Matrix, i: Index): Matrix;
    function shape(m: Matrix): Shape;
    function size(m: Matrix): Size;
    function access_shape(m: Matrix, s: Size): Size;

    // Matrix creation
    function create_matrix(i: Size, j: Size): Matrix;
    function test_vector(): Matrix;
    function test_matrix(): Matrix;
    function test_partial_index(): Index;
    function test_total_index(): Index;
    function create_singleton_index(i: Integer): Index;
    function zeros(i: Size, j: Size): Matrix;

    // IO
    procedure print_matrix(obs m: Matrix);
    procedure print_shape(obs m: Matrix);
    procedure print_number(obs e: Integer);

    // Transformations
    function transpose(m: Matrix): Matrix;

    // Util
    function unwrap_scalar(m: Matrix): Integer;
    function sizeToInteger(s: Size): Integer;
    function integerToSize(i: Integer): Size;


}

implementation MatMult = {

    use BasicOps;

    require function add(a: Integer, b: Integer): Integer;
    require function mult(a: Integer, b: Integer): Integer;
    require function zero(): Integer;
    require function one(): Integer;
    require predicate isLowerThan(a: Integer, b: Integer);

    predicate upperBoundMulElem(m1: Matrix, m2: Matrix, res: Matrix, counter: Integer) = {
        value isLowerThan(counter, sizeToInteger(size(m1)));
    }

    procedure mult_elementwise(obs a: Matrix, obs b: Matrix, upd res: Matrix, upd counter: Integer) = {
        var current_index = create_singleton_index(counter);
        var new_value = mult(unwrap_scalar(get(a, current_index)),
                             unwrap_scalar(get(b, current_index)));
        call set(res, zero(), counter, new_value);
        counter = add(counter, one());
    }

    use WhileLoop2_2[Context1 => Matrix,
                     Context2 => Matrix,
                     State1 => Matrix,
                     State2 => Integer,
                     body => mult_elementwise,
                     cond => upperBoundMulElem,
                     repeat => vecmult];


    procedure sum_vector(obs a: Matrix, upd res: Integer, upd counter: Integer) = {
        var current_index = create_singleton_index(counter);
        res = add(res, unwrap_scalar(get(a, current_index)));
        counter = add(counter, one());
    }

    predicate upperBoundSum (m: Matrix, res: Integer, counter: Integer) = {
        value isLowerThan(counter, sizeToInteger(size(m)));
    }

    use WhileLoop1_2[Context1 => Matrix,
                     State1 => Integer,
                     State2 => Integer,
                     body => sum_vector,
                     cond => upperBoundSum,
                     repeat => mapsum];

    predicate upperBoundMatMult (m1: Matrix, m2: Matrix, res: Matrix,
                                 i: Integer, k: Integer) = {

        var i_dim = sizeToInteger(access_shape(res, integerToSize(zero())));
        var k_dim = sizeToInteger(access_shape(res, integerToSize(one())));

        value isLowerThan(i, i_dim) && isLowerThan(k, k_dim);

    }

    procedure doMatMult(obs m1: Matrix, obs m2: Matrix, upd resM: Matrix, upd i: Integer, upd k: Integer) {

        var slice1 = get(m1, create_singleton_index(i));
        var slice2 = get(m2, create_singleton_index(k));

        var i_dim = sizeToInteger(access_shape(resM, integerToSize(zero())));
        var k_dim = sizeToInteger(access_shape(resM, integerToSize(one())));

        var result_slice = zeros(integerToSize(one()), integerToSize(k_dim));
        var vecmultC = zero();
        call vecmult(slice1, slice2, result_slice, vecmultC);

        var reduced = zero();
        var mapsumC = zero();
        call mapsum(result_slice, reduced, mapsumC);

        call set(resM, i, k, reduced);

        if isLowerThan(add(k, one()), k_dim) then {
            k = add(k, one());
        } else {
            i = add(i, one());
            k = zero();
        };

    }
}

implementation MatMultImpl = {
    use MatMult;

    use WhileLoop2_3 [Context1 => Matrix,
                      Context2 => Matrix,
                      State1 => Matrix,
                      State2 => Integer,
                      State3 => Integer,
                      cond => upperBoundMatMult,
                      body => doMatMult,
                      repeat => matmult];

}

signature IntegerOps = {
    type Integer;

    function zero(): Integer;
    function one(): Integer;
    function add(a: Integer, b: Integer): Integer;
    function mult(a: Integer, b: Integer): Integer;

    predicate isLowerThan(a: Integer, b: Integer);
}

implementation Int32Utils = external C++ base.int32_utils
    IntegerOps[Integer => Int32];

program ArrayProgram = {

    use Int32Utils;
    use MatMultImpl[Integer => Int32];

}