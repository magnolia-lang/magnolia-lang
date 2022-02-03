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

signature IntegerOps = {
    type Integer;

    function zero(): Integer;
    function one(): Integer;
    function add(a: Integer, b: Integer): Integer;
    function sub(a: Integer, b: Integer): Integer;

    function mult(a: Integer, b: Integer): Integer;

    predicate equals(a: Integer, b: Integer);
    predicate isLowerThan(a: Integer, b: Integer);
}


implementation ExtOps = external C++ base.matrix {

    require type Integer;

    type Matrix;
    type Index;
    type Shape;
    type Size;

    // Access
    procedure set(upd m: Matrix, obs i: Integer, obs j: Integer, obs e: Integer);
    function get(m: Matrix, i: Index): Matrix;
    function shape(m: Matrix): Shape;
    function size(m: Matrix): Size;
    function size(s: Shape): Size;
    function access_shape(m: Matrix, s: Size): Size;

    // Matrix creation
    function create_matrix(i: Size, j: Size): Matrix;
    function single_I(i: Integer): Index;
    function create_index(i: Integer, j: Integer): Index;
    function zeros(i: Size, j: Size): Matrix;

    // IO
    procedure print_index(obs i: Index);
    procedure print_matrix(obs m: Matrix);
    procedure print_number(obs e: Integer);
    procedure print_shape(obs m: Matrix);


    // Transformations
    function transpose(m: Matrix): Matrix;

    // Util
    function unwrap_scalar(m: Matrix): Integer;
    function sizeToInteger(s: Size): Integer;
    function integerToSize(i: Integer): Size;

    // Tests
    function test_vector(): Matrix;
    function test_matrix(): Matrix;
    function test_matrix2(): Matrix;
    function test_partial_index(): Index;
    function test_total_index(): Index;
}

implementation MoaOps = {

    use ExtOps;
    use IntegerOps;

}

implementation CatImpl = {

    use MoaOps;

    predicate catUpperBound(m1: Matrix, m2: Matrix, res: Matrix, c1: Integer, c2: Integer) = {

        var row_upper = sizeToInteger(access_shape(res, integerToSize(zero())));
        value isLowerThan(c1, row_upper);
    }

    procedure catBody(obs m1: Matrix, obs m2: Matrix, upd res: Matrix, upd c1: Integer, upd c2: Integer) = {

        var m1_row_bound = sizeToInteger(access_shape(m1, integerToSize(one())));
        var res_row_bound = sizeToInteger(access_shape(res, integerToSize(one())));
        if isLowerThan(c2, m1_row_bound) then {

            var ix = create_index(c1, c2);
            call set(res, c1, c2, unwrap_scalar(get(m1, ix)));

            c2 = add(c2, one());
        } else {

            var ix: Index;

            if equals(sizeToInteger(size(shape(m2))), one()) then {
               // call print_number(c1);
                ix = single_I(sub(c2, m1_row_bound));

                call set(res, c1, c2, unwrap_scalar(get(m2, ix)));

            } else {
                ix = create_index(c1,sub(c2, m1_row_bound));
                call set(res, c1, c2, unwrap_scalar(get(m2, ix)));
            };

            if isLowerThan(c2, sub(res_row_bound, one())) then {
                c2 = add(c2, one());
            } else {

                c1 = add(c1, one());
                c2 = zero();

            };

        };

    }

    use WhileLoop2_3[Context1 => Matrix,
                     Context2 => Matrix,
                     State1 => Matrix,
                     State2 => Integer,
                     State3 => Integer,
                     body => catBody,
                     cond => catUpperBound,
                     repeat => doCat];


    function cat(m1: Matrix, m2: Matrix): Matrix
        guard
            equals(sizeToInteger(access_shape(m1, integerToSize(zero()))),
                   sizeToInteger(access_shape(m2, integerToSize(zero())))) = {

        var x_dim = access_shape(m1, integerToSize(zero()));
        var y_dim: Size;

        if equals(sizeToInteger(size(shape(m2))), one()) then {
            y_dim = integerToSize(
                    add(sizeToInteger(access_shape(m1, integerToSize(one()))),
                        one()));
        } else {
            y_dim = integerToSize(
                    add(sizeToInteger(access_shape(m1, integerToSize(one()))),
                        sizeToInteger(access_shape(m2, integerToSize(one())))));
        };

        var res = zeros(x_dim, y_dim);

        var c1 = zero();
        var c2 = zero();

        call doCat(m1, m2, res, c1, c2);


        value res;
    }
}

implementation Padding = {

    use CatImpl;

    // guard equals(access_shape(m1, one()), access_shape(m2, one()))

    /*
    circular padding on axis i

    */
    function cPadr(m: Matrix, i: Integer): Matrix = {

        var slice = get(m, single_I(i));

        //call print_shape(slice);


        var pRes = transpose(cat(transpose(m), slice));

        value pRes;

    }
    function cPadl(m: Matrix, i: Integer): Matrix = {

        var slice = get(m, single_I(i));

        var pRes = transpose(cat(slice, transpose(m)));

        value pRes;
    }

}

implementation MatMult = {

    use MoaOps;

    require function add(a: Integer, b: Integer): Integer;
    require function sub(a: Integer, b: Integer): Integer;
    require function mult(a: Integer, b: Integer): Integer;
    require function zero(): Integer;
    require function one(): Integer;
    require predicate isLowerThan(a: Integer, b: Integer);

    predicate upperBoundMulElem(m1: Matrix, m2: Matrix, res: Matrix, counter: Integer) = {
        value isLowerThan(counter, sizeToInteger(size(m1)));
    }

    procedure mult_elementwise(obs a: Matrix,
                               obs b: Matrix,
                               upd res: Matrix,
                               upd counter: Integer) = {

        var current_index = single_I(counter);
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


    procedure sum_vector(obs a: Matrix,
                         upd res: Integer,
                         upd counter: Integer) = {

        var current_index = single_I(counter);
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

    procedure iterMatMult(obs m1: Matrix,
                        obs m2: Matrix,
                        upd resM: Matrix,
                        upd i: Integer,
                        upd k: Integer) {

        var slice1 = get(m1, single_I(i));
        var slice2 = get(m2, single_I(k));

        var i_dim = sizeToInteger(access_shape(resM, integerToSize(zero())));
        var k_dim = sizeToInteger(access_shape(resM, integerToSize(one())));

        var result_slice = zeros(integerToSize(one()), integerToSize(i_dim));
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
                      body => iterMatMult,
                      repeat => doMatmult];

    function matmult(m1: Matrix, m2: Matrix): Matrix = {

        var x_dim = access_shape(m1, integerToSize(zero()));
        var y_dim = access_shape(m2, integerToSize(one()));
        var res = zeros(x_dim, y_dim);

        var c1 = zero();
        var c2 = zero();

        call doMatmult(m1, transpose(m2), res, c1, c2);

        value res;
    }

}

implementation Int32Utils = external C++ base.int32_utils
    IntegerOps[Integer => Int32];

program ArrayProgram = {

    use Int32Utils;
    use MatMultImpl[Integer => Int32];
    use Padding[Integer => Int32];

}