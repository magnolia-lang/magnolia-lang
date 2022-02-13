package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.while-loops;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

signature forall_ops = {

    type Context;
    type IndexedContainer;
    type A;
    type B;

    procedure forall_ix(obs ctx1: Context, obs ctx2: Context,
                        obs ixContainer: IndexedContainer,
                        out ixContainer2: IndexedContainer);

    function forall_ix_action(ctx1: Context, ctx2: Context, a: A): B;
}

signature NumberOps = {
    type NumberType;

    function zero(): NumberType;
    function one(): NumberType;
    function add(a: NumberType, b: NumberType): NumberType;
    function sub(a: NumberType, b: NumberType): NumberType;

    function mult(a: NumberType, b: NumberType): NumberType;

    predicate equals(a: NumberType, b: NumberType);
    predicate isLowerThan(a: NumberType, b: NumberType);
}


implementation ExtOps = external C++ base.array {

    require type Element;

    type PaddedArray;
    type Array;
    type Index;
    type Shape;
    type UInt32;

    // unpadded array getters, setters
    function get(a: Array, ix: UInt32): Array;
    function get(a: Array, ix: Index): Array;

    procedure set(upd a: Array, obs ix: UInt32, obs e: Element);
    procedure set(upd a: Array, obs ix: Index, obs e: Element);

    function get_shape_elem(a: Array, i: UInt32): UInt32;
    function drop_shape_elem(a: Array, i: UInt32): Shape;

    // unpadded moa operations that are convenient to have tied to backend
    function dim(a: Array): UInt32;
    function total(a: Array): UInt32;
    function shape(a: Array): Shape;


    // padded array getters
    //function padded_get(a: PaddedArray, ix: Index): PaddedArray;
    //procedure padded_set(upd a: PaddedArray, obs ix: Index, obs e: Element);
    function padded_get_shape_elem(a: PaddedArray, i: UInt32): UInt32;
    function padded_drop_shape_elem(a: PaddedArray, i: UInt32): Shape;


    // padded moa operations
    function padded_dim(a: PaddedArray): UInt32;
    function padded_total(a: PaddedArray): UInt32;
    function padded_shape(a: PaddedArray): Shape;

    // creation
    function create_array(sh: Shape): Array;
    function create_shape1(a: UInt32): Shape;
    function create_shape2(a: UInt32, b: UInt32): Shape;
    function create_shape3(a: UInt32, b: UInt32, c: UInt32): Shape;

    function create_index1(a: UInt32): Index;
    function create_index2(a: UInt32, b: UInt32): Index;
    function create_index3(a: UInt32, b: UInt32, c: UInt32): Index;

    // util
    function unwrap_scalar(a: Array): Element;
    function uint_elem(a: UInt32): Element;
    function elem_uint(a: Element): UInt32;
    function cat_shape(a: Shape, b: Shape): Shape;
    function reverse_index(ix: Index): Index;

    // IO
    procedure print_array(obs a: Array);
    procedure print_index(obs i: Index);
    procedure print_shape(obs sh: Shape);
    procedure print_element(obs e: Element);
    procedure print_uint(obs u: UInt32);

    // testing
    function test_vector2(): Array;
    function test_vector3(): Array;
    function test_vector5(): Array;

    function test_array3_2_2(): Array;
    function test_array3_3(): Array;
    function test_index(): Index;
    function create_padded_array(unpadded_shape: Shape, padded_shape: Shape,
                          padded_array: Array): PaddedArray;
}

implementation Catenation = {

    require function zero(): Element;
    require function one(): Element;
    require function add(a: Element, b: Element): Element;
    require function sub(a: Element, b: Element): Element;
    require function mult(a: Element, b: Element): Element;
    require predicate equals(a: Element, b: Element);
    require predicate isLowerThan(a: Element, b: Element);

    use ExtOps;

    /*########################################
        Vector catenation, i.e. cat(v1, v2)
      ########################################
    */

    /*
    cat_vec_body performs one iteration of putting the element currently
    indexed in its correct position in the catenated vector
    */
    procedure cat_vec_body(obs v1: Array,
                           obs v2: Array,
                           upd res: Array,
                           upd counter: UInt32) {

        var v1_bound = uint_elem(total(v1));
        var ix: Index;

        // conditional determining if we should access v1 or v2
        if isLowerThan(uint_elem(counter), uint_elem(total(v1))) then {
            ix = create_index1(counter);
            call set(res, ix, unwrap_scalar(get(v1, ix)));
        } else {
            ix = create_index1(elem_uint(sub(uint_elem(counter), v1_bound)));
            var res_ix = create_index1(counter);
            call set(res, res_ix, unwrap_scalar(get(v2, ix)));
        };

        counter = elem_uint(add(uint_elem(counter), one()));
    }

    // determines upper bound for the iterator
    predicate cat_vec_cond(v1: Array, v2: Array, res: Array, counter: UInt32) {
        value isLowerThan(uint_elem(counter), uint_elem(total(res)));
    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => Array,
                     State1 => Array,
                     State2 => UInt32,
                     body => cat_vec_body,
                     cond => cat_vec_cond,
                     repeat => cat_vec_repeat];


    /*
    cat_vec takes two vectors as inputs, does a shape analysis, and then calls the cat_vec_repeat procedure with a correctly shaped updatable result vector argument
    */
    function cat_vec(vector1: Array, vector2: Array): Array
        guard dim(vector1) == elem_uint(one()) && dim(vector2) == elem_uint(one()) {

        var res_shape = create_shape1(elem_uint(add(
                                          uint_elem(total(vector1)),
                                          uint_elem(total(vector2)))));

        var res = create_array(res_shape);
        var counter = elem_uint(zero());
        call cat_vec_repeat(vector1, vector2, res, counter);

        value res;

    }

    /*########################################
        Array catenation, i.e. cat(a1, a2)
      ########################################
    */
    procedure cat_body(obs array1: Array,
                       obs array2: Array,
                       upd counter: UInt32,
                       upd res: Array) {

        var s_0 = uint_elem(total(array1));

        if isLowerThan(uint_elem(counter), s_0) then {
            call set(res, counter,
                    unwrap_scalar(get(array1,counter)));
            counter = elem_uint(add(uint_elem(counter), one()));
        } else {
            var ix = elem_uint(sub(uint_elem(counter), s_0));
            call set(res, counter,
                unwrap_scalar(get(array2, ix)));
            counter = elem_uint(add(uint_elem(counter), one()));
        };
    }

    predicate cat_cond(array1: Array, array2: Array, counter: UInt32, res: Array) {
        var upper_bound = uint_elem(total(res));
        value isLowerThan(uint_elem(counter), upper_bound);
    }
    use WhileLoop2_2[Context1 => Array,
                     Context2 => Array,
                     State1 => UInt32,
                     State2 => Array,
                     body => cat_body,
                     cond => cat_cond,
                     repeat => cat_repeat];


    /*
    cat takes two arrays as inputs, does a shape analysis, and then calls the cat_repeat procedure with a correctly shaped updatable result array argument
    */

    /*
    TODO: IN ALL LITERATURE, CAT IS DEFINED IN TERMS OF PARTIAL INDEXING, what

    */
    function cat(array1: Array, array2: Array): Array
        guard drop_shape_elem(array1, elem_uint(zero())) ==
              drop_shape_elem(array2, elem_uint(zero())) {

        /*
        shape of the catenated array is given by:
        (take(1, a1) + take(1, a2)) 'cat' drop(1, a1)
        */
        var take_a1 = uint_elem(get_shape_elem(array1, elem_uint(zero())));
        var take_a2 = uint_elem(get_shape_elem(array2, elem_uint(zero())));
        var drop_a1 = drop_shape_elem(array1, elem_uint(zero()));

        var result_shape = cat_shape(create_shape1(elem_uint(add(take_a1, take_a2))), drop_a1);

        var res = create_array(result_shape);

        var counter = elem_uint(zero());
        call cat_repeat(array1, array2, counter, res);

        value res;
    }
}

implementation Padding = {

    use Catenation;

    function padr(a: Array, ix: UInt32): PaddedArray = {

        var padding = get(a, create_index1(ix));
        var catenated_array = cat(a, padding);
        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;

    }
}

implementation Int32Utils = external C++ base.int32_utils
    NumberOps[NumberType => Int32];

implementation Float64Utils = external C++ base.float64_utils
    NumberOps[NumberType => Float64];


program ArrayProgram = {

    use Int32Utils;
    use Float64Utils;

    use Padding[Element => Int32];

}