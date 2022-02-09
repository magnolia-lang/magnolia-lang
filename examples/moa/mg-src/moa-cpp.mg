package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.while-loops;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

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

    type Array;
    type Index;
    type Shape;
    type UInt32;

    // getters, setters
    function get(a: Array, ix: Index): Array;
    procedure set(upd a: Array, obs ix: Index, obs e: Element);
    function get_shape_elem(a: Array, i: UInt32): UInt32;
    function drop_shape_elem(a: Array, i: UInt32): Shape;

    // moa operations that are convenient to have tied to backend
    function dim(a: Array): UInt32;
    function total(a: Array): UInt32;
    function shape(a: Array): Shape;

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

    /*
        Vector catenation, i.e. cat(v1, v2)
    */
    procedure cat_vec_body(obs v1: Array,
                           obs v2: Array,
                           upd res: Array,
                           upd counter: UInt32) {

        var v1_bound = uint_elem(total(v1));
        var ix: Index;

        if isLowerThan(uint_elem(counter), uint_elem(total(v1))) then {
            ix = create_index1(counter);
            call set(res, ix, unwrap_scalar(get(v1, ix)));
        } else {
            ix = create_index1(elem_uint(sub(uint_elem(counter), v1_bound)));
            call set(res, ix, unwrap_scalar(get(v2, ix)));
        };

        counter = elem_uint(add(uint_elem(counter), one()));
    }

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


    /*
        Array catenation, i.e. cat(a1, a2)
    */
    function cat(array1: Array, array2: Array): Array
        guard drop_shape_elem(array1, elem_uint(zero())) ==
              drop_shape_elem(array2, elem_uint(zero())) {

        // calculating result shape
        var sh1 = get_shape_elem(array1, elem_uint(zero()));
        var sh2 = get_shape_elem(array2, elem_uint(zero()));
        var sh3 = drop_shape_elem(array1, elem_uint(zero()));

        // TODO concat with sh3
        var result_shape = (add(uint_elem(sh1), uint_elem(sh2)));
        call print_element(result_shape);

        value array1;
    }
}


implementation Int32Utils = external C++ base.int32_utils
    NumberOps[NumberType => Int32];

implementation Float64Utils = external C++ base.float64_utils
    NumberOps[NumberType => Float64];


program ArrayProgram = {

    use Int32Utils;
    use Float64Utils;

    use Catenation[Element => Int32];
    //use ExtOps[Element => Float64];

}