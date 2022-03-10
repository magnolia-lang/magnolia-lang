package examples.moa.mg-src.externals.array-externals;

implementation ExtOps = external C++ base.array {

    require type Element;

    type PaddedArray;
    type Array;
    type Index;
    type IndexContainer;
    type Shape;
    type Int;
    type Float;


    // unpadded array getters, setters
    function get(a: Array, ix: Int): Array;
    function get(a: Array, ix: Index): Array;
    function get(a: PaddedArray, ix: Index): Array;

    procedure set(upd a: Array, obs ix: Int, obs e: Element);
    procedure set(upd a: Array, obs ix: Index, obs e: Element);
    procedure set(upd a: Array, obs ix: Index, obs e: Array);
    procedure set(upd a: PaddedArray, obs ix: Index, obs e: Element);

    function get_shape_elem(a: Array, i: Int): Int;
    function drop_shape_elem(a: Array, i: Int): Shape;
    function get_index_elem(ix: Index, i: Int): Int;
    function drop_index_elem(ix: Index, i: Int): Index;
    function get_index_ixc(ixc: IndexContainer, ix: Int): Index;

    // unpadded moa operations that are convenient to have tied to backend
    function dim(a: Array): Int;
    function total(a: Array): Int;
    function total(s: Shape): Int;
    function shape(a: Array): Shape;
    function shape(a: PaddedArray): Shape;
    function size(s: Shape): Int;
    function size(ixc: IndexContainer): Int;

    // padded array getters
    //function padded_get(a: PaddedArray, ix: Index): PaddedArray;
    //procedure padded_set(upd a: PaddedArray, obs ix: Index, obs e: Element);
    function padded_get_shape_elem(a: PaddedArray, i: Int): Int;
    function padded_drop_shape_elem(a: PaddedArray, i: Int): Shape;


    // padded moa operations
    function padded_dim(a: PaddedArray): Int;
    function padded_total(a: PaddedArray): Int;
    function padded_shape(a: PaddedArray): Shape;

    // creation
    function create_array(sh: Shape): Array;
    function create_padded_array(unpadded_shape: Shape, padded_shape: Shape,
                                padded_array: Array): PaddedArray;
    function create_shape1(a: Int): Shape;
    function create_shape2(a: Int, b: Int): Shape;
    function create_shape3(a: Int, b: Int, c: Int): Shape;
    function create_shape4(a: Int, b: Int, c: Int, d: Int): Shape;


    function create_partial_indices(a: Array, i: Int): IndexContainer;
    function create_total_indices(a: Array): IndexContainer;
    function create_total_indices(a: PaddedArray): IndexContainer;
    function create_index1(a: Int): Index;
    function create_index2(a: Int, b: Int): Index;
    function create_index3(a: Int, b: Int, c: Int): Index;
    function create_index4(a: Int, b: Int, c: Int, d: Int): Index;


    // util
    function unwrap_scalar(a: Array): Element;
    function int_elem(a: Int): Element;
    function elem_int(a: Element): Int;
    function float_elem(f: Float): Element;
    function elem_float(e: Element): Float;
    function cat_shape(a: Shape, b: Shape): Shape;
    function cat_index(i: Index, j: Index): Index;
    function reverse_index(ix: Index): Index;
    function reverse_shape(s: Shape): Shape;
    function padded_to_unpadded(a: PaddedArray): Array;

    // IO
    procedure print_array(obs a: Array);
    procedure print_parray(obs a: PaddedArray);
    procedure print_index(obs i: Index);
    procedure print_index_container(obs i: IndexContainer);
    procedure print_shape(obs sh: Shape);
    procedure print_element(obs e: Element);
    procedure print_int(obs u: Int);
    procedure print_float(obs f: Float);

    // testing
    function test_vector2(): Array;
    function test_vector3(): Array;
    function test_vector5(): Array;

    function test_array3_2_2(): Array;
    function test_array3_2_2F(): Array;
    function test_array3_3(): Array;
    function test_array2_2_3_3(): Array;
    function test_index(): Index;

    function fengshui_array(): Array;

    // int/float operations

    function zero(): Int;
    function one(): Int;
    function binary_add(a: Int, b: Int): Int;
    function binary_sub(a: Int, b: Int): Int;
    function mul(a: Int, b: Int): Int;
    function div(a: Int, b: Int): Int;
    function unary_sub(a: Int): Int;

    function abs(a: Int): Int;

    predicate le(a: Int, b: Int);
    predicate lt(a: Int, b: Int);


    function zeroF(): Float;
    function oneF(): Float;
    function binary_add(a: Float, b: Float): Float;
    function binary_sub(a: Float, b: Float): Float;
    function mul(a: Float, b: Float): Float;
    function div(a: Float, b: Float): Float;
    function unary_sub(a: Float): Float;

    function abs(a: Float): Float;

    predicate le(a: Float, b: Float);
    predicate lt(a: Float, b: Float);
}