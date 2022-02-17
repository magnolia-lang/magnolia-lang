package examples.moa.mg-src.externals.array-externals;

implementation ExtOps = external C++ base.array {

    require type Element;

    type PaddedArray;
    type Array;
    type Index;
    type IndexContainer;
    type Shape;
    type UInt32;

    // unpadded array getters, setters
    function get(a: Array, ix: UInt32): Array;
    function get(a: Array, ix: Index): Array;
    function get(a: PaddedArray, ix: Index): Array;

    procedure set(upd a: Array, obs ix: UInt32, obs e: Element);
    procedure set(upd a: Array, obs ix: Index, obs e: Element);
    procedure set(upd a: PaddedArray, obs ix: Index, obs e: Element);

    function get_shape_elem(a: Array, i: UInt32): UInt32;
    function drop_shape_elem(a: Array, i: UInt32): Shape;
    function get_index_elem(ix: Index, i: UInt32): UInt32;
    function drop_index_elem(ix: Index, i: UInt32): Index;
    function get_index_ixc(ixc: IndexContainer, ix: UInt32): Index;

    // unpadded moa operations that are convenient to have tied to backend
    function dim(a: Array): UInt32;
    function total(a: Array): UInt32;
    function total(s: Shape) : UInt32;
    function total(ixc: IndexContainer): UInt32;
    function shape(a: Array): Shape;
    function shape(a: PaddedArray): Shape;


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
    function create_padded_array(unpadded_shape: Shape, padded_shape: Shape,
                                padded_array: Array): PaddedArray;
    function create_shape1(a: UInt32): Shape;
    function create_shape2(a: UInt32, b: UInt32): Shape;
    function create_shape3(a: UInt32, b: UInt32, c: UInt32): Shape;

    function create_valid_indices(a: Array): IndexContainer;
    function create_valid_indices(a: PaddedArray): IndexContainer;
    function create_index1(a: UInt32): Index;
    function create_index2(a: UInt32, b: UInt32): Index;
    function create_index3(a: UInt32, b: UInt32, c: UInt32): Index;

    // util
    function unwrap_scalar(a: Array): Element;
    function uint_elem(a: UInt32): Element;
    function elem_uint(a: Element): UInt32;
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
    procedure print_uint(obs u: UInt32);

    // testing
    function test_vector2(): Array;
    function test_vector3(): Array;
    function test_vector5(): Array;

    function test_array3_2_2(): Array;
    function test_array3_3(): Array;
    function test_index(): Index;

}