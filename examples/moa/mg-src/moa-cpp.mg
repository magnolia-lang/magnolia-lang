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
    type Shape;
    type UInt32;

    function create_array(sh: Shape): Array;
    function create_shape_3(a: UInt32, b: UInt32, c: UInt32): Shape;
    function dim(a: Array): UInt32;
    function total(a: Array): UInt32;

    function get_shape_elem(a: Array, i: UInt32): UInt32;

    procedure print_array(obs a: Array);
    procedure print_shape(obs sh: Shape);
}


implementation Int32Utils = external C++ base.int32_utils
    NumberOps[NumberType => Int32];

implementation Float64Utils = external C++ base.float64_utils
    NumberOps[NumberType => Float64];

program ArrayProgram = {

    use Int32Utils;
    use Float64Utils;

   // use ExtOps[Element => Int32];
    use ExtOps[Element => Float64];

}