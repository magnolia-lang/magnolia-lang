package MOA imports Util;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

concept Array = {

    use Int;

    type Array;
    type E;
    type IndexSpace;
    type Shape;

    function getShapeElem(i: Int, s: Shape): Int;


    /*
    gives us:

        function shapeToArray(s:Shape):A;

        predicates for checking index bounds and lengths
    */
    use UtilFunctions[A=>Array,
                      E=>E,
                      I=>IndexSpace,
                      Shape=>Shape];

    // core unary functions
    function dim(a: Array): Int;
    function shape(a: Array): Shape;
    function total(a: Array): Int;
    function total(s: Shape): Int;

    // core binary operations
    function cat(a1: Array, a2: Array): Array;


    function psi(i: IndexSpace, a: Array): E;

    function take(i: IndexSpace, a: Array): Array;
    function drop(i: IndexSpace, a: Array): Array;


    // transformations
    function reverse(a: Array): Array;
    function rotate(ax: Int, a: Array): Array guard ax < total(shape(a));
    function transpose(a: Array): Array;

    // arithmetic operations on arrays
    use BMap[A => A, E => E, bop => _+_, bopmap => _+_];
    use BMap[A => A, E => E, bop => _*_, bopmap => _*_];

    // ONF
    function reshape(s: Shape, a: Array): Array;
    function gamma(i: IndexSpace, s: Shape): IndexSpace;

}



// put in array concept instead? more to avoid (more) messy code for now
concept Padding = {

    use Array;

    type PaddedArray;

    type Shape_ann;

    function shape_ann(a: A): Shape_ann;

    /*
    functions to extract beginning and end of slices?
    function getSliceStart(s:Shape_ann):Int;
    function getSliceEnd(s: Shape_ann):Int;
    */

     // from padding paper, what is s_j, the j'th element of shape(a)?

    function cpadr(ax: Int, a: Array): PaddedArray;
    function cpadl(ax: Int, a: Array): PaddedArray;


    function liftp(): Array;
    /* prelift_i(d,A) = B

    */
    function prelift(): Array;
    /*
    function lift()
    function prelift()

    function dpadr()
    function dpadl()
*/

}
/*
concept MOA = {

    use Padding[A => LinearArray, I => LinearIndex];
    use Padding[A => MultiArray, I => MultiIndex];

    function iota(i: Int): LinearArray;

    function ravel(a: MultiArray): LinearArray;

}
*/

implementation BasicTypes = external C++ base.basic_types {
    type Int32;
}

implementation ImplSum = external C++ base.array_ops {
    require type Int;
    type Result;
    function foo(): Result;
}

implementation MySumImpl = {
    use ImplSum;

    function doFoo(): Result = foo();
}

program SumProgram = {
    use BasicTypes;
    use MySumImpl[Int => Int32];
}

