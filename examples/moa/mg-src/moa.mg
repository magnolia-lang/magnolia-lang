package moa imports Util;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

signature Array = {

    // the array type
    type A;

    // the element type
    type E;

    // the index type
    type I;

    type Int;
    type Shape;

    // core unary functions
    function dim(a: A): Int;
    function shape(a: A): Shape;
    function total(s: Shape): Int;

    // need some form of predicate to define valid bounds
    predicate validIndex(i: I, a: A);

    // i.e. total(i) < total(shape(a))
    predicate partialIndex(i: I, a: A);
    // i.e. total(i) == total(shape(a))
    predicate totalIndex(i: I, a: A);

    // core binary operations
    function cat(a: A, b: A): A guard shape(a) == shape(b);

    function psi(i: I, a: A): A guard partialIndex(i,a);
    function psi(i: I, a: A): E guard totalIndex(i,a);

    function take(i: I, a: A): A guard validIndex(i,a);
    function drop(i: I, a: A): A guard validIndex(i,a);

    // onf level?
    function reshape(s: Shape, a: A): A guard total(s) == total(shape(a));
    //function gamma

}

// put in array concept instead? more to avoid (more) messy code for now
signature Padding = {

    use Array;

    function shape_ann(a: A): Shape;


    /*
    function padr()
    function padl()
    function prelift()
    function liftp()
    function dpadr()
    function dpadl()
*/



}

signature MoA = {

    use Array[A => LinearArray, I => LinearIndex];
    use Array[A => MultiArray, I => MultiIndex];

    function iota(i: Int): LinearArray;

    function ravel(a: MultiArray): LinearArray;
}



