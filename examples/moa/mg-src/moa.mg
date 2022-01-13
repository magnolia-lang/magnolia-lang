package moa imports Util;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

concept Array = {

    use Int;

    // the array type
    type A;

    // the element type
    type E;

    // the index type
    type I;

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

    // function cat(a: A, b: A): A ;
    use Monoid[M => A, op => cat, id => empty];

    function psi(i: I, a: A): A guard partialIndex(i,a);
    function psi(i: I, a: A): E guard totalIndex(i,a);
    function psi(i: I, s: Shape): Int;  // accessing elements of the shape

    function take(i: I, a: A): A guard validIndex(i,a);
    function drop(i: I, a: A): A guard validIndex(i,a);


    // transformations
    function reverse(a: A): A;

    function rotate(ax: Int, a: A): A guard ax < total(shape(a));

    function transpose(a: A): A;

    // ONF
    function reshape(s: Shape, a: A): A guard total(s) == total(shape(a));
    //function gamma

    // arithmetic operations on arrays
    use BMap[A => A, E => E, bop => _+_, bopmap => _+_];
    use BMap[A => A, E => E, bop => _*_, bopmap => _*_];
}



// put in array concept instead? more to avoid (more) messy code for now
concept Padding = {

    use Array;

    // is this neccessary?
    type Shape_ann;

    function shape_ann(a: A): Shape_ann;

     // from padding paper, what is s_j, the j'th element of shape(a)?
    function lift(j: I, d: Int, q: Int, a: A): A;
    // TODO maybe put as guard instead?
    axiom liftAxiom(j: I, d: Int, q: Int, a: A) {
        assert d*q == psi(j, shape(a));
    }

    function padr(ax: Int, a: A, k: I): A;
    function padl(ax: Int, a: A, k: I): A;


    function liftp(): A;
    /* prelift_i(d,A) = B

    */
    function prelift(): A;
    /*

    function prelift()

    function dpadr()
    function dpadl()
*/

}

concept MoA = {

    use Padding;

    use Array[A => LinearArray, I => LinearIndex];
    use Array[A => MultiArray, I => MultiIndex];

    function iota(i: Int): LinearArray;

    function ravel(a: MultiArray): LinearArray;
}



