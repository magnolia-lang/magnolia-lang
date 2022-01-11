package moa;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/


// TODO remove semigroup and monoid?, starting fresh
concept Semigroup = {
    type S;
    function op(a: S, b: S): S;

    axiom associativeAxiom(a: S, b: S, c: S) {
        assert op(op(a, b), c) == op(a, op(b, c));
    }
}

concept Monoid = {
    use Semigroup[S => M];
    function id(): M;

    axiom idAxiom(a: M) {
        assert op(id(), a) == a;
        assert op(a, id()) == a;
    }
}
// ##############################################

signature Shape = {

    type Shape;
    type Int;

}

signature Array = {

    use Shape;

    // the array type
    type A;

    // the element type
    type E;

    // the index type
    type I;

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
    function psi(i: I, a: A): A guard partialIndex(i,a);
    function psi(i: I, a: A): E guard totalIndex(i,a);

    function take(i: I, a: A): A guard validIndex(i,a);
    function drop(i: I, a: A): A guard validIndex(i,a);

    function reshape(s: Shape, a: A): A guard total(s) == total(shape(a));

}

signature MoA = {

    use Array[A => Index];
    use Array[A => LinearArray, I => LinearIndex];
    use Array[A => MultiArray, I => MultiIndex];

    function iota(i: Int): Index;

    function reshape(s: Shape, a: MultiArray): MultiArray guard total(s) == total(shape(a));
    //function ravel(a: MA):
}
/*
signature Padding = {

    use CoreOperations;

    function shape_ann(a: MA): Shape;

    function padr()
    function padl()
    function prelift()
    function liftp()
    function dpadr()
    function dpadl()

}
*/

