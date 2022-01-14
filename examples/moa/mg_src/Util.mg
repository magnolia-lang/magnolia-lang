/*
* Utilities? Not core operations, but might be useful
*
*
*/
package Util;

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

concept Int = {

    // gives us zero and addition
    use Monoid [M => Int,
                id => zero,
                op => _+_,
                associativeAxiom => addAssociativeAxiom,
                idAxiom => addIdentiyAxiom];

    // gives us one and multiplication
    use Monoid [M => Int,
                id => one,
                op => _*_,
                associativeAxiom => multAssociativeAxiom,
                idAxiom => multIdentityAxiom];

    predicate _<_(a: Int, b: Int);
    predicate _<=_(a: Int, b: Int);

    function _-_(a: Int, b: Int): Int guard b <= a;
    function min(a: Int, b: Int): Int;

    axiom minAxiom(a: Int, b: Int) {
        assert (a <= b) => min(a,b) == a;
        assert (b < a) => min(a,b) == b;
    }

    // l < u && l <= i < u, sometimes written i in [l,u)
    predicate inRange( i: Int, l: Int, u: Int);

    // l <= u && l <= i <= u, sometimes written i in [l,u]
    predicate inRangeInclusive(i: Int, l: Int, u: Int);

    axiom inRangeAxiom(i: Int, lowerInclusive: Int, upper: Int)
    guard lowerInclusive < upper {
        assert inRange(i, lowerInclusive, upper) <=>
        (lowerInclusive <= i) && (i < upper);
    }

    axiom inRangeInclusiveAxiom
    (i: Int, lowerInclusive: Int, upperInclusive: Int)
    guard lowerInclusive <= upperInclusive {
        assert inRangeInclusive(i, lowerInclusive, upperInclusive) <=>
        (lowerInclusive <= i) && (i <= upperInclusive);
    }
}

/*
need to be able to map functions over array elements
*/
concept BMap = {
    type A;
    type E;

    function bop(e1: E, e2: E): E;
    function bopmap(e: E, a: A): A;
}

/*
useful predicates and helper functions
*/

concept UtilFunctions = {

    type A;
    type E;
    type I;
    type Shape;

    // need some form of predicate to define valid bounds
    predicate validIndex(i: I, a: A);
    // i.e. total(i) < total(shape(a))
    predicate partialIndex(i: I, a: A);
    // i.e. total(i) == total(shape(a))
    predicate totalIndex(i: I, a: A);

    // conversions between differernt "equivalent" array types
    function shapeToArray(s: Shape): A;
}
