package examples.moa.mg-src.moa-core-concepts;

concept Array = {

    require type Element;

    require function zero(): Element;
    require function one(): Element;

    require function _+_(a: Element, b: Element): Element;
    require function _-_(a: Element, b: Element): Element;
    require function _*_(a: Element, b: Element): Element;
    require function _/_(a: Element, b: Element): Element;
    require function -_(a: Element): Element;

    require predicate _<_(a: Element, b: Element);
    require predicate _==_(a: Element, b: Element);

    type Array;
    type Index;
    type Shape;

    function get(a: Array, ix: Int): Array;
    function get(a: Array, ix: Index): Array;
    function get(a: PaddedArray, ix: Index): Array;

    procedure set(upd a: Array, obs ix: Int, obs e: Element);
    procedure set(upd a: Array, obs ix: Index, obs e: Element);
    procedure set(upd a: PaddedArray, obs ix: Index, obs e: Element);

    function dim(a: Array): Int;
    function total(a: Array): Int;
    function total(s: Shape) : Int;
    function total(ixc: IndexContainer): Int;


    function reshape(a: Array, s: Shape): Array;

    axiom reshapeShapeAxiom(a: Array, s: Shape) {
        assert total(shape(a)) == total(s);
    }

    function cat(a1: Array, a2: Array): Array;

    axiom catShapeAxiom(a1: Array, a2: Array) {
        assert shape(cat(a1,a2)) == shape(a1) + shape(a2);
    }

    function rotate(sigma: Int, axis: Int, a: Array): Array;

    axiom rotateShapeAxiom(sigma: Int, axis: Int: a: Array) {
        var rotated = rotate(sigma, axis, a);
        assert shape(rotated) == shape(a);
    }

    function take(i: Int, a: Array): Array;
    function drop(i: Int, a: Array): Array;
}