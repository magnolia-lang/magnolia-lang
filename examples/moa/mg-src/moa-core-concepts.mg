package examples.moa.mg-src.moa-core-concepts;

concept MappedOps = {

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

    function _+_(a: Array, b: Array): Array;
    function _-_(a: Array, b: Array): Array;
    function _*_(a: Array, b: Array): Array;
    function _/_(a: Array, b: Array): Array;
    function -_(a: Array): Array;

    predicate _==_(a: Array, b: Array);

    // scalar left arguments
    function _+_(a: Element, b: Array): Array;
    function _-_(a: Element, b: Array): Array;
    function _*_(a: Element, b: Array): Array;
    function _/_(a: Element, b: Array): Array;

    function get(a: Array, ix: Index): Array;
    function unwrap_scalar(a: Array): Element;


    axiom binaryMap(a: Array, b: Array, ix: Index) {
        assert unwrap_scalar(get(a+b, ix)) ==
        unwrap_scalar(get(a, ix)) + unwrap_scalar(get(b, ix));

        assert unwrap_scalar(get(a-b, ix)) ==
        unwrap_scalar(get(a, ix)) - unwrap_scalar(get(b, ix));

        assert unwrap_scalar(get(a*b, ix)) ==
        unwrap_scalar(get(a, ix)) * unwrap_scalar(get(b, ix));

        assert unwrap_scalar(get(a/b, ix)) ==
        unwrap_scalar(get(a, ix)) / unwrap_scalar(get(b, ix));
    }

    axiom scalarLeftMap(e: Element, a: Array, ix: Index) {
        assert unwrap_scalar(get(e+a, ix)) ==
            e + unwrap_scalar(get(a, ix));
        assert unwrap_scalar(get(e-a, ix)) ==
            e - unwrap_scalar(get(a, ix));
        assert unwrap_scalar(get(e*a, ix)) ==
            e * unwrap_scalar(get(a, ix));
        assert unwrap_scalar(get(e/a, ix)) ==
            e / unwrap_scalar(get(a, ix));
    }

    axiom unaryMap(a: Array, ix: Index) {
        assert unwrap_scalar(get(-a, ix)) == - unwrap_scalar(get(a, ix));
    }

}

concept MoaOps = {

    use MappedOps;

    type Shape;
    type PaddedArray;

    type IndexContainer;

    //internal number types
    type Int;
    type Float;

    // psi
    function get(a: Array, ix: Index): Array;
    function get(a: PaddedArray, ix: Index): Array;

    procedure set(upd a: Array, obs ix: Index, obs e: Element);
    procedure set(upd a: PaddedArray, obs ix: Index, obs e: Element);

    // delta
    function dim(a: Array): Int;
    // rho
    function shape(a: Array): Shape;
    function shape(a: PaddedArray): Shape;

    // tau
    function total(a: Array): Int;
    function total(s: Shape): Int;

    function size(s: Shape): Int;
    function size(ixc: IndexContainer): Int;

    // theta
    function rotate(sigma: Int, axis: Int, a: Array): Array;

    axiom rotateShapeAxiom(sigma: Int, axis: Int, a: Array) {
        var rotated = rotate(sigma, axis, a);
        assert shape(rotated) == shape(a);
    }

    function reshape(a: Array, s: Shape): Array;

    function cat(a1: Array, a2: Array): Array;

    function take(i: Int, a: Array): Array;
    function drop(i: Int, a: Array): Array;

    function circular_padl(a: Array, i: Int): PaddedArray;
    function circular_padl(a: PaddedArray, i: Int): PaddedArray;

    function circular_padr(a: Array, i: Int): PaddedArray;
    function circular_padr(a: PaddedArray, i: Int): PaddedArray;

    /*
    axiom padlPadrCommuteAxiom(a: Array, i: Int) = {

        var padlpadr = circular_padl(circular_padr(a,i),i);
        var padrpadl = circular_padr(circular_padl(a,i),i);
        assert padlpadr == padrpadl;
    }*/
}