package examples.moa.mg-src.fdm.mg
    imports examples.moa.mg-src.moa-cpp;

concept MappedOps = {

    require type Element;


    function _+_(a: Element, b: Element): Element;
    function _-_(a: Element, b: Element): Element;
    function _*_(a: Element, b: Element): Element;
    function -_(a: Element): Element;


    function _+_(a: Array, b: Array): Array;
    function _-_(a: Array, b: Array): Array;
    function _*_(a: Array, b: Array): Array;
    function -_(a: Array): Array;


    axiom binaryMap(a: Array, b: Array, ix: Index) {
        assert get(a+b, ix) ==
        get(a, ix) + get(b, ix);

        assert get(a*b, ix) ==
        get(a, ix) * get(b, ix);

        assert get(a-b, ix) ==
        get(a, ix) - get(b, ix);
    }

    axiom unaryMap(a: Array, ix: Index) {
        assert get(-a, ix) == - get(a, ix);
    }

implementation Burger = {

    use MappedOps;
    use ArrayProgram;

}

}



