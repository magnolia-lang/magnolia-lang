package examples.moa.mg-src.fdm.mg
    imports examples.moa.mg-src.moa-cpp;

concept ElementwiseOps = {

    require type Element;


    function _+_(a: Element, b: Element): Element;
    function _-_(a: Element, b: Element): Element;
    function _*_(a: Element, b: Element): Element;
    function -_(a: Element): Element;

}

concept MappedArrayOps = {

    type Array;

    function _+_(a: Array, b: Array): Array;
    function _-_(a: Array, b: Array): Array;
    function _*_(a: Array, b: Array): Array;
    function -_(a: Array): Array;

}