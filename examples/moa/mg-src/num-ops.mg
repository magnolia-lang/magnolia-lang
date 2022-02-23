package examples.moa.mg-src.num-ops;

signature NumOps = {

    type NumberType;

    function zero(): NumberType;
    function one(): NumberType;

    function binary_add(a: NumberType, b: NumberType): NumberType;
    function binary_sub(a: NumberType, b: NumberType): NumberType;
    function mul(a: NumberType, b: NumberType): NumberType;
    function div(a: NumberType, b: NumberType): NumberType;
    function unary_sub(a: NumberType): NumberType;

    predicate lt(a: NumberType, b: NumberType);
   // predicate eq(a: NumberType, b: NumberType);
}