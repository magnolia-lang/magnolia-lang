package examples.moa.mg-src.externals.number-types-externals;

concept NumberOps = {
    type NumberType;

    function zero(): NumberType;
    function one(): NumberType;

    function binary_add(a: NumberType, b: NumberType): NumberType;
    function binary_sub(a: NumberType, b: NumberType): NumberType;
    function mul(a: NumberType, b: NumberType): NumberType;
    function div(a: NumberType, b: NumberType): NumberType;
    function unary_sub(a: NumberType): NumberType;
    function abs(a: NumberType): NumberType;
    predicate eq(a: NumberType, b: NumberType);
    predicate lt(a: NumberType, b: NumberType);
    predicate le(a: NumberType, b: NumberType);

}

implementation Int32Utils = external C++ base.int32_utils
    NumberOps[NumberType => Int32];


implementation Float64Utils = external C++ base.float64_utils
    NumberOps[NumberType => Float64];