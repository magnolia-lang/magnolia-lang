package examples.moa.mg-src.externals.number-types-externals;

signature NumberOps = {
    type NumberType;

    function zero(): NumberType;
    function one(): NumberType;
    function add(a: NumberType, b: NumberType): NumberType;
    function sub(a: NumberType, b: NumberType): NumberType;

    function mult(a: NumberType, b: NumberType): NumberType;

    predicate equals(a: NumberType, b: NumberType);
    predicate isLowerThan(a: NumberType, b: NumberType);
}


implementation Int32Utils = external C++ base.int32_utils
    NumberOps[NumberType => Int32];

implementation Float64Utils = external C++ base.float64_utils
    NumberOps[NumberType => Float64];
