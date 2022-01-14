package examples.moa.testing.mg-src.Sum;

implementation ImplSum = external C++ base.sum_ops {
    require type IntLike;

    function sum(a: IntLike, b: IntLike): IntLike;

    type Result;
    function toString(a: IntLike): Result;
}


implementation BasicTypes = external C++ base.basic_types {
    type Int32;
}

implementation MySumImpl = {
    use ImplSum;

    function doSum(a: IntLike, b: IntLike): Result = toString(sum(a,b));
}

program SumProgram = {
    use BasicTypes;
    use MySumImpl[IntLike => Int32];
}
