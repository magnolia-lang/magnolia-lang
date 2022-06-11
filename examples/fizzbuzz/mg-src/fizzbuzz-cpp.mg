package examples.fizzbuzz.mg-src.fizzbuzz-cpp;

// The longest fizzbuzz in the world!
implementation ImplFizzBuzz = external C++ base.fizzbuzz_ops {
    require type IntLike;
    function modulo(a: IntLike, modulus: IntLike): IntLike;
    function five(): IntLike;
    function three(): IntLike;
    function zero(): IntLike;

    type FizzBuzz;
    function fizz(): FizzBuzz;
    function buzz(): FizzBuzz;
    function fizzbuzz(): FizzBuzz;
    function nope(): FizzBuzz;
}

implementation ImplTypes = external C++ base.basic_types {
    type Int32;
}

implementation MyFizzBuzzImplementation = {
    use ImplFizzBuzz;
    function doFizzBuzz(i: IntLike): FizzBuzz =
        if modulo(i, three()) == zero() && modulo(i, five()) == zero()
        then fizzbuzz()
        else if modulo(i, three()) == zero() then fizz()
        else if modulo(i, five()) == zero() then buzz()
        else nope();
}

program MyFizzBuzzProgram = {
    use ImplTypes;
    use MyFizzBuzzImplementation[IntLike => Int32];
}
