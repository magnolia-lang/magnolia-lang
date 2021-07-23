package examples.fizzbuzz.mg-src.fizzbuzz;

// The longest fizzbuzz in the world!
implementation ImplFizzBuzz = external C++ base.fizzbuzz_ops {
    type IntLike; // TODO: make a required type when the feature is supported.
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

program MyFizzBuzzProgram = {
    use ImplFizzBuzz;
    // We don't have loops, so for the moment, we don't bother.
    function doFizzBuzz(i: IntLike): FizzBuzz =
        if modulo(i, three()) == zero() && modulo(i, five()) == zero()
        then fizzbuzz()
        else if modulo(i, three()) == zero() then fizz()
        else if modulo(i, five()) == zero() then buzz()
        else nope();
}
