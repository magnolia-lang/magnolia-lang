package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.fdm,
            examples.moa.mg-src.moa-core,
            examples.moa.mg-src.num-ops,
            examples.moa.mg-src.externals.number-types-externals;


implementation Int32Utils = external C++ base.int32_utils
    NumOps[NumberType => Int32];


implementation Float64Utils = external C++ base.float64_utils
    NumOps[NumberType => Float64];

program Int32Arrays = {

   use Int32Utils;

   use Padding[Element => Int32,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_  => unary_sub,
              _<_ => lt];

}

program Float64Arrays = {

   use Float64Utils;

   use Padding[Element => Float64,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_ => unary_sub,
              _<_ => lt];
}

program BurgerProgram = {

   use Int32Utils;
   use MoaOps[Element => Int32,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_  => unary_sub,
              _<_ => lt];

}