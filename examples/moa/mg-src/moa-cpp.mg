package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.fdm,
            examples.moa.mg-src.moa-core,
            examples.moa.mg-src.num-ops,
            examples.moa.mg-src.externals.number-types-externals;


implementation Int32Utils = external C++ base.int32_utils
    NumOps[NumberType => Int32];

program ArrayProgram = {

   use Int32Utils;
   use Padding[Element => Int32,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_  => unary_sub,
              _<_ => lt];

}

program TestingSuite = {

   use Int32Utils;
   use Ravel[Element => Int32,
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