package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.fdm,
            examples.moa.mg-src.moa-core,
            examples.moa.mg-src.externals.number-types-externals;




program Int64Arrays = {

   use Int64Utils;

   use Transformations[Element => Int64,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_  => unary_sub,
              _<_ => lt,
              _<=_=> le];

}

program Float64Arrays = {

   use Float64Utils;

   use Transformations[Element => Float64,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_ => unary_sub,
              _<_ => lt,
              _<=_=> le];
}

program BurgerProgram = {

   use Int64Utils;
   use Transformations[Element => Int64,
              _+_ => binary_add,
              _-_ => binary_sub,
              _*_ => mul,
              _/_ => div,
              -_  => unary_sub,
              _<_ => lt,
              _<=_=> le];

}