package examples.moa.mg-src.fdm
    imports examples.moa.mg-src.moa-core,
            examples.moa.mg-src.externals.array-externals,
            examples.moa.mg-src.externals.while-loops;


/*

MoA operations needed for FDM: (source: fengshui)
-shape DONE
-psi   DONE
-rotate DONE
-cat DONE
*/

/*
need:

laplacian?
stencils?


*/


implementation Burger = {

    use Transformations;

    procedure snippet(upd u: Array,
                      obs v: Array,
                      obs u0: Array,
                      obs u1: Array,
                      obs u2: Array,
                      obs c0: Element,
                      obs c1: Element,
                      obs c2: Element,
                      obs c3: Element,
                      obs c4: Element) {

        var shift_v = rotate(elem_int(zero()), elem_int(-one()), v);
        var d1a = -c0 * shift_v;
        var d2a = c1 * shift_v - c2 * u0;

        shift_v = rotate(v, elem_int(zero()), elem_int(one()));
    }

    //use Padding;

}



