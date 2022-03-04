package examples.moa.mg-src.fdm
    imports examples.moa.mg-src.moa-core-impl,
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

implementation Partial = {

    use Transformations;


    //TODO: change
    function deltax(): Float = oneF() + oneF();


    function two(): Int = one(): Int + one(): Int;
    function twoF(): Float = oneF() + oneF();

    function partial1(a: Array, dir: Int): Array = {
        value float_elem((twoF() * deltax())) / (rotate(one(): Int, dir, a) - rotate(-one(): Int, dir, a));
    }

    function partial2(a: Array, dir: Int): Array = {
        value float_elem((twoF() * deltax())) / (rotate(one(): Int, dir, a) + rotate(-one(): Int, dir, a));
    }
}

implementation Burger = {

    use Transformations;

    function two(): Int = one(): Int + one(): Int;
    function twoF(): Float = oneF() + oneF();

    // from fengshui
    procedure snippet(upd u: Array,
                      obs v: Array,
                      obs u0: Array,
                      obs u1: Array,
                      obs u2: Array,
                      obs c0: Float,
                      obs c1: Float,
                      obs c2: Float,
                      obs c3: Float,
                      obs c4: Float) {


        var shift_v = rotate(-one(): Int, zero(): Int, v);
        var d1a = float_elem(-c0) * shift_v;
        var d2a = float_elem(c1) * shift_v - float_elem(c2) * u0;
        shift_v = rotate(one(): Int, zero(): Int, v);
        d1a = d1a + float_elem(c0) * shift_v;
        d2a = d2a + float_elem(c1) * shift_v;


        shift_v = rotate(-one(): Int, one(): Int, v);
        var d1b = float_elem(-c0) * shift_v;
        var d2b = float_elem(c1) * shift_v - float_elem(c2) * u0;
        shift_v = rotate(one(): Int, one(): Int, v);
        d1b = d1b + float_elem(c0) * shift_v;
        d2b = d2b + float_elem(c1) * shift_v;

        shift_v = rotate(-one(): Int, two(): Int, v);
        var d1c = float_elem(-c0) * shift_v;
        var d2c = float_elem(c1) * shift_v - float_elem(c2) * u0;
        shift_v = rotate(one(): Int, two(): Int, v);
        d1c = d1c + float_elem(c0) * shift_v;
        d2c = d2c + float_elem(c1) * shift_v;

        d1a = u0 * d1a + u1 * d1b + u2 *d1c;
        d2a = d2a + d2b + d2c;
        u = u + float_elem(c4) * (float_elem(c3) * d2a - d1a);
    }

    procedure bstep(upd u0: Array, upd u1: Array, upd u2: Array,
                   obs nu: Float, obs dx: Float, obs dt: Float) {

        var c0 = (oneF() / twoF()) / dx;

        call print_float(c0);
        var c1 = oneF()/dx/dx;
        call print_float(c1);
        var c2 = twoF()/dx/dx;
        call print_float(c2);
        var c3 = nu;
        call print_float(c3);
        var c4 = dt/twoF();
        call print_float(c4);

        var v0 = u0;
        var v1 = u1;
        var v2 = u2;

        call snippet(v0,u0,u0,u1,u2,c0,c1,c2,c3,c4);
        call snippet(v1,u1,u0,u1,u2,c0,c1,c2,c3,c4);
        call snippet(v2,u2,u0,u1,u2,c0,c1,c2,c3,c4);
        call snippet(u0,v0,v0,v1,v2,c0,c1,c2,c3,c4);
        call snippet(u1,v1,v0,v1,v2,c0,c1,c2,c3,c4);
        call snippet(u2,v2,v0,v1,v2,c0,c1,c2,c3,c4);

    }

}






