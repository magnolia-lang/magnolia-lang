package examples.pde.mg-src.pde-cpp;

implementation PDE = {
    type Array;
    type Float;
    type Axis;
    type Offset;

    /* Float ops */
    require function -_(f: Float): Float;
    require function _+_(lhs: Float, rhs: Float): Float;
    require function _-_(lhs: Float, rhs: Float): Float;
    require function _*_(lhs: Float, rhs: Float): Float;
    require function _/_(num: Float, den: Float): Float;
    require function one(): Float;
    require function two(): Float;
    require function three(): Float;

    /* Scalar-Array ops */
    require function _+_(lhs: Float, rhs: Array): Array;
    require function _-_(lhs: Float, rhs: Array): Array;
    require function _*_(lhs: Float, rhs: Array): Array;
    require function _/_(num: Float, den: Array): Array;

    /* Array-Array ops */
    require function _+_(lhs: Array, rhs: Array): Array;
    require function _-_(lhs: Array, rhs: Array): Array;
    require function _*_(lhs: Array, rhs: Array): Array;

    /* Rotate */
    require function rotate(a: Array, axis: Axis, o: Offset): Array;

    /* Axis utils */
    require function zero(): Axis;
    require function one(): Axis;
    require function two(): Axis;

    /* Offset utils */
    require function one(): Offset;
    require function -_(o: Offset): Offset;

    /* Solver */
    procedure step(upd u0: Array, upd u1: Array, upd u2: Array) {
        var v0 = u0;
        var v1 = u1;
        var v2 = u2;

        // TODO: transpose as AoS instead of SoA
        v0 = snippet(v0, u0, u0, u1, u2);
        v1 = snippet(v1, u1, u0, u1, u2);
        v2 = snippet(v2, u2, u0, u1, u2);
        u0 = snippet(u0, v0, u0, u1, u2);
        u1 = snippet(u1, v1, u0, u1, u2);
        u2 = snippet(u2, v2, u0, u1, u2);
    }

    require function nu(): Float;
    require function dt(): Float;
    require function dx(): Float;

    type Index;

    require function psi(ix: Index, array: Array): Float;

    function snippet(u: Array, v: Array, u0: Array,
                     u1: Array, u2: Array) : Array =
        u + dt()/(two(): Float) * (nu() * ((one(): Float)/dx()/dx() * (rotate(v, zero(), -one(): Offset) + rotate(v, zero(), one(): Offset) + rotate(v, one(): Axis, -one(): Offset) + rotate(v, one(): Axis, one(): Offset) + rotate(v, two(): Axis, -one(): Offset) + rotate(v, two(): Axis, one(): Offset)) - three() * (two(): Float)/dx()/dx() * u0) - (one(): Float)/(two(): Float)/dx() * ((rotate(v, zero(), one(): Offset) - rotate(v, zero(), -one(): Offset)) * u0 + (rotate(v, one(): Axis, one(): Offset) - rotate(v, one(): Axis, -one(): Offset)) * u1 + (rotate(v, two(): Axis, one(): Offset) - rotate(v, two(): Axis, -one(): Offset)) * u2));
}


program BasePDEProgram = {
    use PDE[-_ => unary_sub, _-_ => binary_sub];
    use ExtArrayOps[ one_float => one
                   , two_float => two
                   , three_float => three
                   , zero_axis => zero
                   , one_axis => one
                   , two_axis => two
                   , one_offset => one
                   ];
    use ExtConstants;
    //use ExtExtendMissingBypass;
}

program PDEProgram0 = BasePDEProgram;
program PDEProgram1 = {
    use (rewrite
            (rewrite
                (generate ToIxwiseGenerator in BasePDEProgram)
            with DNFRules 20)
        with ToIxwise 1);

    use ExtExtendMissingBypass;
}
// program PDEProgram2 = {
//     use (rewrite PDEProgram1 with OFLiftCores 1);
//     use ExtExtendLiftCores;
// }

program PDEProgram = PDEProgram1;

concept ToIxwiseGenerator = {
    type Array;
    type Float;
    type Index;

    function snippet_ix(u: Array, v: Array, u0: Array,
                        u1: Array, u2: Array, ix: Index): Float;

    function snippet(u: Array, v: Array, u0: Array,
                     u1: Array, u2: Array): Array;

    function psi(ix: Index, array: Array): Float;

    axiom toIxwiseGenerator(u: Array, v: Array, u0: Array,
                            u1: Array, u2: Array, ix: Index) {
        assert snippet_ix(u, v, u0, u1, u2, ix) ==
               psi(ix, snippet(u, v, u0, u1, u2));
    }
}

concept ToIxwise = {
    type Array;

    function snippet(u: Array, v: Array, u0: Array,
                     u1: Array, u2: Array): Array;

    function schedule(u: Array, v: Array, u0: Array,
                               u1: Array, u2: Array): Array;

    axiom toIxwiseRule(u: Array, v: Array, u0: Array,
                       u1: Array, u2: Array) {
        assert snippet(u, v, u0, u1, u2) ==
               schedule(u, v, u0, u1, u2);
    }
}

implementation ExtNeededFns = external C++ base.specialize_psi_ops_2 {
    require type Index;
    require type Offset;
    require type ScalarIndex;
    require type Array;
    require type Float;
    require function snippet_ix_specialized(u: Array, v: Array, u0: Array,
                                    u1: Array, u2: Array, i: ScalarIndex,
                                    j: ScalarIndex, k: ScalarIndex): Float;

    function psi(i: ScalarIndex, j: ScalarIndex, k: ScalarIndex, a: Array)
        : Float;
    function schedule_specialized_psi_padded(u: Array, v: Array,
        u0: Array, u1: Array, u2: Array): Array;

    /* OF Specialize Psi extension */
    //TODO: add ScalarIndex, make_ix
    //type ScalarIndex;

    //function make_ix(ix1: ScalarIndex, ix2: ScalarIndex, ix3: ScalarIndex)
    //    : Index;

    /* OF Reduce MakeIx projections */
    function ix_0(ix: Index): ScalarIndex;
    function ix_1(ix: Index): ScalarIndex;
    function ix_2(ix: Index): ScalarIndex;

    /* OF Reduce MakeIx Rotate extension */
    type AxisLength;

    function _+_(six: ScalarIndex, o: Offset): ScalarIndex;
    function _%_(six: ScalarIndex, sc: AxisLength): ScalarIndex;
    function shape_0(): AxisLength;
    function shape_1(): AxisLength;
    function shape_2(): AxisLength;
}



implementation ExtExtendMissingBypass = external C++ base.forall_ops {
    require type Float;
    require type Array;
    require type Offset;
    require type Axis;
    require type Index;
    require type Nat;

    require function snippet_ix(u: Array, v: Array,
                                u0: Array, u1: Array, u2: Array, ix: Index): Float;

    function schedule(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array): Array;

    type ScalarIndex;
    function make_ix(a: ScalarIndex, b: ScalarIndex, c: ScalarIndex): Index;
}

implementation ExtExtendPadding = external C++ base.forall_ops {
    require type Float;
    require type Array;
    require type Offset;
    require type Axis;
    require type Index;
    require type Nat;

    require function snippet_ix(u: Array, v: Array,
                                u0: Array, u1: Array, u2: Array, ix: Index): Float;

    /* OF Pad extension */
    procedure refill_all_padding(upd a: Array);

    function schedule_padded(u: Array, v: Array,
        u0: Array, u1: Array, u2: Array): Array;

    function rotate_ix_padded(ix: Index, axis: Axis, offset: Offset): Index;
}

implementation ExtExtendLiftCores = external C++ base.forall_ops {
    require type Float;
    require type Array;
    require type Offset;
    require type Axis;
    require type Index;
    require type Nat;

    require function snippet_ix(u: Array, v: Array,
                                u0: Array, u1: Array, u2: Array, ix: Index): Float;
    function schedule_threaded(u: Array, v: Array, u0: Array,
                               u1: Array, u2: Array, nbThreads: Nat): Array;
    function nbCores(): Nat;
}


implementation ExtArrayOps = external C++ base.array_ops {
    require type Float;
    type Array;
    type Offset;
    type Axis;
    type Index;
    type Nat;

    function psi(ix: Index, array: Array): Float;
    //procedure set(obs ix: Index, upd array: Array, obs v: Float);

    /* Float ops */
    function unary_sub(f: Float): Float;
    function _+_(lhs: Float, rhs: Float): Float;
    function binary_sub(lhs: Float, rhs: Float): Float; // TODO: look at bug here
    function _*_(lhs: Float, rhs: Float): Float;
    function _/_(num: Float, den: Float): Float;
    function one_float(): Float;
    function two_float(): Float;
    function three_float(): Float;

    /* Scalar-Array ops */
    function _+_(lhs: Float, rhs: Array): Array;
    function binary_sub(lhs: Float, rhs: Array): Array;
    function _*_(lhs: Float, rhs: Array): Array;
    function _/_(num: Float, den: Array): Array;

    /* Array-Array ops */
    function _+_(lhs: Array, rhs: Array): Array;
    function binary_sub(lhs: Array, rhs: Array): Array;
    function _*_(lhs: Array, rhs: Array): Array;

    /* Rotate */
    function rotate(a: Array, axis: Axis, o: Offset): Array;
    function rotate_ix(ix: Index, axis: Axis, o: Offset): Index;

    /* Axis utils */
    function zero_axis(): Axis;
    function one_axis(): Axis;
    function two_axis(): Axis;

    /* Offset utils */
    function one_offset(): Offset;
    function unary_sub(o: Offset): Offset;
}

implementation ExtConstants = external C++ base.constants {
    type Float;
    function nu(): Float;
    function dt(): Float;
    function dx(): Float;
}

concept DNFGenericBinopRule = {
    type E;
    type Array;
    type Index;

    function binop(lhs: E, rhs: E): E;
    function binop(lhs: E, rhs: Array): Array;
    function binop(lhs: Array, rhs: Array): Array;
    function psi(ix: Index, array: Array): E;

    // R1
    axiom binopArrayRule(ix: Index, lhs: Array, rhs: Array) {
        assert psi(ix, binop(lhs, rhs)) == binop(psi(ix, lhs), psi(ix, rhs));
    }

    // R2
    axiom binopScalarRule(ix: Index, lhs: E, rhs: Array) {
        assert psi(ix, binop(lhs, rhs)) == binop(lhs, psi(ix, rhs));
    }
}

concept DNFRules = {
    use DNFGenericBinopRule[ E => Float
                           , binop => _+_
                           , binopArrayRule => binopArrayRulePlus
                           , binopScalarRule => binopScalarRulePlus
                           ];
    use DNFGenericBinopRule[ E => Float
                           , binop => _*_
                           , binopArrayRule => binopArrayRuleMul
                           , binopScalarRule => binopScalarRuleMul
                           ];
    use DNFGenericBinopRule[ E => Float
                           , binop => binary_sub
                           , binopArrayRule => binopArrayRuleSub
                           , binopScalarRule => binopScalarRuleSub
                           ];
    // TODO: R3
    type Axis;
    type Offset;
    function rotate(array: Array, axis: Axis, offset: Offset): Array;
    function rotate_ix(ix: Index, axis: Axis, offset: Offset): Index;

    axiom rotateRule(ix: Index, array: Array, axis: Axis,
                     offset: Offset) {
        assert psi(ix, rotate(array, axis, offset)) ==
               psi(rotate_ix(ix, axis, offset), array);
    }
}

/// End of DNF base, start of experiments

concept OFLiftCores = {
    type Array;
    type Axis;
    type Nat;

    function nbCores(): Nat;

    function schedule_threaded(u: Array, v: Array, u0: Array,
                                        u1: Array, u2: Array, nbThreads: Nat): Array;
    function schedule(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array): Array;

    axiom liftCoresRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array) {
        var d = nbCores();

        assert schedule(u, v, u0, u1, u2) ==
               schedule_threaded(u, v, u0, u1, u2, d);
    }
}