package examples.pde.mg-src.pde-cpp imports examples.pde.mg-src.pde-pad;

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
    v0 = substep(v0, u0, u0, u1, u2);
    v1 = substep(v1, u1, u0, u1, u2);
    v2 = substep(v2, u2, u0, u1, u2);
    u0 = substep(u0, v0, u0, u1, u2);
    u1 = substep(u1, v1, u0, u1, u2);
    u2 = substep(u2, v2, u0, u1, u2);
  }

  require function nu(): Float;
  require function dt(): Float;
  require function dx(): Float;

  type Index;

  require function psi(ix: Index, array: Array): Float;

  function substep(u: Array, v: Array, u0: Array,
                   u1: Array, u2: Array) : Array =
    u + dt()/(two(): Float) * (nu() * ((one(): Float)/dx()/dx() *
      (rotate(v, zero(), -one(): Offset) +
       rotate(v, zero(), one(): Offset) +
       rotate(v, one(): Axis, -one(): Offset) +
       rotate(v, one(): Axis, one(): Offset) +
       rotate(v, two(): Axis, -one(): Offset) +
       rotate(v, two(): Axis, one(): Offset)) -
    three() * (two(): Float)/dx()/dx() * u0) -
    (one(): Float)/(two(): Float)/dx() *
      ((rotate(v, zero(), one(): Offset) -
        rotate(v, zero(), -one(): Offset)) * u0 +
       (rotate(v, one(): Axis, one(): Offset) -
        rotate(v, one(): Axis, -one(): Offset)) * u1 +
       (rotate(v, two(): Axis, one(): Offset) -
        rotate(v, two(): Axis, -one(): Offset)) * u2));
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

program PDEProgramDNF = {
  use (rewrite
        (rewrite
          (generate ToIxwiseGenerator in BasePDEProgram)
      with DNFRules 20)
    with ToIxwise 1);

  use ExtExtendMissingBypass;
}
// program PDEProgram2 = {
//   use (rewrite PDEProgramDNF with OFLiftCores 1);
//   use ExtExtendLiftCores;
// }

program PDEProgramPadded = {
  use (rewrite PDEProgramDNF with OFPad 1);
  use ExtExtendPadding;
}

program PDEProgram3D = {
  use (rewrite
        (rewrite
          (rewrite
            (generate OFSpecializeSubstepGenerator in PDEProgramDNF)
          with OFSpecializePsi 10)
        with OFReduceMakeIxRotate 20)
      with SwitchSchedule[ sourceSchedule => schedule
                         , targetSchedule => schedule3D
                         ] 1);

  use ExtNeededFns[schedule3DPadded => schedule3D];
}

program PDEProgram3DPadded = {
  use (rewrite
        (rewrite
          (rewrite
            (rewrite
              (generate OFSpecializeSubstepGenerator in PDEProgramDNF)
            with OFSpecializePsi 10)
          with OFReduceMakeIxRotate 20)
        with OFPad[schedulePadded =>
                   schedule3DPadded] 1)
      with OFEliminateModuloPadding 10);

  use ExtNeededFns; // pulling in psi, schedules, etc...
}

program PDEProgram = PDEProgramDNF;

concept ToIxwiseGenerator = {
  type Array;
  type Float;
  type Index;

  function substepIx(u: Array, v: Array, u0: Array,
                     u1: Array, u2: Array, ix: Index): Float;

  function substep(u: Array, v: Array, u0: Array,
                   u1: Array, u2: Array): Array;

  function psi(ix: Index, array: Array): Float;

  axiom toIxwiseGenerator(u: Array, v: Array, u0: Array,
              u1: Array, u2: Array, ix: Index) {
    assert substepIx(u, v, u0, u1, u2, ix) ==
           psi(ix, substep(u, v, u0, u1, u2));
  }
}

concept ToIxwise = {
  type Array;

  function substep(u: Array, v: Array, u0: Array,
           u1: Array, u2: Array): Array;

  function schedule(u: Array, v: Array,
                    u0: Array, u1: Array, u2: Array): Array;

  axiom toIxwiseRule(u: Array, v: Array, u0: Array,
             u1: Array, u2: Array) {
    assert substep(u, v, u0, u1, u2) ==
           schedule(u, v, u0, u1, u2);
  }
}

implementation ExtNeededFns = external C++ base.specialize_psi_ops_2 {
  require type Axis;
  require type Index;
  require type Offset;
  require type ScalarIndex;
  require type Array;
  require type Float;
  require function substepIx3D(u: Array, v: Array, u0: Array,
                  u1: Array, u2: Array, i: ScalarIndex,
                  j: ScalarIndex, k: ScalarIndex): Float;


  procedure refillPadding(upd a: Array);

  /* OF Specialize Psi extension */
  function psi(i: ScalarIndex,
               j: ScalarIndex,
               k: ScalarIndex,
               a: Array): Float;
  function schedule3DPadded(u: Array,
                            v: Array,
                            u0: Array,
                            u1: Array,
                            u2: Array): Array;

  /* OF Pad extension */
  function rotateIxPadded(ix: Index, axis: Axis, o: Offset): Index;

  /* OF Reduce MakeIx projections */
  function ix0(ix: Index): ScalarIndex;
  function ix1(ix: Index): ScalarIndex;
  function ix2(ix: Index): ScalarIndex;

  /* OF Reduce MakeIx Rotate extension */
  type AxisLength;

  function _+_(six: ScalarIndex, o: Offset): ScalarIndex;
  function _%_(six: ScalarIndex, sc: AxisLength): ScalarIndex;
  function shape0(): AxisLength;
  function shape1(): AxisLength;
  function shape2(): AxisLength;
}


implementation ExtExtendMissingBypass = external C++ base.forall_ops {
  require type Float;
  require type Array;
  require type Offset;
  require type Axis;
  require type Index;
  require type Nat;

  require function substepIx(u: Array, v: Array,
                u0: Array, u1: Array, u2: Array, ix: Index): Float;

  function schedule(u: Array, v: Array,
                 u0: Array, u1: Array, u2: Array): Array;

  type ScalarIndex;
  function mkIx(a: ScalarIndex, b: ScalarIndex, c: ScalarIndex): Index;
}

implementation ExtExtendPadding = external C++ base.forall_ops {
  require type Float;
  require type Array;
  require type Offset;
  require type Axis;
  require type Index;
  require type Nat;

  require function substepIx(u: Array, v: Array,
                u0: Array, u1: Array, u2: Array, ix: Index): Float;

  /* OF Pad extension */
  procedure refillPadding(upd a: Array);

  function schedulePadded(u: Array, v: Array,
    u0: Array, u1: Array, u2: Array): Array;

  function rotateIxPadded(ix: Index, axis: Axis, offset: Offset): Index;
}

implementation ExtExtendLiftCores = external C++ base.forall_ops {
  require type Float;
  require type Array;
  require type Offset;
  require type Axis;
  require type Index;
  require type Nat;

  require function substepIx(u: Array, v: Array,
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
  function rotateIx(ix: Index, axis: Axis, o: Offset): Index;

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
  function rotateIx(ix: Index, axis: Axis, offset: Offset): Index;

  axiom rotateRule(ix: Index, array: Array, axis: Axis,
           offset: Offset) {
    assert psi(ix, rotate(array, axis, offset)) ==
         psi(rotateIx(ix, axis, offset), array);
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

concept SwitchSchedule = {
  type Array;

  function sourceSchedule(u: Array, v: Array, u0: Array, u1: Array, u2: Array)
    : Array;

  function targetSchedule(u: Array, v: Array, u0: Array, u1: Array, u2: Array)
    : Array;

  axiom switchScheduleRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array) {
    assert sourceSchedule(u, v, u0, u1, u2) == targetSchedule(u, v, u0, u1, u2);
  }
}