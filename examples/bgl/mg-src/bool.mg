package examples.bgl.mg-src.bool;

concept Bool = {
  type Bool;

  function btrue(): Bool;
  function bfalse(): Bool;

  axiom boolHasTwoValues(b: Bool) {
    assert btrue() != bfalse();
    assert b == btrue() || b == bfalse();
  }
}