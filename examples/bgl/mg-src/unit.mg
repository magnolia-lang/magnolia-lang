package examples.bgl.mg-src.unit;

concept Unit = {
  type Unit;
  function unit(): Unit;

  axiom contractible(u1: Unit, u2: Unit) {
    assert u1 == u2;
  }
}