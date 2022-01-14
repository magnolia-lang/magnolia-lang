package examples.bgl.mg-src.vector;

concept Vector = {
    require type A;
    type Vector;

    function empty(): Vector;
    procedure pushBack(obs a: A, upd v: Vector);
}
