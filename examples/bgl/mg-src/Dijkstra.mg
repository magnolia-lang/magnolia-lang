package examples.bgl.mg-src.Dijkstra
    imports examples.bgl.mg-src.BasicConcepts;

concept Distance = {
    type Distance;
    type Point;

    function distance(p1: Point, p2: Point): Distance;
}

implementation Dijkstra = {
    use Queue;
}
