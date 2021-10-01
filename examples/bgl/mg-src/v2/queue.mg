package examples.bgl.mg-src.v2.queue;

concept Queue = {
    type Queue;
    type A;

    predicate isEmpty(q: Queue);

    function push(a: A, q: Queue): Queue;
    function pop(q: Queue): Queue;
    function front(q: Queue): A;
}
