package examples.bgl.mg-src.v2.queue;

concept Queue = {
    require type A;
    type Queue;

    predicate isEmpty(q: Queue);

    function push(a: A, q: Queue): Queue;
    function pop(q: Queue): Queue guard !isEmpty(q);
    function front(q: Queue): A;
}
