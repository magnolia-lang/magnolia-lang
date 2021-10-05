package examples.bgl_v2.mg-src.queue
    imports examples.bgl_v2.mg-src.property_map;

concept Queue = {
    require type A;
    type Queue;

    predicate isEmpty(q: Queue);

    function empty(): Queue;
    function push(a: A, q: Queue): Queue;
    function pop(q: Queue): Queue guard !isEmpty(q);
    function front(q: Queue): A;
}

concept UpdateablePriorityQueue = {
    require type A;
    require type Priority;

    use Queue[ Queue => PriorityQueue ];
    use ReadPropertyMap[ Key => A
                       , Value => Priority
                       , PropertyMap => PriorityMap
                       ];

    function update(pm: PriorityMap, a: A, pq: PriorityQueue): PriorityQueue;
}
