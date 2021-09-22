#include "gen/examples/bgl/mg-src/bgl.hpp"


namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl {
    string IncidenceGraphWalk::__string;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_emptyVertexCollection::operator()() {
        return __hash_set0.nil();
    };


    IncidenceGraphWalk::_emptyVertexCollection IncidenceGraphWalk::emptyVertexCollection;

    pair<IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::VertexQueue> IncidenceGraphWalk::__pair;

    tuple_3<IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::VertexQueue, IncidenceGraphWalk::VertexCollection> IncidenceGraphWalk::__tuple_3;

    IncidenceGraphWalk::VertexQueue IncidenceGraphWalk::_dequeue::operator()(const IncidenceGraphWalk::VertexQueue& q) {
        return __queue.dequeue(q);
    };


    IncidenceGraphWalk::_dequeue IncidenceGraphWalk::dequeue;

    IncidenceGraphWalk::VertexQueue IncidenceGraphWalk::_emptyQueue::operator()() {
        return __queue.nil();
    };


    IncidenceGraphWalk::_emptyQueue IncidenceGraphWalk::emptyQueue;

    bool IncidenceGraphWalk::_isQueueEmpty::operator()(const IncidenceGraphWalk::VertexQueue& q) {
        return __queue.empty(q);
    };


    IncidenceGraphWalk::_isQueueEmpty IncidenceGraphWalk::isQueueEmpty;

    edge<IncidenceGraphWalk::StringVertex> IncidenceGraphWalk::__edge;

    hash_set<IncidenceGraphWalk::StringVertex> IncidenceGraphWalk::__hash_set0;

    pprinter<IncidenceGraphWalk::StringVertex> IncidenceGraphWalk::__pprinter;

    queue<IncidenceGraphWalk::StringVertex> IncidenceGraphWalk::__queue;

    IncidenceGraphWalk::VertexQueue IncidenceGraphWalk::_enqueue::operator()(const IncidenceGraphWalk::VertexQueue& q, const IncidenceGraphWalk::StringVertex& e) {
        return __queue.enqueue(q, e);
    };


    IncidenceGraphWalk::_enqueue IncidenceGraphWalk::enqueue;

    void IncidenceGraphWalk::_pprint::operator()(const IncidenceGraphWalk::StringVertex& e) {
        return __pprinter.pprint(e);
    };


    IncidenceGraphWalk::_pprint IncidenceGraphWalk::pprint;

    IncidenceGraphWalk::OuterStateTwople IncidenceGraphWalk::_twople::operator()(const IncidenceGraphWalk::VertexCollection& a, const IncidenceGraphWalk::VertexQueue& b) {
        return __pair.make_pair(a, b);
    };


    IncidenceGraphWalk::_twople IncidenceGraphWalk::twople;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_first::operator()(const IncidenceGraphWalk::OuterStateTwople& p) {
        return __pair.first(p);
    };

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_first::operator()(const IncidenceGraphWalk::InnerStateTriple& t) {
        return __tuple_3.first(t);
    };

    IncidenceGraphWalk::StringVertex IncidenceGraphWalk::_first::operator()(const IncidenceGraphWalk::VertexQueue& q) {
        return __queue.first(q);
    };


    IncidenceGraphWalk::_first IncidenceGraphWalk::first;

    IncidenceGraphWalk::VertexQueue IncidenceGraphWalk::_second::operator()(const IncidenceGraphWalk::OuterStateTwople& p) {
        return __pair.second(p);
    };

    IncidenceGraphWalk::VertexQueue IncidenceGraphWalk::_second::operator()(const IncidenceGraphWalk::InnerStateTriple& t) {
        return __tuple_3.second(t);
    };


    IncidenceGraphWalk::_second IncidenceGraphWalk::second;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_third::operator()(const IncidenceGraphWalk::InnerStateTriple& t) {
        return __tuple_3.third(t);
    };


    IncidenceGraphWalk::_third IncidenceGraphWalk::third;

    IncidenceGraphWalk::InnerStateTriple IncidenceGraphWalk::_triple::operator()(const IncidenceGraphWalk::VertexCollection& a, const IncidenceGraphWalk::VertexQueue& b, const IncidenceGraphWalk::VertexCollection& c) {
        return __tuple_3.make_tuple(a, b, c);
    };


    IncidenceGraphWalk::_triple IncidenceGraphWalk::triple;

    IncidenceGraphWalk::EdgeCollection IncidenceGraphWalk::_emptyEdgeCollection::operator()() {
        return __hash_set.nil();
    };


    IncidenceGraphWalk::_emptyEdgeCollection IncidenceGraphWalk::emptyEdgeCollection;

    bool IncidenceGraphWalk::_isCollectionEmpty::operator()(const IncidenceGraphWalk::VertexCollection& h) {
        return __hash_set0.empty(h);
    };

    bool IncidenceGraphWalk::_isCollectionEmpty::operator()(const IncidenceGraphWalk::EdgeCollection& h) {
        return __hash_set.empty(h);
    };


    IncidenceGraphWalk::_isCollectionEmpty IncidenceGraphWalk::isCollectionEmpty;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_mapEdgeTarget::operator()(const IncidenceGraphWalk::EdgeCollection& dc) {
        return __map_function.map(dc);
    };


    IncidenceGraphWalk::_mapEdgeTarget IncidenceGraphWalk::mapEdgeTarget;

    hash_set<IncidenceGraphWalk::Edge> IncidenceGraphWalk::__hash_set;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_addToCollection::operator()(const IncidenceGraphWalk::VertexCollection& h, const IncidenceGraphWalk::StringVertex& e) {
        return __hash_set0.insert(h, e);
    };

    IncidenceGraphWalk::EdgeCollection IncidenceGraphWalk::_addToCollection::operator()(const IncidenceGraphWalk::EdgeCollection& h, const IncidenceGraphWalk::Edge& e) {
        return __hash_set.insert(h, e);
    };


    IncidenceGraphWalk::_addToCollection IncidenceGraphWalk::addToCollection;

    IncidenceGraphWalk::StringVertex IncidenceGraphWalk::_extractOneElement::operator()(const IncidenceGraphWalk::VertexCollection& h) {
        return __hash_set0.min(h);
    };

    IncidenceGraphWalk::Edge IncidenceGraphWalk::_extractOneElement::operator()(const IncidenceGraphWalk::EdgeCollection& h) {
        return __hash_set.min(h);
    };


    IncidenceGraphWalk::_extractOneElement IncidenceGraphWalk::extractOneElement;

    bool IncidenceGraphWalk::_isIn::operator()(const IncidenceGraphWalk::VertexCollection& h, const IncidenceGraphWalk::StringVertex& e) {
        return __hash_set0.member(h, e);
    };

    bool IncidenceGraphWalk::_isIn::operator()(const IncidenceGraphWalk::EdgeCollection& h, const IncidenceGraphWalk::Edge& e) {
        return __hash_set.member(h, e);
    };


    IncidenceGraphWalk::_isIn IncidenceGraphWalk::isIn;

    IncidenceGraphWalk::Edge IncidenceGraphWalk::_make_edge::operator()(const IncidenceGraphWalk::StringVertex& src, const IncidenceGraphWalk::StringVertex& tgt) {
        return __edge.make_edge(src, tgt);
    };


    IncidenceGraphWalk::_make_edge IncidenceGraphWalk::make_edge;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_removeFromCollection::operator()(const IncidenceGraphWalk::VertexCollection& h, const IncidenceGraphWalk::StringVertex& e) {
        return __hash_set0.remove(h, e);
    };

    IncidenceGraphWalk::EdgeCollection IncidenceGraphWalk::_removeFromCollection::operator()(const IncidenceGraphWalk::EdgeCollection& h, const IncidenceGraphWalk::Edge& e) {
        return __hash_set.remove(h, e);
    };


    IncidenceGraphWalk::_removeFromCollection IncidenceGraphWalk::removeFromCollection;

    IncidenceGraphWalk::StringVertex IncidenceGraphWalk::_source::operator()(const IncidenceGraphWalk::Edge& e) {
        return __edge.source(e);
    };


    IncidenceGraphWalk::_source IncidenceGraphWalk::source;

    IncidenceGraphWalk::StringVertex IncidenceGraphWalk::_target::operator()(const IncidenceGraphWalk::Edge& e) {
        return __edge.target(e);
    };


    void IncidenceGraphWalk::_addToQueueLoop::operator()(IncidenceGraphWalk::InnerStateTriple& state, const IncidenceGraphWalk::Graph& ctx) {
        return __while_loop.repeat(state, ctx);
    };


    IncidenceGraphWalk::_addToQueueLoop IncidenceGraphWalk::addToQueueLoop;

    bool IncidenceGraphWalk::_addToQueueLoopCondition::operator()(const IncidenceGraphWalk::InnerStateTriple& st, const IncidenceGraphWalk::Graph& g) {
        return !IncidenceGraphWalk::isCollectionEmpty(IncidenceGraphWalk::first(st));
    };


    IncidenceGraphWalk::_addToQueueLoopCondition IncidenceGraphWalk::addToQueueLoopCondition;

    void IncidenceGraphWalk::_addToQueueStep::operator()(IncidenceGraphWalk::InnerStateTriple& st, const IncidenceGraphWalk::Graph& g) {
        IncidenceGraphWalk::VertexQueue vertexQueue = IncidenceGraphWalk::second(st);
        IncidenceGraphWalk::VertexCollection visitedVertices = IncidenceGraphWalk::third(st);
        IncidenceGraphWalk::StringVertex nextVertex = IncidenceGraphWalk::extractOneElement(IncidenceGraphWalk::first(st));
        IncidenceGraphWalk::VertexQueue newVertexQueue = IncidenceGraphWalk::isIn(visitedVertices, nextVertex) ? IncidenceGraphWalk::second(st) : IncidenceGraphWalk::enqueue(IncidenceGraphWalk::second(st), nextVertex);
        IncidenceGraphWalk::VertexCollection newVisitedVertices = IncidenceGraphWalk::addToCollection(visitedVertices, nextVertex);
        st = IncidenceGraphWalk::triple(IncidenceGraphWalk::removeFromCollection(IncidenceGraphWalk::first(st), nextVertex), newVertexQueue, newVisitedVertices);
    };


    while_loop<IncidenceGraphWalk::Graph, IncidenceGraphWalk::InnerStateTriple, IncidenceGraphWalk::_addToQueueLoopCondition, IncidenceGraphWalk::_addToQueueStep> IncidenceGraphWalk::__while_loop;

    IncidenceGraphWalk::_addToQueueStep IncidenceGraphWalk::addToQueueStep;

    IncidenceGraphWalk::VertexCollection IncidenceGraphWalk::_adjacentVertices::operator()(const IncidenceGraphWalk::Graph& g, const IncidenceGraphWalk::StringVertex& v) {
        return IncidenceGraphWalk::mapEdgeTarget(IncidenceGraphWalk::outEdges(g, v));
    };


    IncidenceGraphWalk::_adjacentVertices IncidenceGraphWalk::adjacentVertices;

    void IncidenceGraphWalk::_bfs::operator()(const IncidenceGraphWalk::Graph& g, const IncidenceGraphWalk::StringVertex& start) {
        IncidenceGraphWalk::VertexQueue queue = IncidenceGraphWalk::enqueue(IncidenceGraphWalk::emptyQueue(), start);
        IncidenceGraphWalk::VertexCollection visitedVertices = IncidenceGraphWalk::emptyVertexCollection();
        IncidenceGraphWalk::OuterStateTwople outerState = IncidenceGraphWalk::twople(visitedVertices, queue);
        IncidenceGraphWalk::mainLoop(outerState, g);
    };


    IncidenceGraphWalk::_bfs IncidenceGraphWalk::bfs;

    void IncidenceGraphWalk::_mainLoop::operator()(IncidenceGraphWalk::OuterStateTwople& state, const IncidenceGraphWalk::Graph& ctx) {
        return __while_loop0.repeat(state, ctx);
    };


    IncidenceGraphWalk::_mainLoop IncidenceGraphWalk::mainLoop;

    bool IncidenceGraphWalk::_mainLoopCondition::operator()(const IncidenceGraphWalk::OuterStateTwople& st, const IncidenceGraphWalk::Graph& _) {
        IncidenceGraphWalk::VertexQueue vertexQueue = IncidenceGraphWalk::second(st);
        return !IncidenceGraphWalk::isQueueEmpty(vertexQueue);
    };


    IncidenceGraphWalk::_mainLoopCondition IncidenceGraphWalk::mainLoopCondition;

    IncidenceGraphWalk::EdgeCollection IncidenceGraphWalk::_outEdges::operator()(const IncidenceGraphWalk::Graph& g, const IncidenceGraphWalk::StringVertex& v) {
        return __incidence_graph.outEdges(g, v);
    };


    IncidenceGraphWalk::_outEdges IncidenceGraphWalk::outEdges;

    void IncidenceGraphWalk::_visitVertex::operator()(IncidenceGraphWalk::OuterStateTwople& st, const IncidenceGraphWalk::Graph& g) {
        IncidenceGraphWalk::StringVertex v = IncidenceGraphWalk::first(IncidenceGraphWalk::second(st));
        IncidenceGraphWalk::pprint(v);
        IncidenceGraphWalk::VertexCollection visitedVertices = IncidenceGraphWalk::addToCollection(IncidenceGraphWalk::first(st), v);
        IncidenceGraphWalk::InnerStateTriple innerState = IncidenceGraphWalk::triple(IncidenceGraphWalk::adjacentVertices(g, v), IncidenceGraphWalk::second(st), visitedVertices);
        IncidenceGraphWalk::addToQueueLoop(innerState, g);
        IncidenceGraphWalk::VertexCollection newVisitedVertices = IncidenceGraphWalk::third(innerState);
        IncidenceGraphWalk::VertexQueue newQueue = IncidenceGraphWalk::dequeue(IncidenceGraphWalk::second(innerState));
        st = IncidenceGraphWalk::twople(newVisitedVertices, newQueue);
    };


    while_loop<IncidenceGraphWalk::Graph, IncidenceGraphWalk::OuterStateTwople, IncidenceGraphWalk::_mainLoopCondition, IncidenceGraphWalk::_visitVertex> IncidenceGraphWalk::__while_loop0;

    IncidenceGraphWalk::_visitVertex IncidenceGraphWalk::visitVertex;

    incidence_graph<IncidenceGraphWalk::Edge, IncidenceGraphWalk::EdgeCollection, IncidenceGraphWalk::StringVertex, IncidenceGraphWalk::_addToCollection, IncidenceGraphWalk::_emptyEdgeCollection, IncidenceGraphWalk::_extractOneElement, IncidenceGraphWalk::_isCollectionEmpty, IncidenceGraphWalk::_isIn, IncidenceGraphWalk::_make_edge, IncidenceGraphWalk::_removeFromCollection, IncidenceGraphWalk::_source, IncidenceGraphWalk::_target> IncidenceGraphWalk::__incidence_graph;

    map_function<IncidenceGraphWalk::Edge, IncidenceGraphWalk::StringVertex, IncidenceGraphWalk::EdgeCollection, IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::_addToCollection, IncidenceGraphWalk::_emptyVertexCollection, IncidenceGraphWalk::_extractOneElement, IncidenceGraphWalk::_target, IncidenceGraphWalk::_isCollectionEmpty, IncidenceGraphWalk::_removeFromCollection> IncidenceGraphWalk::__map_function;

    IncidenceGraphWalk::_target IncidenceGraphWalk::target;

} // examples
} // bgl
} // mg_src
} // bgl