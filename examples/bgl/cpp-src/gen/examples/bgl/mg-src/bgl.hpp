#include "base.hpp"
#include <cassert>


namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl {
struct IncidenceGraphWalk {
private:
    static string __string;
public:
    typedef string::String StringVertex;
    typedef hash_set<IncidenceGraphWalk::StringVertex>::HashSet VertexCollection;
    struct _emptyVertexCollection {
        IncidenceGraphWalk::VertexCollection operator()();
    };

    static IncidenceGraphWalk::_emptyVertexCollection emptyVertexCollection;
    typedef queue<IncidenceGraphWalk::StringVertex>::Queue VertexQueue;
private:
    static pair<IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::VertexQueue> __pair;
    static tuple_3<IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::VertexQueue, IncidenceGraphWalk::VertexCollection> __tuple_3;
public:
    struct _dequeue {
        IncidenceGraphWalk::VertexQueue operator()(const IncidenceGraphWalk::VertexQueue& q);
    };

    static IncidenceGraphWalk::_dequeue dequeue;
    struct _emptyQueue {
        IncidenceGraphWalk::VertexQueue operator()();
    };

    static IncidenceGraphWalk::_emptyQueue emptyQueue;
    struct _isQueueEmpty {
        bool operator()(const IncidenceGraphWalk::VertexQueue& q);
    };

    static IncidenceGraphWalk::_isQueueEmpty isQueueEmpty;
private:
    static edge<IncidenceGraphWalk::StringVertex> __edge;
    static hash_set<IncidenceGraphWalk::StringVertex> __hash_set0;
    static pprinter<IncidenceGraphWalk::StringVertex> __pprinter;
    static queue<IncidenceGraphWalk::StringVertex> __queue;
public:
    struct _enqueue {
        IncidenceGraphWalk::VertexQueue operator()(const IncidenceGraphWalk::VertexQueue& q, const IncidenceGraphWalk::StringVertex& e);
    };

    static IncidenceGraphWalk::_enqueue enqueue;
    struct _pprint {
        void operator()(const IncidenceGraphWalk::StringVertex& e);
    };

    static IncidenceGraphWalk::_pprint pprint;
    typedef pair<IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::VertexQueue>::Pair OuterStateTwople;
    struct _twople {
        IncidenceGraphWalk::OuterStateTwople operator()(const IncidenceGraphWalk::VertexCollection& a, const IncidenceGraphWalk::VertexQueue& b);
    };

    static IncidenceGraphWalk::_twople twople;
    typedef tuple_3<IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::VertexQueue, IncidenceGraphWalk::VertexCollection>::Tuple InnerStateTriple;
    struct _first {
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::OuterStateTwople& p);
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::InnerStateTriple& t);
        IncidenceGraphWalk::StringVertex operator()(const IncidenceGraphWalk::VertexQueue& q);
    };

    static IncidenceGraphWalk::_first first;
    struct _second {
        IncidenceGraphWalk::VertexQueue operator()(const IncidenceGraphWalk::OuterStateTwople& p);
        IncidenceGraphWalk::VertexQueue operator()(const IncidenceGraphWalk::InnerStateTriple& t);
    };

    static IncidenceGraphWalk::_second second;
    struct _third {
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::InnerStateTriple& t);
    };

    static IncidenceGraphWalk::_third third;
    struct _triple {
        IncidenceGraphWalk::InnerStateTriple operator()(const IncidenceGraphWalk::VertexCollection& a, const IncidenceGraphWalk::VertexQueue& b, const IncidenceGraphWalk::VertexCollection& c);
    };

    static IncidenceGraphWalk::_triple triple;
    typedef edge<IncidenceGraphWalk::StringVertex>::Edge Edge;
    typedef hash_set<IncidenceGraphWalk::Edge>::HashSet EdgeCollection;
    struct _emptyEdgeCollection {
        IncidenceGraphWalk::EdgeCollection operator()();
    };

    static IncidenceGraphWalk::_emptyEdgeCollection emptyEdgeCollection;
    struct _isCollectionEmpty {
        bool operator()(const IncidenceGraphWalk::VertexCollection& h);
        bool operator()(const IncidenceGraphWalk::EdgeCollection& h);
    };

    static IncidenceGraphWalk::_isCollectionEmpty isCollectionEmpty;
    struct _mapEdgeTarget {
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::EdgeCollection& dc);
    };

    static IncidenceGraphWalk::_mapEdgeTarget mapEdgeTarget;
private:
    static hash_set<IncidenceGraphWalk::Edge> __hash_set;
public:
    struct _addToCollection {
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::VertexCollection& h, const IncidenceGraphWalk::StringVertex& e);
        IncidenceGraphWalk::EdgeCollection operator()(const IncidenceGraphWalk::EdgeCollection& h, const IncidenceGraphWalk::Edge& e);
    };

    static IncidenceGraphWalk::_addToCollection addToCollection;
    struct _extractOneElement {
        IncidenceGraphWalk::StringVertex operator()(const IncidenceGraphWalk::VertexCollection& h);
        IncidenceGraphWalk::Edge operator()(const IncidenceGraphWalk::EdgeCollection& h);
    };

    static IncidenceGraphWalk::_extractOneElement extractOneElement;
    struct _isIn {
        bool operator()(const IncidenceGraphWalk::VertexCollection& h, const IncidenceGraphWalk::StringVertex& e);
        bool operator()(const IncidenceGraphWalk::EdgeCollection& h, const IncidenceGraphWalk::Edge& e);
    };

    static IncidenceGraphWalk::_isIn isIn;
    struct _make_edge {
        IncidenceGraphWalk::Edge operator()(const IncidenceGraphWalk::StringVertex& src, const IncidenceGraphWalk::StringVertex& tgt);
    };

    static IncidenceGraphWalk::_make_edge make_edge;
    struct _removeFromCollection {
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::VertexCollection& h, const IncidenceGraphWalk::StringVertex& e);
        IncidenceGraphWalk::EdgeCollection operator()(const IncidenceGraphWalk::EdgeCollection& h, const IncidenceGraphWalk::Edge& e);
    };

    static IncidenceGraphWalk::_removeFromCollection removeFromCollection;
    struct _source {
        IncidenceGraphWalk::StringVertex operator()(const IncidenceGraphWalk::Edge& e);
    };

    static IncidenceGraphWalk::_source source;
    struct _target {
        IncidenceGraphWalk::StringVertex operator()(const IncidenceGraphWalk::Edge& e);
    };

    typedef incidence_graph<IncidenceGraphWalk::Edge, IncidenceGraphWalk::EdgeCollection, IncidenceGraphWalk::StringVertex, IncidenceGraphWalk::_addToCollection, IncidenceGraphWalk::_emptyEdgeCollection, IncidenceGraphWalk::_extractOneElement, IncidenceGraphWalk::_isCollectionEmpty, IncidenceGraphWalk::_isIn, IncidenceGraphWalk::_make_edge, IncidenceGraphWalk::_removeFromCollection, IncidenceGraphWalk::_source, IncidenceGraphWalk::_target>::Graph Graph;
    struct _addToQueueLoop {
        void operator()(IncidenceGraphWalk::InnerStateTriple& state, const IncidenceGraphWalk::Graph& ctx);
    };

    static IncidenceGraphWalk::_addToQueueLoop addToQueueLoop;
    struct _addToQueueLoopCondition {
        bool operator()(const IncidenceGraphWalk::InnerStateTriple& st, const IncidenceGraphWalk::Graph& g);
    };

    static IncidenceGraphWalk::_addToQueueLoopCondition addToQueueLoopCondition;
    struct _addToQueueStep {
        void operator()(IncidenceGraphWalk::InnerStateTriple& st, const IncidenceGraphWalk::Graph& g);
    };

private:
    static while_loop<IncidenceGraphWalk::Graph, IncidenceGraphWalk::InnerStateTriple, IncidenceGraphWalk::_addToQueueLoopCondition, IncidenceGraphWalk::_addToQueueStep> __while_loop;
public:
    static IncidenceGraphWalk::_addToQueueStep addToQueueStep;
    struct _adjacentVertices {
        IncidenceGraphWalk::VertexCollection operator()(const IncidenceGraphWalk::Graph& g, const IncidenceGraphWalk::StringVertex& v);
    };

    static IncidenceGraphWalk::_adjacentVertices adjacentVertices;
    struct _bfs {
        void operator()(const IncidenceGraphWalk::Graph& g, const IncidenceGraphWalk::StringVertex& start);
    };

    static IncidenceGraphWalk::_bfs bfs;
    struct _mainLoop {
        void operator()(IncidenceGraphWalk::OuterStateTwople& state, const IncidenceGraphWalk::Graph& ctx);
    };

    static IncidenceGraphWalk::_mainLoop mainLoop;
    struct _mainLoopCondition {
        bool operator()(const IncidenceGraphWalk::OuterStateTwople& st, const IncidenceGraphWalk::Graph& _);
    };

    static IncidenceGraphWalk::_mainLoopCondition mainLoopCondition;
    struct _outEdges {
        IncidenceGraphWalk::EdgeCollection operator()(const IncidenceGraphWalk::Graph& g, const IncidenceGraphWalk::StringVertex& v);
    };

    static IncidenceGraphWalk::_outEdges outEdges;
    struct _visitVertex {
        void operator()(IncidenceGraphWalk::OuterStateTwople& st, const IncidenceGraphWalk::Graph& g);
    };

private:
    static while_loop<IncidenceGraphWalk::Graph, IncidenceGraphWalk::OuterStateTwople, IncidenceGraphWalk::_mainLoopCondition, IncidenceGraphWalk::_visitVertex> __while_loop0;
public:
    static IncidenceGraphWalk::_visitVertex visitVertex;
private:
    static incidence_graph<IncidenceGraphWalk::Edge, IncidenceGraphWalk::EdgeCollection, IncidenceGraphWalk::StringVertex, IncidenceGraphWalk::_addToCollection, IncidenceGraphWalk::_emptyEdgeCollection, IncidenceGraphWalk::_extractOneElement, IncidenceGraphWalk::_isCollectionEmpty, IncidenceGraphWalk::_isIn, IncidenceGraphWalk::_make_edge, IncidenceGraphWalk::_removeFromCollection, IncidenceGraphWalk::_source, IncidenceGraphWalk::_target> __incidence_graph;
    static map_function<IncidenceGraphWalk::Edge, IncidenceGraphWalk::StringVertex, IncidenceGraphWalk::EdgeCollection, IncidenceGraphWalk::VertexCollection, IncidenceGraphWalk::_addToCollection, IncidenceGraphWalk::_emptyVertexCollection, IncidenceGraphWalk::_extractOneElement, IncidenceGraphWalk::_target, IncidenceGraphWalk::_isCollectionEmpty, IncidenceGraphWalk::_removeFromCollection> __map_function;
public:
    static IncidenceGraphWalk::_target target;
};
} // examples
} // bgl
} // mg_src
} // bgl