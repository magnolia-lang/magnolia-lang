#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
struct CppBFSTestVisitor {
private:
    static color_marker __color_marker;
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef list<CppBFSTestVisitor::Vertex>::List VertexList;
    struct _emptyVertexList {
        CppBFSTestVisitor::VertexList operator()();
    };

    static CppBFSTestVisitor::_emptyVertexList emptyVertexList;
private:
    static edge<CppBFSTestVisitor::Vertex> __edge;
    static fifo_queue<CppBFSTestVisitor::Vertex> __fifo_queue;
    static list<CppBFSTestVisitor::Vertex> __list0;
public:
    typedef base_types::Int Int;
    typedef fifo_queue<CppBFSTestVisitor::Vertex>::FIFOQueue FIFOQueue;
    struct _empty {
        CppBFSTestVisitor::FIFOQueue operator()();
    };

    static CppBFSTestVisitor::_empty empty;
    struct _front {
        CppBFSTestVisitor::Vertex operator()(const CppBFSTestVisitor::FIFOQueue& q);
    };

    static CppBFSTestVisitor::_front front;
    struct _isEmptyQueue {
        bool operator()(const CppBFSTestVisitor::FIFOQueue& q);
    };

    static CppBFSTestVisitor::_isEmptyQueue isEmptyQueue;
    struct _pop {
        CppBFSTestVisitor::FIFOQueue operator()(const CppBFSTestVisitor::FIFOQueue& q);
    };

    static CppBFSTestVisitor::_pop pop;
    struct _push {
        CppBFSTestVisitor::FIFOQueue operator()(const CppBFSTestVisitor::Vertex& a, const CppBFSTestVisitor::FIFOQueue& q);
    };

    static CppBFSTestVisitor::_push push;
    typedef edge<CppBFSTestVisitor::Vertex>::Edge Edge;
    typedef list<CppBFSTestVisitor::Edge>::List EdgeList;
    struct _emptyEdgeList {
        CppBFSTestVisitor::EdgeList operator()();
    };

    static CppBFSTestVisitor::_emptyEdgeList emptyEdgeList;
    struct _isEmpty {
        bool operator()(const CppBFSTestVisitor::EdgeList& l);
        bool operator()(const CppBFSTestVisitor::VertexList& l);
    };

    static CppBFSTestVisitor::_isEmpty isEmpty;
    struct _tail {
        CppBFSTestVisitor::EdgeList operator()(const CppBFSTestVisitor::EdgeList& l);
        CppBFSTestVisitor::VertexList operator()(const CppBFSTestVisitor::VertexList& l);
    };

    static CppBFSTestVisitor::_tail tail;
private:
    static list<CppBFSTestVisitor::Edge> __list;
public:
    struct _cons {
        CppBFSTestVisitor::EdgeList operator()(const CppBFSTestVisitor::Edge& a, const CppBFSTestVisitor::EdgeList& l);
        CppBFSTestVisitor::VertexList operator()(const CppBFSTestVisitor::Vertex& a, const CppBFSTestVisitor::VertexList& l);
    };

    static CppBFSTestVisitor::_cons cons;
    struct _head {
        CppBFSTestVisitor::Edge operator()(const CppBFSTestVisitor::EdgeList& l);
        CppBFSTestVisitor::Vertex operator()(const CppBFSTestVisitor::VertexList& l);
    };

    static CppBFSTestVisitor::_head head;
    struct _makeEdge {
        CppBFSTestVisitor::Edge operator()(const CppBFSTestVisitor::Vertex& s, const CppBFSTestVisitor::Vertex& t);
    };

    static CppBFSTestVisitor::_makeEdge makeEdge;
    struct _src {
        CppBFSTestVisitor::Vertex operator()(const CppBFSTestVisitor::Edge& e);
    };

    static CppBFSTestVisitor::_src src;
    struct _tgt {
        CppBFSTestVisitor::Vertex operator()(const CppBFSTestVisitor::Edge& e);
    };

    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Edge, CppBFSTestVisitor::EdgeList, CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyEdgeList, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_makeEdge, CppBFSTestVisitor::_src, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tgt>::Graph Graph;
    typedef pair<CppBFSTestVisitor::Graph, CppBFSTestVisitor::Vertex>::Pair InnerLoopContext;
private:
    static pair<CppBFSTestVisitor::Graph, CppBFSTestVisitor::Vertex> __pair;
public:
    struct _breadthFirstSearch {
        CppBFSTestVisitor::VertexList operator()(const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::Vertex& start, const CppBFSTestVisitor::VertexList& init);
    };

    static CppBFSTestVisitor::_breadthFirstSearch breadthFirstSearch;
    struct _defaultAction {
        void operator()(const CppBFSTestVisitor::Vertex& edgeOrVertex, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexList& a);
        void operator()(const CppBFSTestVisitor::Edge& edgeOrVertex, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexList& a);
    };

    static CppBFSTestVisitor::_defaultAction defaultAction;
    struct _discoverVertex {
        void operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexList& a);
    };

    static CppBFSTestVisitor::_discoverVertex discoverVertex;
    struct _makeInnerLoopContext {
        CppBFSTestVisitor::InnerLoopContext operator()(const CppBFSTestVisitor::Graph& a, const CppBFSTestVisitor::Vertex& b);
    };

    static CppBFSTestVisitor::_makeInnerLoopContext makeInnerLoopContext;
    struct _outEdges {
        CppBFSTestVisitor::EdgeList operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g);
    };

    static CppBFSTestVisitor::_outEdges outEdges;
    struct _vertices {
        CppBFSTestVisitor::VertexList operator()(const CppBFSTestVisitor::Graph& g);
    };

    static CppBFSTestVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Edge, CppBFSTestVisitor::EdgeList, CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyEdgeList, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_makeEdge, CppBFSTestVisitor::_src, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tgt>::VertexCount VertexCount;
    struct _numVertices {
        CppBFSTestVisitor::VertexCount operator()(const CppBFSTestVisitor::Graph& g);
    };

    static CppBFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        CppBFSTestVisitor::VertexCount operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g);
    };

    static CppBFSTestVisitor::_outDegree outDegree;
private:
    static incidence_and_vertex_list_graph<CppBFSTestVisitor::Edge, CppBFSTestVisitor::EdgeList, CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyEdgeList, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_makeEdge, CppBFSTestVisitor::_src, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tgt> __incidence_and_vertex_list_graph;
public:
    static CppBFSTestVisitor::_tgt tgt;
    typedef color_marker::Color Color;
    typedef read_write_property_map<CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::Color, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_tail>::PropertyMap ColorPropertyMap;
    typedef triplet<CppBFSTestVisitor::VertexList, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap>::Triplet OuterLoopState;
    typedef pair<CppBFSTestVisitor::OuterLoopState, CppBFSTestVisitor::EdgeList>::Pair InnerLoopState;
    struct _bfsInnerLoopCond {
        bool operator()(const CppBFSTestVisitor::InnerLoopState& state, const CppBFSTestVisitor::InnerLoopContext& ctx);
    };

    static CppBFSTestVisitor::_bfsInnerLoopCond bfsInnerLoopCond;
    struct _bfsInnerLoopRepeat {
        void operator()(CppBFSTestVisitor::InnerLoopState& s, const CppBFSTestVisitor::InnerLoopContext& c);
    };

    static CppBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        void operator()(CppBFSTestVisitor::InnerLoopState& state, const CppBFSTestVisitor::InnerLoopContext& ctx);
    };

private:
    static while_loop<CppBFSTestVisitor::InnerLoopContext, CppBFSTestVisitor::InnerLoopState, CppBFSTestVisitor::_bfsInnerLoopCond, CppBFSTestVisitor::_bfsInnerLoopStep> __while_loop0;
public:
    static CppBFSTestVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
private:
    static pair<CppBFSTestVisitor::OuterLoopState, CppBFSTestVisitor::EdgeList> __pair0;
public:
    struct _bfsOuterLoopCond {
        bool operator()(const CppBFSTestVisitor::OuterLoopState& state, const CppBFSTestVisitor::Graph& g);
    };

    static CppBFSTestVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        void operator()(CppBFSTestVisitor::OuterLoopState& s, const CppBFSTestVisitor::Graph& c);
    };

    static CppBFSTestVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        void operator()(CppBFSTestVisitor::OuterLoopState& state, const CppBFSTestVisitor::Graph& g);
    };

private:
    static while_loop<CppBFSTestVisitor::Graph, CppBFSTestVisitor::OuterLoopState, CppBFSTestVisitor::_bfsOuterLoopCond, CppBFSTestVisitor::_bfsOuterLoopStep> __while_loop;
public:
    static CppBFSTestVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _first {
        CppBFSTestVisitor::Graph operator()(const CppBFSTestVisitor::InnerLoopContext& p);
        CppBFSTestVisitor::OuterLoopState operator()(const CppBFSTestVisitor::InnerLoopState& p);
        CppBFSTestVisitor::VertexList operator()(const CppBFSTestVisitor::OuterLoopState& p);
    };

    static CppBFSTestVisitor::_first first;
    struct _makeInnerLoopState {
        CppBFSTestVisitor::InnerLoopState operator()(const CppBFSTestVisitor::OuterLoopState& a, const CppBFSTestVisitor::EdgeList& b);
    };

    static CppBFSTestVisitor::_makeInnerLoopState makeInnerLoopState;
    struct _projectionBehaviorPair {
        void operator()(const CppBFSTestVisitor::Graph& a, const CppBFSTestVisitor::Vertex& b);
        void operator()(const CppBFSTestVisitor::OuterLoopState& a, const CppBFSTestVisitor::EdgeList& b);
    };

    static CppBFSTestVisitor::_projectionBehaviorPair projectionBehaviorPair;
    struct _second {
        CppBFSTestVisitor::Vertex operator()(const CppBFSTestVisitor::InnerLoopContext& p);
        CppBFSTestVisitor::EdgeList operator()(const CppBFSTestVisitor::InnerLoopState& p);
        CppBFSTestVisitor::FIFOQueue operator()(const CppBFSTestVisitor::OuterLoopState& p);
    };

    static CppBFSTestVisitor::_second second;
    struct _whileLoopBehavior {
        void operator()(const CppBFSTestVisitor::OuterLoopState& s, const CppBFSTestVisitor::Graph& c);
        void operator()(const CppBFSTestVisitor::InnerLoopState& s, const CppBFSTestVisitor::InnerLoopContext& c);
    };

    static CppBFSTestVisitor::_whileLoopBehavior whileLoopBehavior;
private:
    static triplet<CppBFSTestVisitor::VertexList, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap> __triplet;
public:
    struct _breadthFirstVisit {
        void operator()(const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::Vertex& s, CppBFSTestVisitor::VertexList& a, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c);
    };

    static CppBFSTestVisitor::_breadthFirstVisit breadthFirstVisit;
    struct _emptyMap {
        CppBFSTestVisitor::ColorPropertyMap operator()();
    };

    static CppBFSTestVisitor::_emptyMap emptyMap;
    struct _makeOuterLoopState {
        CppBFSTestVisitor::OuterLoopState operator()(const CppBFSTestVisitor::VertexList& a, const CppBFSTestVisitor::FIFOQueue& b, const CppBFSTestVisitor::ColorPropertyMap& c);
    };

    static CppBFSTestVisitor::_makeOuterLoopState makeOuterLoopState;
    struct _projectionBehaviorTriplet {
        void operator()(const CppBFSTestVisitor::VertexList& a, const CppBFSTestVisitor::FIFOQueue& b, const CppBFSTestVisitor::ColorPropertyMap& c);
    };

    static CppBFSTestVisitor::_projectionBehaviorTriplet projectionBehaviorTriplet;
    struct _third {
        CppBFSTestVisitor::ColorPropertyMap operator()(const CppBFSTestVisitor::OuterLoopState& p);
    };

    static CppBFSTestVisitor::_third third;
private:
    static read_write_property_map<CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::Color, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_tail> __read_write_property_map;
public:
    struct _black {
        CppBFSTestVisitor::Color operator()();
    };

    static CppBFSTestVisitor::_black black;
    struct _get {
        CppBFSTestVisitor::Color operator()(const CppBFSTestVisitor::ColorPropertyMap& pm, const CppBFSTestVisitor::Vertex& k);
    };

    static CppBFSTestVisitor::_get get;
    struct _gray {
        CppBFSTestVisitor::Color operator()();
    };

    static CppBFSTestVisitor::_gray gray;
    struct _initMap {
        CppBFSTestVisitor::ColorPropertyMap operator()(const CppBFSTestVisitor::VertexList& kl, const CppBFSTestVisitor::Color& v);
    };

    static CppBFSTestVisitor::_initMap initMap;
    struct _put {
        CppBFSTestVisitor::ColorPropertyMap operator()(const CppBFSTestVisitor::ColorPropertyMap& pm, const CppBFSTestVisitor::Vertex& k, const CppBFSTestVisitor::Color& v);
    };

    static CppBFSTestVisitor::_put put;
    struct _white {
        CppBFSTestVisitor::Color operator()();
    };

    static CppBFSTestVisitor::_white white;
};
} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp

namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
struct CppDijkstraVisitor {
private:
    static color_marker __color_marker;
    static base_types __base_types;
    static base_float_ops __base_float_ops;
public:
    typedef base_types::Vertex Vertex;
    typedef list<CppDijkstraVisitor::Vertex>::List VertexList;
    struct _emptyVertexList {
        CppDijkstraVisitor::VertexList operator()();
    };

    static CppDijkstraVisitor::_emptyVertexList emptyVertexList;
    typedef pair<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::Vertex>::Pair VertexPair;
    typedef list<CppDijkstraVisitor::VertexPair>::List VertexPairList;
    struct _emptyVertexPairList {
        CppDijkstraVisitor::VertexPairList operator()();
    };

    static CppDijkstraVisitor::_emptyVertexPairList emptyVertexPairList;
private:
    static list<CppDijkstraVisitor::VertexPair> __list1;
    static edge<CppDijkstraVisitor::Vertex> __edge;
    static list<CppDijkstraVisitor::Vertex> __list0;
    static pair<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::Vertex> __pair1;
public:
    struct _makeVertexPair {
        CppDijkstraVisitor::VertexPair operator()(const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::Vertex& b);
    };

    static CppDijkstraVisitor::_makeVertexPair makeVertexPair;
    typedef base_types::Int Int;
    typedef edge<CppDijkstraVisitor::Vertex>::Edge Edge;
    typedef list<CppDijkstraVisitor::Edge>::List EdgeList;
    struct _emptyEdgeList {
        CppDijkstraVisitor::EdgeList operator()();
    };

    static CppDijkstraVisitor::_emptyEdgeList emptyEdgeList;
    struct _isEmpty {
        bool operator()(const CppDijkstraVisitor::EdgeList& l);
        bool operator()(const CppDijkstraVisitor::VertexList& l);
        bool operator()(const CppDijkstraVisitor::VertexPairList& l);
    };

    static CppDijkstraVisitor::_isEmpty isEmpty;
    struct _tail {
        CppDijkstraVisitor::EdgeList operator()(const CppDijkstraVisitor::EdgeList& l);
        CppDijkstraVisitor::VertexList operator()(const CppDijkstraVisitor::VertexList& l);
        CppDijkstraVisitor::VertexPairList operator()(const CppDijkstraVisitor::VertexPairList& l);
    };

    static CppDijkstraVisitor::_tail tail;
private:
    static list<CppDijkstraVisitor::Edge> __list;
public:
    struct _cons {
        CppDijkstraVisitor::EdgeList operator()(const CppDijkstraVisitor::Edge& a, const CppDijkstraVisitor::EdgeList& l);
        CppDijkstraVisitor::VertexList operator()(const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::VertexList& l);
        CppDijkstraVisitor::VertexPairList operator()(const CppDijkstraVisitor::VertexPair& a, const CppDijkstraVisitor::VertexPairList& l);
    };

    static CppDijkstraVisitor::_cons cons;
    struct _head {
        CppDijkstraVisitor::Edge operator()(const CppDijkstraVisitor::EdgeList& l);
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::VertexList& l);
        CppDijkstraVisitor::VertexPair operator()(const CppDijkstraVisitor::VertexPairList& l);
    };

    typedef read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail>::PropertyMap VertexPredecessorMap;
    typedef pair<CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::VertexList>::Pair PopulateVPMapState;
    struct _populateVPMapLoopCond {
        bool operator()(const CppDijkstraVisitor::PopulateVPMapState& state, const CppDijkstraVisitor::Vertex& s);
    };

    static CppDijkstraVisitor::_populateVPMapLoopCond populateVPMapLoopCond;
    struct _populateVPMapLoopRepeat {
        void operator()(CppDijkstraVisitor::PopulateVPMapState& s, const CppDijkstraVisitor::Vertex& c);
    };

    static CppDijkstraVisitor::_populateVPMapLoopRepeat populateVPMapLoopRepeat;
    struct _populateVPMapLoopStep {
        void operator()(CppDijkstraVisitor::PopulateVPMapState& state, const CppDijkstraVisitor::Vertex& s);
    };

private:
    static while_loop<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::PopulateVPMapState, CppDijkstraVisitor::_populateVPMapLoopCond, CppDijkstraVisitor::_populateVPMapLoopStep> __while_loop1;
public:
    static CppDijkstraVisitor::_populateVPMapLoopStep populateVPMapLoopStep;
private:
    static pair<CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::VertexList> __pair2;
public:
    struct _emptyVPMap {
        CppDijkstraVisitor::VertexPredecessorMap operator()();
    };

    static CppDijkstraVisitor::_emptyVPMap emptyVPMap;
    struct _makePair {
        CppDijkstraVisitor::PopulateVPMapState operator()(const CppDijkstraVisitor::VertexPredecessorMap& a, const CppDijkstraVisitor::VertexList& b);
    };

    static CppDijkstraVisitor::_makePair makePair;
private:
    static read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> __read_write_property_map2;
public:
    static CppDijkstraVisitor::_head head;
    struct _makeEdge {
        CppDijkstraVisitor::Edge operator()(const CppDijkstraVisitor::Vertex& s, const CppDijkstraVisitor::Vertex& t);
    };

    static CppDijkstraVisitor::_makeEdge makeEdge;
    struct _src {
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::Edge& e);
    };

    static CppDijkstraVisitor::_src src;
    struct _tgt {
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::Edge& e);
    };

    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_makeEdge, CppDijkstraVisitor::_src, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tgt>::Graph Graph;
    typedef pair<CppDijkstraVisitor::Graph, CppDijkstraVisitor::Vertex>::Pair InnerLoopContext;
private:
    static pair<CppDijkstraVisitor::Graph, CppDijkstraVisitor::Vertex> __pair;
public:
    struct _makeInnerLoopContext {
        CppDijkstraVisitor::InnerLoopContext operator()(const CppDijkstraVisitor::Graph& a, const CppDijkstraVisitor::Vertex& b);
    };

    static CppDijkstraVisitor::_makeInnerLoopContext makeInnerLoopContext;
    struct _outEdges {
        CppDijkstraVisitor::EdgeList operator()(const CppDijkstraVisitor::Vertex& v, const CppDijkstraVisitor::Graph& g);
    };

    static CppDijkstraVisitor::_outEdges outEdges;
    struct _vertices {
        CppDijkstraVisitor::VertexList operator()(const CppDijkstraVisitor::Graph& g);
    };

    static CppDijkstraVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_makeEdge, CppDijkstraVisitor::_src, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tgt>::VertexCount VertexCount;
    struct _numVertices {
        CppDijkstraVisitor::VertexCount operator()(const CppDijkstraVisitor::Graph& g);
    };

    static CppDijkstraVisitor::_numVertices numVertices;
    struct _outDegree {
        CppDijkstraVisitor::VertexCount operator()(const CppDijkstraVisitor::Vertex& v, const CppDijkstraVisitor::Graph& g);
    };

    static CppDijkstraVisitor::_outDegree outDegree;
private:
    static incidence_and_vertex_list_graph<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_makeEdge, CppDijkstraVisitor::_src, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tgt> __incidence_and_vertex_list_graph;
public:
    static CppDijkstraVisitor::_tgt tgt;
    typedef base_float_ops::Float Cost;
    typedef read_write_property_map<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail>::PropertyMap EdgeCostMap;
    struct _emptyECMap {
        CppDijkstraVisitor::EdgeCostMap operator()();
    };

    static CppDijkstraVisitor::_emptyECMap emptyECMap;
    typedef read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail>::PropertyMap VertexCostMap;
    typedef triplet<CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::EdgeCostMap>::Triplet StateWithMaps;
    struct _getEdgeCostMap {
        CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::StateWithMaps& p);
    };

    static CppDijkstraVisitor::_getEdgeCostMap getEdgeCostMap;
    struct _getVertexPredecessorMap {
        CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::StateWithMaps& p);
    };

    static CppDijkstraVisitor::_getVertexPredecessorMap getVertexPredecessorMap;
    struct _putVertexPredecessorMap {
        CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::VertexPredecessorMap& vpm, const CppDijkstraVisitor::StateWithMaps& swm);
    };

    static CppDijkstraVisitor::_putVertexPredecessorMap putVertexPredecessorMap;
private:
    static triplet<CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::EdgeCostMap> __triplet0;
public:
    struct _emptyVCMap {
        CppDijkstraVisitor::VertexCostMap operator()();
    };

    static CppDijkstraVisitor::_emptyVCMap emptyVCMap;
    struct _getVertexCostMap {
        CppDijkstraVisitor::VertexCostMap operator()(const CppDijkstraVisitor::StateWithMaps& p);
    };

    static CppDijkstraVisitor::_getVertexCostMap getVertexCostMap;
    struct _makeStateWithMaps {
        CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::VertexCostMap& a, const CppDijkstraVisitor::VertexPredecessorMap& b, const CppDijkstraVisitor::EdgeCostMap& c);
    };

    static CppDijkstraVisitor::_makeStateWithMaps makeStateWithMaps;
    struct _putVertexCostMap {
        CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::StateWithMaps& swm);
    };

    static CppDijkstraVisitor::_putVertexCostMap putVertexCostMap;
    struct _relax {
        void operator()(const CppDijkstraVisitor::Edge& e, const CppDijkstraVisitor::EdgeCostMap& ecm, CppDijkstraVisitor::VertexCostMap& vcm, CppDijkstraVisitor::VertexPredecessorMap& vpm);
    };

    static CppDijkstraVisitor::_relax relax;
private:
    static read_write_property_map<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> __read_write_property_map;
    static read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> __read_write_property_map1;
public:
    struct _dijkstraShortestPaths {
        void operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::Vertex& start, CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::EdgeCostMap& ecm, const CppDijkstraVisitor::Cost& initialCost, CppDijkstraVisitor::VertexPredecessorMap& vpm);
    };

    static CppDijkstraVisitor::_dijkstraShortestPaths dijkstraShortestPaths;
    struct _less {
        bool operator()(const CppDijkstraVisitor::Cost& i1, const CppDijkstraVisitor::Cost& i2);
    };

    static CppDijkstraVisitor::_less less;
    struct _plus {
        CppDijkstraVisitor::Cost operator()(const CppDijkstraVisitor::Cost& i1, const CppDijkstraVisitor::Cost& i2);
    };

    static CppDijkstraVisitor::_plus plus;
    typedef color_marker::Color Color;
    typedef read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Color, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail>::PropertyMap ColorPropertyMap;
    struct _emptyMap {
        CppDijkstraVisitor::ColorPropertyMap operator()();
    };

    static CppDijkstraVisitor::_emptyMap emptyMap;
private:
    static read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Color, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> __read_write_property_map0;
public:
    struct _black {
        CppDijkstraVisitor::Color operator()();
    };

    static CppDijkstraVisitor::_black black;
    struct _get {
        CppDijkstraVisitor::Cost operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::Vertex& k);
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::VertexPredecessorMap& pm, const CppDijkstraVisitor::Vertex& k);
        CppDijkstraVisitor::Cost operator()(const CppDijkstraVisitor::EdgeCostMap& pm, const CppDijkstraVisitor::Edge& k);
        CppDijkstraVisitor::Color operator()(const CppDijkstraVisitor::ColorPropertyMap& pm, const CppDijkstraVisitor::Vertex& k);
    };

    typedef priority_queue<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::Cost, CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::_get>::PriorityQueue PriorityQueue;
    typedef triplet<CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap>::Triplet OuterLoopState;
    typedef pair<CppDijkstraVisitor::OuterLoopState, CppDijkstraVisitor::EdgeList>::Pair InnerLoopState;
    struct _bfsInnerLoopCond {
        bool operator()(const CppDijkstraVisitor::InnerLoopState& state, const CppDijkstraVisitor::InnerLoopContext& ctx);
    };

    static CppDijkstraVisitor::_bfsInnerLoopCond bfsInnerLoopCond;
    struct _bfsInnerLoopRepeat {
        void operator()(CppDijkstraVisitor::InnerLoopState& s, const CppDijkstraVisitor::InnerLoopContext& c);
    };

    static CppDijkstraVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        void operator()(CppDijkstraVisitor::InnerLoopState& state, const CppDijkstraVisitor::InnerLoopContext& ctx);
    };

private:
    static while_loop<CppDijkstraVisitor::InnerLoopContext, CppDijkstraVisitor::InnerLoopState, CppDijkstraVisitor::_bfsInnerLoopCond, CppDijkstraVisitor::_bfsInnerLoopStep> __while_loop0;
public:
    static CppDijkstraVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
private:
    static pair<CppDijkstraVisitor::OuterLoopState, CppDijkstraVisitor::EdgeList> __pair0;
public:
    struct _bfsOuterLoopCond {
        bool operator()(const CppDijkstraVisitor::OuterLoopState& state, const CppDijkstraVisitor::Graph& g);
    };

    static CppDijkstraVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        void operator()(CppDijkstraVisitor::OuterLoopState& s, const CppDijkstraVisitor::Graph& c);
    };

    static CppDijkstraVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        void operator()(CppDijkstraVisitor::OuterLoopState& state, const CppDijkstraVisitor::Graph& g);
    };

private:
    static while_loop<CppDijkstraVisitor::Graph, CppDijkstraVisitor::OuterLoopState, CppDijkstraVisitor::_bfsOuterLoopCond, CppDijkstraVisitor::_bfsOuterLoopStep> __while_loop;
public:
    static CppDijkstraVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _first {
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::VertexPair& p);
        CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::PopulateVPMapState& p);
        CppDijkstraVisitor::Graph operator()(const CppDijkstraVisitor::InnerLoopContext& p);
        CppDijkstraVisitor::OuterLoopState operator()(const CppDijkstraVisitor::InnerLoopState& p);
        CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::OuterLoopState& p);
    };

    static CppDijkstraVisitor::_first first;
    struct _makeInnerLoopState {
        CppDijkstraVisitor::InnerLoopState operator()(const CppDijkstraVisitor::OuterLoopState& a, const CppDijkstraVisitor::EdgeList& b);
    };

    static CppDijkstraVisitor::_makeInnerLoopState makeInnerLoopState;
    struct _projectionBehaviorPair {
        void operator()(const CppDijkstraVisitor::VertexPredecessorMap& a, const CppDijkstraVisitor::VertexList& b);
        void operator()(const CppDijkstraVisitor::Graph& a, const CppDijkstraVisitor::Vertex& b);
        void operator()(const CppDijkstraVisitor::OuterLoopState& a, const CppDijkstraVisitor::EdgeList& b);
    };

    static CppDijkstraVisitor::_projectionBehaviorPair projectionBehaviorPair;
    struct _third {
        CppDijkstraVisitor::ColorPropertyMap operator()(const CppDijkstraVisitor::OuterLoopState& p);
    };

    static CppDijkstraVisitor::_third third;
    struct _whileLoopBehavior {
        void operator()(const CppDijkstraVisitor::PopulateVPMapState& s, const CppDijkstraVisitor::Vertex& c);
        void operator()(const CppDijkstraVisitor::OuterLoopState& s, const CppDijkstraVisitor::Graph& c);
        void operator()(const CppDijkstraVisitor::InnerLoopState& s, const CppDijkstraVisitor::InnerLoopContext& c);
    };

    static CppDijkstraVisitor::_whileLoopBehavior whileLoopBehavior;
private:
    static triplet<CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap> __triplet;
public:
    struct _blackTarget {
        void operator()(const CppDijkstraVisitor::Edge& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a);
    };

    static CppDijkstraVisitor::_blackTarget blackTarget;
    struct _breadthFirstVisit {
        void operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::Vertex& s, CppDijkstraVisitor::StateWithMaps& a, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c);
    };

    static CppDijkstraVisitor::_breadthFirstVisit breadthFirstVisit;
    struct _discoverVertex {
        void operator()(const CppDijkstraVisitor::Vertex& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a);
    };

    static CppDijkstraVisitor::_discoverVertex discoverVertex;
    struct _emptyPriorityQueue {
        CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::VertexCostMap& pm);
    };

    static CppDijkstraVisitor::_emptyPriorityQueue emptyPriorityQueue;
    struct _examineEdge {
        void operator()(const CppDijkstraVisitor::Edge& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a);
    };

    static CppDijkstraVisitor::_examineEdge examineEdge;
    struct _examineVertex {
        void operator()(const CppDijkstraVisitor::Vertex& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a);
    };

    static CppDijkstraVisitor::_examineVertex examineVertex;
    struct _finishVertex {
        void operator()(const CppDijkstraVisitor::Vertex& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a);
    };

    static CppDijkstraVisitor::_finishVertex finishVertex;
    struct _front {
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::PriorityQueue& q);
    };

    static CppDijkstraVisitor::_front front;
    struct _grayTarget {
        void operator()(const CppDijkstraVisitor::Edge& e, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& pq, CppDijkstraVisitor::StateWithMaps& swm);
    };

    static CppDijkstraVisitor::_grayTarget grayTarget;
    struct _isEmptyQueue {
        bool operator()(const CppDijkstraVisitor::PriorityQueue& q);
    };

    static CppDijkstraVisitor::_isEmptyQueue isEmptyQueue;
    struct _makeOuterLoopState {
        CppDijkstraVisitor::OuterLoopState operator()(const CppDijkstraVisitor::StateWithMaps& a, const CppDijkstraVisitor::PriorityQueue& b, const CppDijkstraVisitor::ColorPropertyMap& c);
    };

    static CppDijkstraVisitor::_makeOuterLoopState makeOuterLoopState;
    struct _nonTreeEdge {
        void operator()(const CppDijkstraVisitor::Edge& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a);
    };

    static CppDijkstraVisitor::_nonTreeEdge nonTreeEdge;
    struct _pop {
        CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::PriorityQueue& q);
    };

    static CppDijkstraVisitor::_pop pop;
    struct _projectionBehaviorTriplet {
        void operator()(const CppDijkstraVisitor::StateWithMaps& a, const CppDijkstraVisitor::PriorityQueue& b, const CppDijkstraVisitor::ColorPropertyMap& c);
    };

    static CppDijkstraVisitor::_projectionBehaviorTriplet projectionBehaviorTriplet;
    struct _push {
        CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::PriorityQueue& q);
    };

    static CppDijkstraVisitor::_push push;
    struct _second {
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::VertexPair& p);
        CppDijkstraVisitor::VertexList operator()(const CppDijkstraVisitor::PopulateVPMapState& p);
        CppDijkstraVisitor::Vertex operator()(const CppDijkstraVisitor::InnerLoopContext& p);
        CppDijkstraVisitor::EdgeList operator()(const CppDijkstraVisitor::InnerLoopState& p);
        CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::OuterLoopState& p);
    };

    static CppDijkstraVisitor::_second second;
    struct _treeEdge {
        void operator()(const CppDijkstraVisitor::Edge& e, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& pq, CppDijkstraVisitor::StateWithMaps& swm);
    };

    static CppDijkstraVisitor::_treeEdge treeEdge;
    struct _update {
        CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::PriorityQueue& pq);
    };

    static CppDijkstraVisitor::_update update;
private:
    static priority_queue<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::Cost, CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::_get> __priority_queue;
public:
    static CppDijkstraVisitor::_get get;
    struct _gray {
        CppDijkstraVisitor::Color operator()();
    };

    static CppDijkstraVisitor::_gray gray;
    struct _initMap {
        CppDijkstraVisitor::VertexCostMap operator()(const CppDijkstraVisitor::VertexList& kl, const CppDijkstraVisitor::Cost& v);
        CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::VertexList& kl, const CppDijkstraVisitor::Vertex& v);
        CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::EdgeList& kl, const CppDijkstraVisitor::Cost& v);
        CppDijkstraVisitor::ColorPropertyMap operator()(const CppDijkstraVisitor::VertexList& kl, const CppDijkstraVisitor::Color& v);
    };

    static CppDijkstraVisitor::_initMap initMap;
    struct _put {
        CppDijkstraVisitor::VertexCostMap operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::Vertex& k, const CppDijkstraVisitor::Cost& v);
        CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::VertexPredecessorMap& pm, const CppDijkstraVisitor::Vertex& k, const CppDijkstraVisitor::Vertex& v);
        CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::EdgeCostMap& pm, const CppDijkstraVisitor::Edge& k, const CppDijkstraVisitor::Cost& v);
        CppDijkstraVisitor::ColorPropertyMap operator()(const CppDijkstraVisitor::ColorPropertyMap& pm, const CppDijkstraVisitor::Vertex& k, const CppDijkstraVisitor::Color& v);
    };

    static CppDijkstraVisitor::_put put;
    struct _white {
        CppDijkstraVisitor::Color operator()();
    };

    static CppDijkstraVisitor::_white white;
};
} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp