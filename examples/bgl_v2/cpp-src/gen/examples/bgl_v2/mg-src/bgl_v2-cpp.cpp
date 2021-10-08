#include "gen/examples/bgl_v2/mg-src/bgl_v2-cpp.hpp"


namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
    color_marker CppBFSTestVisitor::__color_marker;

    base_types CppBFSTestVisitor::__base_types;

    CppBFSTestVisitor::VertexList CppBFSTestVisitor::_emptyVertexList::operator()() {
        return __list0.empty();
    };


    CppBFSTestVisitor::_emptyVertexList CppBFSTestVisitor::emptyVertexList;

    edge<CppBFSTestVisitor::Vertex> CppBFSTestVisitor::__edge;

    fifo_queue<CppBFSTestVisitor::Vertex> CppBFSTestVisitor::__fifo_queue;

    list<CppBFSTestVisitor::Vertex> CppBFSTestVisitor::__list0;

    CppBFSTestVisitor::FIFOQueue CppBFSTestVisitor::_empty::operator()() {
        return __fifo_queue.empty();
    };


    CppBFSTestVisitor::_empty CppBFSTestVisitor::empty;

    CppBFSTestVisitor::Vertex CppBFSTestVisitor::_front::operator()(const CppBFSTestVisitor::FIFOQueue& q) {
        return __fifo_queue.front(q);
    };


    CppBFSTestVisitor::_front CppBFSTestVisitor::front;

    bool CppBFSTestVisitor::_isEmptyQueue::operator()(const CppBFSTestVisitor::FIFOQueue& q) {
        return __fifo_queue.isEmpty(q);
    };


    CppBFSTestVisitor::_isEmptyQueue CppBFSTestVisitor::isEmptyQueue;

    CppBFSTestVisitor::FIFOQueue CppBFSTestVisitor::_pop::operator()(const CppBFSTestVisitor::FIFOQueue& q) {
        return __fifo_queue.pop(q);
    };


    CppBFSTestVisitor::_pop CppBFSTestVisitor::pop;

    CppBFSTestVisitor::FIFOQueue CppBFSTestVisitor::_push::operator()(const CppBFSTestVisitor::Vertex& a, const CppBFSTestVisitor::FIFOQueue& q) {
        return __fifo_queue.push(a, q);
    };


    CppBFSTestVisitor::_push CppBFSTestVisitor::push;

    CppBFSTestVisitor::EdgeList CppBFSTestVisitor::_emptyEdgeList::operator()() {
        return __list.empty();
    };


    CppBFSTestVisitor::_emptyEdgeList CppBFSTestVisitor::emptyEdgeList;

    bool CppBFSTestVisitor::_isEmpty::operator()(const CppBFSTestVisitor::EdgeList& l) {
        return __list.isEmpty(l);
    };

    bool CppBFSTestVisitor::_isEmpty::operator()(const CppBFSTestVisitor::VertexList& l) {
        return __list0.isEmpty(l);
    };


    CppBFSTestVisitor::_isEmpty CppBFSTestVisitor::isEmpty;

    CppBFSTestVisitor::EdgeList CppBFSTestVisitor::_tail::operator()(const CppBFSTestVisitor::EdgeList& l) {
        return __list.tail(l);
    };

    CppBFSTestVisitor::VertexList CppBFSTestVisitor::_tail::operator()(const CppBFSTestVisitor::VertexList& l) {
        return __list0.tail(l);
    };


    CppBFSTestVisitor::_tail CppBFSTestVisitor::tail;

    list<CppBFSTestVisitor::Edge> CppBFSTestVisitor::__list;

    CppBFSTestVisitor::EdgeList CppBFSTestVisitor::_cons::operator()(const CppBFSTestVisitor::Edge& a, const CppBFSTestVisitor::EdgeList& l) {
        return __list.cons(a, l);
    };

    CppBFSTestVisitor::VertexList CppBFSTestVisitor::_cons::operator()(const CppBFSTestVisitor::Vertex& a, const CppBFSTestVisitor::VertexList& l) {
        return __list0.cons(a, l);
    };


    CppBFSTestVisitor::_cons CppBFSTestVisitor::cons;

    CppBFSTestVisitor::Edge CppBFSTestVisitor::_head::operator()(const CppBFSTestVisitor::EdgeList& l) {
        return __list.head(l);
    };

    CppBFSTestVisitor::Vertex CppBFSTestVisitor::_head::operator()(const CppBFSTestVisitor::VertexList& l) {
        return __list0.head(l);
    };


    CppBFSTestVisitor::_head CppBFSTestVisitor::head;

    CppBFSTestVisitor::Edge CppBFSTestVisitor::_makeEdge::operator()(const CppBFSTestVisitor::Vertex& s, const CppBFSTestVisitor::Vertex& t) {
        return __edge.makeEdge(s, t);
    };


    CppBFSTestVisitor::_makeEdge CppBFSTestVisitor::makeEdge;

    CppBFSTestVisitor::Vertex CppBFSTestVisitor::_src::operator()(const CppBFSTestVisitor::Edge& e) {
        return __edge.src(e);
    };


    CppBFSTestVisitor::_src CppBFSTestVisitor::src;

    CppBFSTestVisitor::Vertex CppBFSTestVisitor::_tgt::operator()(const CppBFSTestVisitor::Edge& e) {
        return __edge.tgt(e);
    };


    pair<CppBFSTestVisitor::Graph, CppBFSTestVisitor::Vertex> CppBFSTestVisitor::__pair;

    CppBFSTestVisitor::VertexList CppBFSTestVisitor::_breadthFirstSearch::operator()(const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::Vertex& start, const CppBFSTestVisitor::VertexList& init) {
        CppBFSTestVisitor::FIFOQueue q = CppBFSTestVisitor::empty();
        CppBFSTestVisitor::ColorPropertyMap c = CppBFSTestVisitor::initMap(CppBFSTestVisitor::vertices(g), CppBFSTestVisitor::white());
        CppBFSTestVisitor::VertexList a = init;
        CppBFSTestVisitor::breadthFirstVisit(g, start, a, q, c);
        return a;
    };


    CppBFSTestVisitor::_breadthFirstSearch CppBFSTestVisitor::breadthFirstSearch;

    void CppBFSTestVisitor::_defaultAction::operator()(const CppBFSTestVisitor::Vertex& edgeOrVertex, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexList& a) {
        ;
    };

    void CppBFSTestVisitor::_defaultAction::operator()(const CppBFSTestVisitor::Edge& edgeOrVertex, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexList& a) {
        ;
    };


    CppBFSTestVisitor::_defaultAction CppBFSTestVisitor::defaultAction;

    void CppBFSTestVisitor::_discoverVertex::operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexList& a) {
        a = CppBFSTestVisitor::cons(v, a);
    };


    CppBFSTestVisitor::_discoverVertex CppBFSTestVisitor::discoverVertex;

    CppBFSTestVisitor::InnerLoopContext CppBFSTestVisitor::_makeInnerLoopContext::operator()(const CppBFSTestVisitor::Graph& a, const CppBFSTestVisitor::Vertex& b) {
        return __pair.makePair(a, b);
    };


    CppBFSTestVisitor::_makeInnerLoopContext CppBFSTestVisitor::makeInnerLoopContext;

    CppBFSTestVisitor::EdgeList CppBFSTestVisitor::_outEdges::operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.outEdges(v, g);
    };


    CppBFSTestVisitor::_outEdges CppBFSTestVisitor::outEdges;

    CppBFSTestVisitor::VertexList CppBFSTestVisitor::_vertices::operator()(const CppBFSTestVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.vertices(g);
    };


    CppBFSTestVisitor::_vertices CppBFSTestVisitor::vertices;

    CppBFSTestVisitor::VertexCount CppBFSTestVisitor::_numVertices::operator()(const CppBFSTestVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.numVertices(g);
    };


    CppBFSTestVisitor::_numVertices CppBFSTestVisitor::numVertices;

    CppBFSTestVisitor::VertexCount CppBFSTestVisitor::_outDegree::operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.outDegree(v, g);
    };


    CppBFSTestVisitor::_outDegree CppBFSTestVisitor::outDegree;

    incidence_and_vertex_list_graph<CppBFSTestVisitor::Edge, CppBFSTestVisitor::EdgeList, CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyEdgeList, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_makeEdge, CppBFSTestVisitor::_src, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tail, CppBFSTestVisitor::_tgt> CppBFSTestVisitor::__incidence_and_vertex_list_graph;

    CppBFSTestVisitor::_tgt CppBFSTestVisitor::tgt;

    bool CppBFSTestVisitor::_bfsInnerLoopCond::operator()(const CppBFSTestVisitor::InnerLoopState& state, const CppBFSTestVisitor::InnerLoopContext& ctx) {
        CppBFSTestVisitor::EdgeList edgeList = CppBFSTestVisitor::second(state);
        return !CppBFSTestVisitor::isEmpty(edgeList);
    };


    CppBFSTestVisitor::_bfsInnerLoopCond CppBFSTestVisitor::bfsInnerLoopCond;

    void CppBFSTestVisitor::_bfsInnerLoopRepeat::operator()(CppBFSTestVisitor::InnerLoopState& s, const CppBFSTestVisitor::InnerLoopContext& c) {
        return __while_loop0.repeat(s, c);
    };


    CppBFSTestVisitor::_bfsInnerLoopRepeat CppBFSTestVisitor::bfsInnerLoopRepeat;

    void CppBFSTestVisitor::_bfsInnerLoopStep::operator()(CppBFSTestVisitor::InnerLoopState& state, const CppBFSTestVisitor::InnerLoopContext& ctx) {
        CppBFSTestVisitor::Graph g = CppBFSTestVisitor::first(ctx);
        CppBFSTestVisitor::Vertex u = CppBFSTestVisitor::second(ctx);
        CppBFSTestVisitor::OuterLoopState outerState = CppBFSTestVisitor::first(state);
        CppBFSTestVisitor::VertexList x1 = CppBFSTestVisitor::first(outerState);
        CppBFSTestVisitor::FIFOQueue q1 = CppBFSTestVisitor::second(outerState);
        CppBFSTestVisitor::ColorPropertyMap c1 = CppBFSTestVisitor::third(outerState);
        CppBFSTestVisitor::EdgeList edgeList = CppBFSTestVisitor::second(state);
        CppBFSTestVisitor::Edge e = CppBFSTestVisitor::head(edgeList);
        CppBFSTestVisitor::EdgeList es = CppBFSTestVisitor::tail(edgeList);
        CppBFSTestVisitor::Vertex v = CppBFSTestVisitor::tgt(e);
        CppBFSTestVisitor::defaultAction(e, g, q1, x1);
        CppBFSTestVisitor::Color vc = CppBFSTestVisitor::get(c1, v);
        if ((vc) == (CppBFSTestVisitor::white()))
        {
            CppBFSTestVisitor::defaultAction(e, g, q1, x1);
            CppBFSTestVisitor::ColorPropertyMap c2 = CppBFSTestVisitor::put(c1, v, CppBFSTestVisitor::gray());
            CppBFSTestVisitor::discoverVertex(v, g, q1, x1);
            state = CppBFSTestVisitor::makeInnerLoopState(CppBFSTestVisitor::makeOuterLoopState(x1, CppBFSTestVisitor::push(v, q1), c2), es);
        }
        else
            if ((vc) == (CppBFSTestVisitor::gray()))
            {
                CppBFSTestVisitor::defaultAction(e, g, q1, x1);
                state = CppBFSTestVisitor::makeInnerLoopState(CppBFSTestVisitor::makeOuterLoopState(x1, q1, c1), es);
            }
            else
            {
                CppBFSTestVisitor::defaultAction(e, g, q1, x1);
                CppBFSTestVisitor::ColorPropertyMap c2 = CppBFSTestVisitor::put(c1, u, CppBFSTestVisitor::black());
                state = CppBFSTestVisitor::makeInnerLoopState(CppBFSTestVisitor::makeOuterLoopState(x1, q1, c2), es);
            }
    };


    while_loop<CppBFSTestVisitor::InnerLoopContext, CppBFSTestVisitor::InnerLoopState, CppBFSTestVisitor::_bfsInnerLoopCond, CppBFSTestVisitor::_bfsInnerLoopStep> CppBFSTestVisitor::__while_loop0;

    CppBFSTestVisitor::_bfsInnerLoopStep CppBFSTestVisitor::bfsInnerLoopStep;

    pair<CppBFSTestVisitor::OuterLoopState, CppBFSTestVisitor::EdgeList> CppBFSTestVisitor::__pair0;

    bool CppBFSTestVisitor::_bfsOuterLoopCond::operator()(const CppBFSTestVisitor::OuterLoopState& state, const CppBFSTestVisitor::Graph& g) {
        CppBFSTestVisitor::FIFOQueue q = CppBFSTestVisitor::second(state);
        return !CppBFSTestVisitor::isEmptyQueue(q);
    };


    CppBFSTestVisitor::_bfsOuterLoopCond CppBFSTestVisitor::bfsOuterLoopCond;

    void CppBFSTestVisitor::_bfsOuterLoopRepeat::operator()(CppBFSTestVisitor::OuterLoopState& s, const CppBFSTestVisitor::Graph& c) {
        return __while_loop.repeat(s, c);
    };


    CppBFSTestVisitor::_bfsOuterLoopRepeat CppBFSTestVisitor::bfsOuterLoopRepeat;

    void CppBFSTestVisitor::_bfsOuterLoopStep::operator()(CppBFSTestVisitor::OuterLoopState& state, const CppBFSTestVisitor::Graph& g) {
        CppBFSTestVisitor::VertexList x = CppBFSTestVisitor::first(state);
        CppBFSTestVisitor::FIFOQueue q1 = CppBFSTestVisitor::second(state);
        CppBFSTestVisitor::ColorPropertyMap c = CppBFSTestVisitor::third(state);
        CppBFSTestVisitor::Vertex u = CppBFSTestVisitor::front(q1);
        CppBFSTestVisitor::FIFOQueue q2 = CppBFSTestVisitor::pop(q1);
        CppBFSTestVisitor::defaultAction(u, g, q2, x);
        CppBFSTestVisitor::InnerLoopState innerState = CppBFSTestVisitor::makeInnerLoopState(CppBFSTestVisitor::makeOuterLoopState(x, q2, c), CppBFSTestVisitor::outEdges(u, g));
        CppBFSTestVisitor::InnerLoopContext innerContext = CppBFSTestVisitor::makeInnerLoopContext(g, u);
        CppBFSTestVisitor::bfsInnerLoopRepeat(innerState, innerContext);
        CppBFSTestVisitor::OuterLoopState outerLoopStateAfterInnerLoop = CppBFSTestVisitor::first(innerState);
        CppBFSTestVisitor::VertexList x_end = CppBFSTestVisitor::first(outerLoopStateAfterInnerLoop);
        CppBFSTestVisitor::FIFOQueue q_end = CppBFSTestVisitor::second(outerLoopStateAfterInnerLoop);
        CppBFSTestVisitor::ColorPropertyMap c_end = CppBFSTestVisitor::third(outerLoopStateAfterInnerLoop);
        CppBFSTestVisitor::defaultAction(u, g, q_end, x_end);
        state = CppBFSTestVisitor::makeOuterLoopState(x_end, q_end, c_end);
    };


    while_loop<CppBFSTestVisitor::Graph, CppBFSTestVisitor::OuterLoopState, CppBFSTestVisitor::_bfsOuterLoopCond, CppBFSTestVisitor::_bfsOuterLoopStep> CppBFSTestVisitor::__while_loop;

    CppBFSTestVisitor::_bfsOuterLoopStep CppBFSTestVisitor::bfsOuterLoopStep;

    CppBFSTestVisitor::Graph CppBFSTestVisitor::_first::operator()(const CppBFSTestVisitor::InnerLoopContext& p) {
        return __pair.first(p);
    };

    CppBFSTestVisitor::OuterLoopState CppBFSTestVisitor::_first::operator()(const CppBFSTestVisitor::InnerLoopState& p) {
        return __pair0.first(p);
    };

    CppBFSTestVisitor::VertexList CppBFSTestVisitor::_first::operator()(const CppBFSTestVisitor::OuterLoopState& p) {
        return __triplet.first(p);
    };


    CppBFSTestVisitor::_first CppBFSTestVisitor::first;

    CppBFSTestVisitor::InnerLoopState CppBFSTestVisitor::_makeInnerLoopState::operator()(const CppBFSTestVisitor::OuterLoopState& a, const CppBFSTestVisitor::EdgeList& b) {
        return __pair0.makePair(a, b);
    };


    CppBFSTestVisitor::_makeInnerLoopState CppBFSTestVisitor::makeInnerLoopState;

    void CppBFSTestVisitor::_projectionBehaviorPair::operator()(const CppBFSTestVisitor::Graph& a, const CppBFSTestVisitor::Vertex& b) {
        CppBFSTestVisitor::InnerLoopContext pair = CppBFSTestVisitor::makeInnerLoopContext(a, b);
        assert((CppBFSTestVisitor::first(pair)) == (a));
        assert((CppBFSTestVisitor::second(pair)) == (b));
    };

    void CppBFSTestVisitor::_projectionBehaviorPair::operator()(const CppBFSTestVisitor::OuterLoopState& a, const CppBFSTestVisitor::EdgeList& b) {
        CppBFSTestVisitor::InnerLoopState pair = CppBFSTestVisitor::makeInnerLoopState(a, b);
        assert((CppBFSTestVisitor::first(pair)) == (a));
        assert((CppBFSTestVisitor::second(pair)) == (b));
    };


    CppBFSTestVisitor::_projectionBehaviorPair CppBFSTestVisitor::projectionBehaviorPair;

    CppBFSTestVisitor::Vertex CppBFSTestVisitor::_second::operator()(const CppBFSTestVisitor::InnerLoopContext& p) {
        return __pair.second(p);
    };

    CppBFSTestVisitor::EdgeList CppBFSTestVisitor::_second::operator()(const CppBFSTestVisitor::InnerLoopState& p) {
        return __pair0.second(p);
    };

    CppBFSTestVisitor::FIFOQueue CppBFSTestVisitor::_second::operator()(const CppBFSTestVisitor::OuterLoopState& p) {
        return __triplet.second(p);
    };


    CppBFSTestVisitor::_second CppBFSTestVisitor::second;

    void CppBFSTestVisitor::_whileLoopBehavior::operator()(const CppBFSTestVisitor::OuterLoopState& s, const CppBFSTestVisitor::Graph& c) {
        CppBFSTestVisitor::OuterLoopState mutableState = s;
        if (CppBFSTestVisitor::bfsOuterLoopCond(s, c))
        {
            CppBFSTestVisitor::OuterLoopState mutableState1 = s;
            CppBFSTestVisitor::OuterLoopState mutableState2 = s;
            CppBFSTestVisitor::bfsOuterLoopRepeat(mutableState1, c);
            CppBFSTestVisitor::bfsOuterLoopStep(mutableState2, c);
            assert((mutableState1) == (mutableState2));
        }
        else
        {
            CppBFSTestVisitor::OuterLoopState mutableState1 = s;
            CppBFSTestVisitor::bfsOuterLoopRepeat(mutableState1, c);
            assert((mutableState1) == (s));
        }
    };

    void CppBFSTestVisitor::_whileLoopBehavior::operator()(const CppBFSTestVisitor::InnerLoopState& s, const CppBFSTestVisitor::InnerLoopContext& c) {
        CppBFSTestVisitor::InnerLoopState mutableState = s;
        if (CppBFSTestVisitor::bfsInnerLoopCond(s, c))
        {
            CppBFSTestVisitor::InnerLoopState mutableState1 = s;
            CppBFSTestVisitor::InnerLoopState mutableState2 = s;
            CppBFSTestVisitor::bfsInnerLoopRepeat(mutableState1, c);
            CppBFSTestVisitor::bfsInnerLoopStep(mutableState2, c);
            assert((mutableState1) == (mutableState2));
        }
        else
        {
            CppBFSTestVisitor::InnerLoopState mutableState1 = s;
            CppBFSTestVisitor::bfsInnerLoopRepeat(mutableState1, c);
            assert((mutableState1) == (s));
        }
    };


    CppBFSTestVisitor::_whileLoopBehavior CppBFSTestVisitor::whileLoopBehavior;

    triplet<CppBFSTestVisitor::VertexList, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap> CppBFSTestVisitor::__triplet;

    void CppBFSTestVisitor::_breadthFirstVisit::operator()(const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::Vertex& s, CppBFSTestVisitor::VertexList& a, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c) {
        CppBFSTestVisitor::discoverVertex(s, g, q, a);
        CppBFSTestVisitor::FIFOQueue q1 = CppBFSTestVisitor::push(s, q);
        CppBFSTestVisitor::ColorPropertyMap c1 = CppBFSTestVisitor::put(c, s, CppBFSTestVisitor::gray());
        CppBFSTestVisitor::OuterLoopState outerState = CppBFSTestVisitor::makeOuterLoopState(a, q1, c1);
        CppBFSTestVisitor::bfsOuterLoopRepeat(outerState, g);
        a = CppBFSTestVisitor::first(outerState);
    };


    CppBFSTestVisitor::_breadthFirstVisit CppBFSTestVisitor::breadthFirstVisit;

    CppBFSTestVisitor::ColorPropertyMap CppBFSTestVisitor::_emptyMap::operator()() {
        return __read_write_property_map.emptyMap();
    };


    CppBFSTestVisitor::_emptyMap CppBFSTestVisitor::emptyMap;

    CppBFSTestVisitor::OuterLoopState CppBFSTestVisitor::_makeOuterLoopState::operator()(const CppBFSTestVisitor::VertexList& a, const CppBFSTestVisitor::FIFOQueue& b, const CppBFSTestVisitor::ColorPropertyMap& c) {
        return __triplet.makeTriplet(a, b, c);
    };


    CppBFSTestVisitor::_makeOuterLoopState CppBFSTestVisitor::makeOuterLoopState;

    void CppBFSTestVisitor::_projectionBehaviorTriplet::operator()(const CppBFSTestVisitor::VertexList& a, const CppBFSTestVisitor::FIFOQueue& b, const CppBFSTestVisitor::ColorPropertyMap& c) {
        CppBFSTestVisitor::OuterLoopState triplet = CppBFSTestVisitor::makeOuterLoopState(a, b, c);
        assert((CppBFSTestVisitor::first(triplet)) == (a));
        assert((CppBFSTestVisitor::second(triplet)) == (b));
        assert((CppBFSTestVisitor::third(triplet)) == (c));
    };


    CppBFSTestVisitor::_projectionBehaviorTriplet CppBFSTestVisitor::projectionBehaviorTriplet;

    CppBFSTestVisitor::ColorPropertyMap CppBFSTestVisitor::_third::operator()(const CppBFSTestVisitor::OuterLoopState& p) {
        return __triplet.third(p);
    };


    CppBFSTestVisitor::_third CppBFSTestVisitor::third;

    read_write_property_map<CppBFSTestVisitor::Vertex, CppBFSTestVisitor::VertexList, CppBFSTestVisitor::Color, CppBFSTestVisitor::_cons, CppBFSTestVisitor::_emptyVertexList, CppBFSTestVisitor::_head, CppBFSTestVisitor::_isEmpty, CppBFSTestVisitor::_tail> CppBFSTestVisitor::__read_write_property_map;

    CppBFSTestVisitor::Color CppBFSTestVisitor::_black::operator()() {
        return __color_marker.black();
    };


    CppBFSTestVisitor::_black CppBFSTestVisitor::black;

    CppBFSTestVisitor::Color CppBFSTestVisitor::_get::operator()(const CppBFSTestVisitor::ColorPropertyMap& pm, const CppBFSTestVisitor::Vertex& k) {
        return __read_write_property_map.get(pm, k);
    };


    CppBFSTestVisitor::_get CppBFSTestVisitor::get;

    CppBFSTestVisitor::Color CppBFSTestVisitor::_gray::operator()() {
        return __color_marker.gray();
    };


    CppBFSTestVisitor::_gray CppBFSTestVisitor::gray;

    CppBFSTestVisitor::ColorPropertyMap CppBFSTestVisitor::_initMap::operator()(const CppBFSTestVisitor::VertexList& kl, const CppBFSTestVisitor::Color& v) {
        return __read_write_property_map.initMap(kl, v);
    };


    CppBFSTestVisitor::_initMap CppBFSTestVisitor::initMap;

    CppBFSTestVisitor::ColorPropertyMap CppBFSTestVisitor::_put::operator()(const CppBFSTestVisitor::ColorPropertyMap& pm, const CppBFSTestVisitor::Vertex& k, const CppBFSTestVisitor::Color& v) {
        return __read_write_property_map.put(pm, k, v);
    };


    CppBFSTestVisitor::_put CppBFSTestVisitor::put;

    CppBFSTestVisitor::Color CppBFSTestVisitor::_white::operator()() {
        return __color_marker.white();
    };


    CppBFSTestVisitor::_white CppBFSTestVisitor::white;

} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp

namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
    color_marker CppDijkstraVisitor::__color_marker;

    base_types CppDijkstraVisitor::__base_types;

    base_float_ops CppDijkstraVisitor::__base_float_ops;

    CppDijkstraVisitor::VertexList CppDijkstraVisitor::_emptyVertexList::operator()() {
        return __list0.empty();
    };


    CppDijkstraVisitor::_emptyVertexList CppDijkstraVisitor::emptyVertexList;

    CppDijkstraVisitor::VertexPairList CppDijkstraVisitor::_emptyVertexPairList::operator()() {
        return __list1.empty();
    };


    CppDijkstraVisitor::_emptyVertexPairList CppDijkstraVisitor::emptyVertexPairList;

    list<CppDijkstraVisitor::VertexPair> CppDijkstraVisitor::__list1;

    edge<CppDijkstraVisitor::Vertex> CppDijkstraVisitor::__edge;

    list<CppDijkstraVisitor::Vertex> CppDijkstraVisitor::__list0;

    pair<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::Vertex> CppDijkstraVisitor::__pair1;

    CppDijkstraVisitor::VertexPair CppDijkstraVisitor::_makeVertexPair::operator()(const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::Vertex& b) {
        return __pair1.makePair(a, b);
    };


    CppDijkstraVisitor::_makeVertexPair CppDijkstraVisitor::makeVertexPair;

    CppDijkstraVisitor::EdgeList CppDijkstraVisitor::_emptyEdgeList::operator()() {
        return __list.empty();
    };


    CppDijkstraVisitor::_emptyEdgeList CppDijkstraVisitor::emptyEdgeList;

    bool CppDijkstraVisitor::_isEmpty::operator()(const CppDijkstraVisitor::EdgeList& l) {
        return __list.isEmpty(l);
    };

    bool CppDijkstraVisitor::_isEmpty::operator()(const CppDijkstraVisitor::VertexList& l) {
        return __list0.isEmpty(l);
    };

    bool CppDijkstraVisitor::_isEmpty::operator()(const CppDijkstraVisitor::VertexPairList& l) {
        return __list1.isEmpty(l);
    };


    CppDijkstraVisitor::_isEmpty CppDijkstraVisitor::isEmpty;

    CppDijkstraVisitor::EdgeList CppDijkstraVisitor::_tail::operator()(const CppDijkstraVisitor::EdgeList& l) {
        return __list.tail(l);
    };

    CppDijkstraVisitor::VertexList CppDijkstraVisitor::_tail::operator()(const CppDijkstraVisitor::VertexList& l) {
        return __list0.tail(l);
    };

    CppDijkstraVisitor::VertexPairList CppDijkstraVisitor::_tail::operator()(const CppDijkstraVisitor::VertexPairList& l) {
        return __list1.tail(l);
    };


    CppDijkstraVisitor::_tail CppDijkstraVisitor::tail;

    list<CppDijkstraVisitor::Edge> CppDijkstraVisitor::__list;

    CppDijkstraVisitor::EdgeList CppDijkstraVisitor::_cons::operator()(const CppDijkstraVisitor::Edge& a, const CppDijkstraVisitor::EdgeList& l) {
        return __list.cons(a, l);
    };

    CppDijkstraVisitor::VertexList CppDijkstraVisitor::_cons::operator()(const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::VertexList& l) {
        return __list0.cons(a, l);
    };

    CppDijkstraVisitor::VertexPairList CppDijkstraVisitor::_cons::operator()(const CppDijkstraVisitor::VertexPair& a, const CppDijkstraVisitor::VertexPairList& l) {
        return __list1.cons(a, l);
    };


    CppDijkstraVisitor::_cons CppDijkstraVisitor::cons;

    CppDijkstraVisitor::Edge CppDijkstraVisitor::_head::operator()(const CppDijkstraVisitor::EdgeList& l) {
        return __list.head(l);
    };

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_head::operator()(const CppDijkstraVisitor::VertexList& l) {
        return __list0.head(l);
    };

    CppDijkstraVisitor::VertexPair CppDijkstraVisitor::_head::operator()(const CppDijkstraVisitor::VertexPairList& l) {
        return __list1.head(l);
    };


    bool CppDijkstraVisitor::_populateVPMapLoopCond::operator()(const CppDijkstraVisitor::PopulateVPMapState& state, const CppDijkstraVisitor::Vertex& s) {
        return !CppDijkstraVisitor::isEmpty(CppDijkstraVisitor::second(state));
    };


    CppDijkstraVisitor::_populateVPMapLoopCond CppDijkstraVisitor::populateVPMapLoopCond;

    void CppDijkstraVisitor::_populateVPMapLoopRepeat::operator()(CppDijkstraVisitor::PopulateVPMapState& s, const CppDijkstraVisitor::Vertex& c) {
        return __while_loop1.repeat(s, c);
    };


    CppDijkstraVisitor::_populateVPMapLoopRepeat CppDijkstraVisitor::populateVPMapLoopRepeat;

    void CppDijkstraVisitor::_populateVPMapLoopStep::operator()(CppDijkstraVisitor::PopulateVPMapState& state, const CppDijkstraVisitor::Vertex& s) {
        CppDijkstraVisitor::VertexPredecessorMap vpm = CppDijkstraVisitor::first(state);
        CppDijkstraVisitor::VertexList vertexList = CppDijkstraVisitor::second(state);
        CppDijkstraVisitor::Vertex v = CppDijkstraVisitor::head(vertexList);
        state = CppDijkstraVisitor::makePair(CppDijkstraVisitor::put(vpm, v, v), CppDijkstraVisitor::tail(vertexList));
    };


    while_loop<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::PopulateVPMapState, CppDijkstraVisitor::_populateVPMapLoopCond, CppDijkstraVisitor::_populateVPMapLoopStep> CppDijkstraVisitor::__while_loop1;

    CppDijkstraVisitor::_populateVPMapLoopStep CppDijkstraVisitor::populateVPMapLoopStep;

    pair<CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::VertexList> CppDijkstraVisitor::__pair2;

    CppDijkstraVisitor::VertexPredecessorMap CppDijkstraVisitor::_emptyVPMap::operator()() {
        return __read_write_property_map2.emptyMap();
    };


    CppDijkstraVisitor::_emptyVPMap CppDijkstraVisitor::emptyVPMap;

    CppDijkstraVisitor::PopulateVPMapState CppDijkstraVisitor::_makePair::operator()(const CppDijkstraVisitor::VertexPredecessorMap& a, const CppDijkstraVisitor::VertexList& b) {
        return __pair2.makePair(a, b);
    };


    CppDijkstraVisitor::_makePair CppDijkstraVisitor::makePair;

    read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> CppDijkstraVisitor::__read_write_property_map2;

    CppDijkstraVisitor::_head CppDijkstraVisitor::head;

    CppDijkstraVisitor::Edge CppDijkstraVisitor::_makeEdge::operator()(const CppDijkstraVisitor::Vertex& s, const CppDijkstraVisitor::Vertex& t) {
        return __edge.makeEdge(s, t);
    };


    CppDijkstraVisitor::_makeEdge CppDijkstraVisitor::makeEdge;

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_src::operator()(const CppDijkstraVisitor::Edge& e) {
        return __edge.src(e);
    };


    CppDijkstraVisitor::_src CppDijkstraVisitor::src;

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_tgt::operator()(const CppDijkstraVisitor::Edge& e) {
        return __edge.tgt(e);
    };


    pair<CppDijkstraVisitor::Graph, CppDijkstraVisitor::Vertex> CppDijkstraVisitor::__pair;

    CppDijkstraVisitor::InnerLoopContext CppDijkstraVisitor::_makeInnerLoopContext::operator()(const CppDijkstraVisitor::Graph& a, const CppDijkstraVisitor::Vertex& b) {
        return __pair.makePair(a, b);
    };


    CppDijkstraVisitor::_makeInnerLoopContext CppDijkstraVisitor::makeInnerLoopContext;

    CppDijkstraVisitor::EdgeList CppDijkstraVisitor::_outEdges::operator()(const CppDijkstraVisitor::Vertex& v, const CppDijkstraVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.outEdges(v, g);
    };


    CppDijkstraVisitor::_outEdges CppDijkstraVisitor::outEdges;

    CppDijkstraVisitor::VertexList CppDijkstraVisitor::_vertices::operator()(const CppDijkstraVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.vertices(g);
    };


    CppDijkstraVisitor::_vertices CppDijkstraVisitor::vertices;

    CppDijkstraVisitor::VertexCount CppDijkstraVisitor::_numVertices::operator()(const CppDijkstraVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.numVertices(g);
    };


    CppDijkstraVisitor::_numVertices CppDijkstraVisitor::numVertices;

    CppDijkstraVisitor::VertexCount CppDijkstraVisitor::_outDegree::operator()(const CppDijkstraVisitor::Vertex& v, const CppDijkstraVisitor::Graph& g) {
        return __incidence_and_vertex_list_graph.outDegree(v, g);
    };


    CppDijkstraVisitor::_outDegree CppDijkstraVisitor::outDegree;

    incidence_and_vertex_list_graph<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_makeEdge, CppDijkstraVisitor::_src, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tail, CppDijkstraVisitor::_tgt> CppDijkstraVisitor::__incidence_and_vertex_list_graph;

    CppDijkstraVisitor::_tgt CppDijkstraVisitor::tgt;

    CppDijkstraVisitor::EdgeCostMap CppDijkstraVisitor::_emptyECMap::operator()() {
        return __read_write_property_map.emptyMap();
    };


    CppDijkstraVisitor::_emptyECMap CppDijkstraVisitor::emptyECMap;

    CppDijkstraVisitor::EdgeCostMap CppDijkstraVisitor::_getEdgeCostMap::operator()(const CppDijkstraVisitor::StateWithMaps& p) {
        return __triplet0.third(p);
    };


    CppDijkstraVisitor::_getEdgeCostMap CppDijkstraVisitor::getEdgeCostMap;

    CppDijkstraVisitor::VertexPredecessorMap CppDijkstraVisitor::_getVertexPredecessorMap::operator()(const CppDijkstraVisitor::StateWithMaps& p) {
        return __triplet0.second(p);
    };


    CppDijkstraVisitor::_getVertexPredecessorMap CppDijkstraVisitor::getVertexPredecessorMap;

    CppDijkstraVisitor::StateWithMaps CppDijkstraVisitor::_putVertexPredecessorMap::operator()(const CppDijkstraVisitor::VertexPredecessorMap& vpm, const CppDijkstraVisitor::StateWithMaps& swm) {
        return CppDijkstraVisitor::makeStateWithMaps(CppDijkstraVisitor::getVertexCostMap(swm), vpm, CppDijkstraVisitor::getEdgeCostMap(swm));
    };


    CppDijkstraVisitor::_putVertexPredecessorMap CppDijkstraVisitor::putVertexPredecessorMap;

    triplet<CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::EdgeCostMap> CppDijkstraVisitor::__triplet0;

    CppDijkstraVisitor::VertexCostMap CppDijkstraVisitor::_emptyVCMap::operator()() {
        return __read_write_property_map1.emptyMap();
    };


    CppDijkstraVisitor::_emptyVCMap CppDijkstraVisitor::emptyVCMap;

    CppDijkstraVisitor::VertexCostMap CppDijkstraVisitor::_getVertexCostMap::operator()(const CppDijkstraVisitor::StateWithMaps& p) {
        return __triplet0.first(p);
    };


    CppDijkstraVisitor::_getVertexCostMap CppDijkstraVisitor::getVertexCostMap;

    CppDijkstraVisitor::StateWithMaps CppDijkstraVisitor::_makeStateWithMaps::operator()(const CppDijkstraVisitor::VertexCostMap& a, const CppDijkstraVisitor::VertexPredecessorMap& b, const CppDijkstraVisitor::EdgeCostMap& c) {
        return __triplet0.makeTriplet(a, b, c);
    };


    CppDijkstraVisitor::_makeStateWithMaps CppDijkstraVisitor::makeStateWithMaps;

    CppDijkstraVisitor::StateWithMaps CppDijkstraVisitor::_putVertexCostMap::operator()(const CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::StateWithMaps& swm) {
        return CppDijkstraVisitor::makeStateWithMaps(vcm, CppDijkstraVisitor::getVertexPredecessorMap(swm), CppDijkstraVisitor::getEdgeCostMap(swm));
    };


    CppDijkstraVisitor::_putVertexCostMap CppDijkstraVisitor::putVertexCostMap;

    void CppDijkstraVisitor::_relax::operator()(const CppDijkstraVisitor::Edge& e, const CppDijkstraVisitor::EdgeCostMap& ecm, CppDijkstraVisitor::VertexCostMap& vcm, CppDijkstraVisitor::VertexPredecessorMap& vpm) {
        CppDijkstraVisitor::Vertex u = CppDijkstraVisitor::src(e);
        CppDijkstraVisitor::Vertex v = CppDijkstraVisitor::tgt(e);
        CppDijkstraVisitor::Cost uCost = CppDijkstraVisitor::get(vcm, u);
        CppDijkstraVisitor::Cost vCost = CppDijkstraVisitor::get(vcm, v);
        CppDijkstraVisitor::Cost edgeCost = CppDijkstraVisitor::get(ecm, e);
        if (CppDijkstraVisitor::less(CppDijkstraVisitor::plus(uCost, edgeCost), vCost))
        {
            vcm = CppDijkstraVisitor::put(vcm, v, CppDijkstraVisitor::plus(uCost, edgeCost));
            vpm = CppDijkstraVisitor::put(vpm, v, u);
        }
        else
            ;
    };


    CppDijkstraVisitor::_relax CppDijkstraVisitor::relax;

    read_write_property_map<CppDijkstraVisitor::Edge, CppDijkstraVisitor::EdgeList, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyEdgeList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> CppDijkstraVisitor::__read_write_property_map;

    read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> CppDijkstraVisitor::__read_write_property_map1;

    void CppDijkstraVisitor::_dijkstraShortestPaths::operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::Vertex& start, CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::EdgeCostMap& ecm, const CppDijkstraVisitor::Cost& initialCost, CppDijkstraVisitor::VertexPredecessorMap& vpm) {
        vcm = CppDijkstraVisitor::put(vcm, start, initialCost);
        CppDijkstraVisitor::PopulateVPMapState populateVPMapState = CppDijkstraVisitor::makePair(CppDijkstraVisitor::emptyVPMap(), CppDijkstraVisitor::vertices(g));
        CppDijkstraVisitor::populateVPMapLoopRepeat(populateVPMapState, start);
        vpm = CppDijkstraVisitor::first(populateVPMapState);
        CppDijkstraVisitor::PriorityQueue pq = CppDijkstraVisitor::emptyPriorityQueue(vcm);
        CppDijkstraVisitor::StateWithMaps swm = CppDijkstraVisitor::makeStateWithMaps(vcm, vpm, ecm);
        CppDijkstraVisitor::ColorPropertyMap c = CppDijkstraVisitor::initMap(CppDijkstraVisitor::vertices(g), CppDijkstraVisitor::white());
        CppDijkstraVisitor::breadthFirstVisit(g, start, swm, pq, c);
        vcm = CppDijkstraVisitor::getVertexCostMap(swm);
        vpm = CppDijkstraVisitor::getVertexPredecessorMap(swm);
    };


    CppDijkstraVisitor::_dijkstraShortestPaths CppDijkstraVisitor::dijkstraShortestPaths;

    bool CppDijkstraVisitor::_less::operator()(const CppDijkstraVisitor::Cost& i1, const CppDijkstraVisitor::Cost& i2) {
        return __base_float_ops.less(i1, i2);
    };


    CppDijkstraVisitor::_less CppDijkstraVisitor::less;

    CppDijkstraVisitor::Cost CppDijkstraVisitor::_plus::operator()(const CppDijkstraVisitor::Cost& i1, const CppDijkstraVisitor::Cost& i2) {
        return __base_float_ops.plus(i1, i2);
    };


    CppDijkstraVisitor::_plus CppDijkstraVisitor::plus;

    CppDijkstraVisitor::ColorPropertyMap CppDijkstraVisitor::_emptyMap::operator()() {
        return __read_write_property_map0.emptyMap();
    };


    CppDijkstraVisitor::_emptyMap CppDijkstraVisitor::emptyMap;

    read_write_property_map<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::VertexList, CppDijkstraVisitor::Color, CppDijkstraVisitor::_cons, CppDijkstraVisitor::_emptyVertexList, CppDijkstraVisitor::_head, CppDijkstraVisitor::_isEmpty, CppDijkstraVisitor::_tail> CppDijkstraVisitor::__read_write_property_map0;

    CppDijkstraVisitor::Color CppDijkstraVisitor::_black::operator()() {
        return __color_marker.black();
    };


    CppDijkstraVisitor::_black CppDijkstraVisitor::black;

    CppDijkstraVisitor::Cost CppDijkstraVisitor::_get::operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::Vertex& k) {
        return __read_write_property_map1.get(pm, k);
    };

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_get::operator()(const CppDijkstraVisitor::VertexPredecessorMap& pm, const CppDijkstraVisitor::Vertex& k) {
        return __read_write_property_map2.get(pm, k);
    };

    CppDijkstraVisitor::Cost CppDijkstraVisitor::_get::operator()(const CppDijkstraVisitor::EdgeCostMap& pm, const CppDijkstraVisitor::Edge& k) {
        return __read_write_property_map.get(pm, k);
    };

    CppDijkstraVisitor::Color CppDijkstraVisitor::_get::operator()(const CppDijkstraVisitor::ColorPropertyMap& pm, const CppDijkstraVisitor::Vertex& k) {
        return __read_write_property_map0.get(pm, k);
    };


    bool CppDijkstraVisitor::_bfsInnerLoopCond::operator()(const CppDijkstraVisitor::InnerLoopState& state, const CppDijkstraVisitor::InnerLoopContext& ctx) {
        CppDijkstraVisitor::EdgeList edgeList = CppDijkstraVisitor::second(state);
        return !CppDijkstraVisitor::isEmpty(edgeList);
    };


    CppDijkstraVisitor::_bfsInnerLoopCond CppDijkstraVisitor::bfsInnerLoopCond;

    void CppDijkstraVisitor::_bfsInnerLoopRepeat::operator()(CppDijkstraVisitor::InnerLoopState& s, const CppDijkstraVisitor::InnerLoopContext& c) {
        return __while_loop0.repeat(s, c);
    };


    CppDijkstraVisitor::_bfsInnerLoopRepeat CppDijkstraVisitor::bfsInnerLoopRepeat;

    void CppDijkstraVisitor::_bfsInnerLoopStep::operator()(CppDijkstraVisitor::InnerLoopState& state, const CppDijkstraVisitor::InnerLoopContext& ctx) {
        CppDijkstraVisitor::Graph g = CppDijkstraVisitor::first(ctx);
        CppDijkstraVisitor::Vertex u = CppDijkstraVisitor::second(ctx);
        CppDijkstraVisitor::OuterLoopState outerState = CppDijkstraVisitor::first(state);
        CppDijkstraVisitor::StateWithMaps x1 = CppDijkstraVisitor::first(outerState);
        CppDijkstraVisitor::PriorityQueue q1 = CppDijkstraVisitor::second(outerState);
        CppDijkstraVisitor::ColorPropertyMap c1 = CppDijkstraVisitor::third(outerState);
        CppDijkstraVisitor::EdgeList edgeList = CppDijkstraVisitor::second(state);
        CppDijkstraVisitor::Edge e = CppDijkstraVisitor::head(edgeList);
        CppDijkstraVisitor::EdgeList es = CppDijkstraVisitor::tail(edgeList);
        CppDijkstraVisitor::Vertex v = CppDijkstraVisitor::tgt(e);
        CppDijkstraVisitor::examineEdge(e, g, q1, x1);
        CppDijkstraVisitor::Color vc = CppDijkstraVisitor::get(c1, v);
        if ((vc) == (CppDijkstraVisitor::white()))
        {
            CppDijkstraVisitor::treeEdge(e, g, q1, x1);
            CppDijkstraVisitor::ColorPropertyMap c2 = CppDijkstraVisitor::put(c1, v, CppDijkstraVisitor::gray());
            CppDijkstraVisitor::discoverVertex(v, g, q1, x1);
            state = CppDijkstraVisitor::makeInnerLoopState(CppDijkstraVisitor::makeOuterLoopState(x1, CppDijkstraVisitor::push(v, q1), c2), es);
        }
        else
            if ((vc) == (CppDijkstraVisitor::gray()))
            {
                CppDijkstraVisitor::grayTarget(e, g, q1, x1);
                state = CppDijkstraVisitor::makeInnerLoopState(CppDijkstraVisitor::makeOuterLoopState(x1, q1, c1), es);
            }
            else
            {
                CppDijkstraVisitor::blackTarget(e, g, q1, x1);
                CppDijkstraVisitor::ColorPropertyMap c2 = CppDijkstraVisitor::put(c1, u, CppDijkstraVisitor::black());
                state = CppDijkstraVisitor::makeInnerLoopState(CppDijkstraVisitor::makeOuterLoopState(x1, q1, c2), es);
            }
    };


    while_loop<CppDijkstraVisitor::InnerLoopContext, CppDijkstraVisitor::InnerLoopState, CppDijkstraVisitor::_bfsInnerLoopCond, CppDijkstraVisitor::_bfsInnerLoopStep> CppDijkstraVisitor::__while_loop0;

    CppDijkstraVisitor::_bfsInnerLoopStep CppDijkstraVisitor::bfsInnerLoopStep;

    pair<CppDijkstraVisitor::OuterLoopState, CppDijkstraVisitor::EdgeList> CppDijkstraVisitor::__pair0;

    bool CppDijkstraVisitor::_bfsOuterLoopCond::operator()(const CppDijkstraVisitor::OuterLoopState& state, const CppDijkstraVisitor::Graph& g) {
        CppDijkstraVisitor::PriorityQueue q = CppDijkstraVisitor::second(state);
        return !CppDijkstraVisitor::isEmptyQueue(q);
    };


    CppDijkstraVisitor::_bfsOuterLoopCond CppDijkstraVisitor::bfsOuterLoopCond;

    void CppDijkstraVisitor::_bfsOuterLoopRepeat::operator()(CppDijkstraVisitor::OuterLoopState& s, const CppDijkstraVisitor::Graph& c) {
        return __while_loop.repeat(s, c);
    };


    CppDijkstraVisitor::_bfsOuterLoopRepeat CppDijkstraVisitor::bfsOuterLoopRepeat;

    void CppDijkstraVisitor::_bfsOuterLoopStep::operator()(CppDijkstraVisitor::OuterLoopState& state, const CppDijkstraVisitor::Graph& g) {
        CppDijkstraVisitor::StateWithMaps x = CppDijkstraVisitor::first(state);
        CppDijkstraVisitor::PriorityQueue q1 = CppDijkstraVisitor::second(state);
        CppDijkstraVisitor::ColorPropertyMap c = CppDijkstraVisitor::third(state);
        CppDijkstraVisitor::Vertex u = CppDijkstraVisitor::front(q1);
        CppDijkstraVisitor::PriorityQueue q2 = CppDijkstraVisitor::pop(q1);
        CppDijkstraVisitor::examineVertex(u, g, q2, x);
        CppDijkstraVisitor::InnerLoopState innerState = CppDijkstraVisitor::makeInnerLoopState(CppDijkstraVisitor::makeOuterLoopState(x, q2, c), CppDijkstraVisitor::outEdges(u, g));
        CppDijkstraVisitor::InnerLoopContext innerContext = CppDijkstraVisitor::makeInnerLoopContext(g, u);
        CppDijkstraVisitor::bfsInnerLoopRepeat(innerState, innerContext);
        CppDijkstraVisitor::OuterLoopState outerLoopStateAfterInnerLoop = CppDijkstraVisitor::first(innerState);
        CppDijkstraVisitor::StateWithMaps x_end = CppDijkstraVisitor::first(outerLoopStateAfterInnerLoop);
        CppDijkstraVisitor::PriorityQueue q_end = CppDijkstraVisitor::second(outerLoopStateAfterInnerLoop);
        CppDijkstraVisitor::ColorPropertyMap c_end = CppDijkstraVisitor::third(outerLoopStateAfterInnerLoop);
        CppDijkstraVisitor::finishVertex(u, g, q_end, x_end);
        state = CppDijkstraVisitor::makeOuterLoopState(x_end, q_end, c_end);
    };


    while_loop<CppDijkstraVisitor::Graph, CppDijkstraVisitor::OuterLoopState, CppDijkstraVisitor::_bfsOuterLoopCond, CppDijkstraVisitor::_bfsOuterLoopStep> CppDijkstraVisitor::__while_loop;

    CppDijkstraVisitor::_bfsOuterLoopStep CppDijkstraVisitor::bfsOuterLoopStep;

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_first::operator()(const CppDijkstraVisitor::VertexPair& p) {
        return __pair1.first(p);
    };

    CppDijkstraVisitor::VertexPredecessorMap CppDijkstraVisitor::_first::operator()(const CppDijkstraVisitor::PopulateVPMapState& p) {
        return __pair2.first(p);
    };

    CppDijkstraVisitor::Graph CppDijkstraVisitor::_first::operator()(const CppDijkstraVisitor::InnerLoopContext& p) {
        return __pair.first(p);
    };

    CppDijkstraVisitor::OuterLoopState CppDijkstraVisitor::_first::operator()(const CppDijkstraVisitor::InnerLoopState& p) {
        return __pair0.first(p);
    };

    CppDijkstraVisitor::StateWithMaps CppDijkstraVisitor::_first::operator()(const CppDijkstraVisitor::OuterLoopState& p) {
        return __triplet.first(p);
    };


    CppDijkstraVisitor::_first CppDijkstraVisitor::first;

    CppDijkstraVisitor::InnerLoopState CppDijkstraVisitor::_makeInnerLoopState::operator()(const CppDijkstraVisitor::OuterLoopState& a, const CppDijkstraVisitor::EdgeList& b) {
        return __pair0.makePair(a, b);
    };


    CppDijkstraVisitor::_makeInnerLoopState CppDijkstraVisitor::makeInnerLoopState;

    void CppDijkstraVisitor::_projectionBehaviorPair::operator()(const CppDijkstraVisitor::VertexPredecessorMap& a, const CppDijkstraVisitor::VertexList& b) {
        CppDijkstraVisitor::PopulateVPMapState pair = CppDijkstraVisitor::makePair(a, b);
        assert((CppDijkstraVisitor::first(pair)) == (a));
        assert((CppDijkstraVisitor::second(pair)) == (b));
    };

    void CppDijkstraVisitor::_projectionBehaviorPair::operator()(const CppDijkstraVisitor::Graph& a, const CppDijkstraVisitor::Vertex& b) {
        CppDijkstraVisitor::InnerLoopContext pair = CppDijkstraVisitor::makeInnerLoopContext(a, b);
        assert((CppDijkstraVisitor::first(pair)) == (a));
        assert((CppDijkstraVisitor::second(pair)) == (b));
    };

    void CppDijkstraVisitor::_projectionBehaviorPair::operator()(const CppDijkstraVisitor::OuterLoopState& a, const CppDijkstraVisitor::EdgeList& b) {
        CppDijkstraVisitor::InnerLoopState pair = CppDijkstraVisitor::makeInnerLoopState(a, b);
        assert((CppDijkstraVisitor::first(pair)) == (a));
        assert((CppDijkstraVisitor::second(pair)) == (b));
    };


    CppDijkstraVisitor::_projectionBehaviorPair CppDijkstraVisitor::projectionBehaviorPair;

    CppDijkstraVisitor::ColorPropertyMap CppDijkstraVisitor::_third::operator()(const CppDijkstraVisitor::OuterLoopState& p) {
        return __triplet.third(p);
    };


    CppDijkstraVisitor::_third CppDijkstraVisitor::third;

    void CppDijkstraVisitor::_whileLoopBehavior::operator()(const CppDijkstraVisitor::PopulateVPMapState& s, const CppDijkstraVisitor::Vertex& c) {
        CppDijkstraVisitor::PopulateVPMapState mutableState = s;
        if (CppDijkstraVisitor::populateVPMapLoopCond(s, c))
        {
            CppDijkstraVisitor::PopulateVPMapState mutableState1 = s;
            CppDijkstraVisitor::PopulateVPMapState mutableState2 = s;
            CppDijkstraVisitor::populateVPMapLoopRepeat(mutableState1, c);
            CppDijkstraVisitor::populateVPMapLoopStep(mutableState2, c);
            assert((mutableState1) == (mutableState2));
        }
        else
        {
            CppDijkstraVisitor::PopulateVPMapState mutableState1 = s;
            CppDijkstraVisitor::populateVPMapLoopRepeat(mutableState1, c);
            assert((mutableState1) == (s));
        }
    };

    void CppDijkstraVisitor::_whileLoopBehavior::operator()(const CppDijkstraVisitor::OuterLoopState& s, const CppDijkstraVisitor::Graph& c) {
        CppDijkstraVisitor::OuterLoopState mutableState = s;
        if (CppDijkstraVisitor::bfsOuterLoopCond(s, c))
        {
            CppDijkstraVisitor::OuterLoopState mutableState1 = s;
            CppDijkstraVisitor::OuterLoopState mutableState2 = s;
            CppDijkstraVisitor::bfsOuterLoopRepeat(mutableState1, c);
            CppDijkstraVisitor::bfsOuterLoopStep(mutableState2, c);
            assert((mutableState1) == (mutableState2));
        }
        else
        {
            CppDijkstraVisitor::OuterLoopState mutableState1 = s;
            CppDijkstraVisitor::bfsOuterLoopRepeat(mutableState1, c);
            assert((mutableState1) == (s));
        }
    };

    void CppDijkstraVisitor::_whileLoopBehavior::operator()(const CppDijkstraVisitor::InnerLoopState& s, const CppDijkstraVisitor::InnerLoopContext& c) {
        CppDijkstraVisitor::InnerLoopState mutableState = s;
        if (CppDijkstraVisitor::bfsInnerLoopCond(s, c))
        {
            CppDijkstraVisitor::InnerLoopState mutableState1 = s;
            CppDijkstraVisitor::InnerLoopState mutableState2 = s;
            CppDijkstraVisitor::bfsInnerLoopRepeat(mutableState1, c);
            CppDijkstraVisitor::bfsInnerLoopStep(mutableState2, c);
            assert((mutableState1) == (mutableState2));
        }
        else
        {
            CppDijkstraVisitor::InnerLoopState mutableState1 = s;
            CppDijkstraVisitor::bfsInnerLoopRepeat(mutableState1, c);
            assert((mutableState1) == (s));
        }
    };


    CppDijkstraVisitor::_whileLoopBehavior CppDijkstraVisitor::whileLoopBehavior;

    triplet<CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap> CppDijkstraVisitor::__triplet;

    void CppDijkstraVisitor::_blackTarget::operator()(const CppDijkstraVisitor::Edge& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
        ;
    };


    CppDijkstraVisitor::_blackTarget CppDijkstraVisitor::blackTarget;

    void CppDijkstraVisitor::_breadthFirstVisit::operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::Vertex& s, CppDijkstraVisitor::StateWithMaps& a, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c) {
        CppDijkstraVisitor::discoverVertex(s, g, q, a);
        CppDijkstraVisitor::PriorityQueue q1 = CppDijkstraVisitor::push(s, q);
        CppDijkstraVisitor::ColorPropertyMap c1 = CppDijkstraVisitor::put(c, s, CppDijkstraVisitor::gray());
        CppDijkstraVisitor::OuterLoopState outerState = CppDijkstraVisitor::makeOuterLoopState(a, q1, c1);
        CppDijkstraVisitor::bfsOuterLoopRepeat(outerState, g);
        a = CppDijkstraVisitor::first(outerState);
    };


    CppDijkstraVisitor::_breadthFirstVisit CppDijkstraVisitor::breadthFirstVisit;

    void CppDijkstraVisitor::_discoverVertex::operator()(const CppDijkstraVisitor::Vertex& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
        ;
    };


    CppDijkstraVisitor::_discoverVertex CppDijkstraVisitor::discoverVertex;

    CppDijkstraVisitor::PriorityQueue CppDijkstraVisitor::_emptyPriorityQueue::operator()(const CppDijkstraVisitor::VertexCostMap& pm) {
        return __priority_queue.empty(pm);
    };


    CppDijkstraVisitor::_emptyPriorityQueue CppDijkstraVisitor::emptyPriorityQueue;

    void CppDijkstraVisitor::_examineEdge::operator()(const CppDijkstraVisitor::Edge& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
        ;
    };


    CppDijkstraVisitor::_examineEdge CppDijkstraVisitor::examineEdge;

    void CppDijkstraVisitor::_examineVertex::operator()(const CppDijkstraVisitor::Vertex& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
        ;
    };


    CppDijkstraVisitor::_examineVertex CppDijkstraVisitor::examineVertex;

    void CppDijkstraVisitor::_finishVertex::operator()(const CppDijkstraVisitor::Vertex& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
        ;
    };


    CppDijkstraVisitor::_finishVertex CppDijkstraVisitor::finishVertex;

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_front::operator()(const CppDijkstraVisitor::PriorityQueue& q) {
        return __priority_queue.front(q);
    };


    CppDijkstraVisitor::_front CppDijkstraVisitor::front;

    void CppDijkstraVisitor::_grayTarget::operator()(const CppDijkstraVisitor::Edge& e, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& pq, CppDijkstraVisitor::StateWithMaps& swm) {
        CppDijkstraVisitor::VertexCostMap origVcm = CppDijkstraVisitor::getVertexCostMap(swm);
        CppDijkstraVisitor::VertexPredecessorMap vpm = CppDijkstraVisitor::getVertexPredecessorMap(swm);
        CppDijkstraVisitor::EdgeCostMap ecm = CppDijkstraVisitor::getEdgeCostMap(swm);
        CppDijkstraVisitor::VertexCostMap vcm = origVcm;
        CppDijkstraVisitor::relax(e, ecm, vcm, vpm);
        if ((vcm) == (origVcm))
            ;
        else
        {
            swm = CppDijkstraVisitor::putVertexPredecessorMap(vpm, CppDijkstraVisitor::putVertexCostMap(vcm, swm));
            pq = CppDijkstraVisitor::update(vcm, CppDijkstraVisitor::tgt(e), pq);
        }
    };


    CppDijkstraVisitor::_grayTarget CppDijkstraVisitor::grayTarget;

    bool CppDijkstraVisitor::_isEmptyQueue::operator()(const CppDijkstraVisitor::PriorityQueue& q) {
        return __priority_queue.isEmpty(q);
    };


    CppDijkstraVisitor::_isEmptyQueue CppDijkstraVisitor::isEmptyQueue;

    CppDijkstraVisitor::OuterLoopState CppDijkstraVisitor::_makeOuterLoopState::operator()(const CppDijkstraVisitor::StateWithMaps& a, const CppDijkstraVisitor::PriorityQueue& b, const CppDijkstraVisitor::ColorPropertyMap& c) {
        return __triplet.makeTriplet(a, b, c);
    };


    CppDijkstraVisitor::_makeOuterLoopState CppDijkstraVisitor::makeOuterLoopState;

    void CppDijkstraVisitor::_nonTreeEdge::operator()(const CppDijkstraVisitor::Edge& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
        ;
    };


    CppDijkstraVisitor::_nonTreeEdge CppDijkstraVisitor::nonTreeEdge;

    CppDijkstraVisitor::PriorityQueue CppDijkstraVisitor::_pop::operator()(const CppDijkstraVisitor::PriorityQueue& q) {
        return __priority_queue.pop(q);
    };


    CppDijkstraVisitor::_pop CppDijkstraVisitor::pop;

    void CppDijkstraVisitor::_projectionBehaviorTriplet::operator()(const CppDijkstraVisitor::StateWithMaps& a, const CppDijkstraVisitor::PriorityQueue& b, const CppDijkstraVisitor::ColorPropertyMap& c) {
        CppDijkstraVisitor::OuterLoopState triplet = CppDijkstraVisitor::makeOuterLoopState(a, b, c);
        assert((CppDijkstraVisitor::first(triplet)) == (a));
        assert((CppDijkstraVisitor::second(triplet)) == (b));
        assert((CppDijkstraVisitor::third(triplet)) == (c));
    };


    CppDijkstraVisitor::_projectionBehaviorTriplet CppDijkstraVisitor::projectionBehaviorTriplet;

    CppDijkstraVisitor::PriorityQueue CppDijkstraVisitor::_push::operator()(const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::PriorityQueue& q) {
        return __priority_queue.push(a, q);
    };


    CppDijkstraVisitor::_push CppDijkstraVisitor::push;

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_second::operator()(const CppDijkstraVisitor::VertexPair& p) {
        return __pair1.second(p);
    };

    CppDijkstraVisitor::VertexList CppDijkstraVisitor::_second::operator()(const CppDijkstraVisitor::PopulateVPMapState& p) {
        return __pair2.second(p);
    };

    CppDijkstraVisitor::Vertex CppDijkstraVisitor::_second::operator()(const CppDijkstraVisitor::InnerLoopContext& p) {
        return __pair.second(p);
    };

    CppDijkstraVisitor::EdgeList CppDijkstraVisitor::_second::operator()(const CppDijkstraVisitor::InnerLoopState& p) {
        return __pair0.second(p);
    };

    CppDijkstraVisitor::PriorityQueue CppDijkstraVisitor::_second::operator()(const CppDijkstraVisitor::OuterLoopState& p) {
        return __triplet.second(p);
    };


    CppDijkstraVisitor::_second CppDijkstraVisitor::second;

    void CppDijkstraVisitor::_treeEdge::operator()(const CppDijkstraVisitor::Edge& e, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& pq, CppDijkstraVisitor::StateWithMaps& swm) {
        CppDijkstraVisitor::VertexCostMap vcm = CppDijkstraVisitor::getVertexCostMap(swm);
        CppDijkstraVisitor::VertexPredecessorMap vpm = CppDijkstraVisitor::getVertexPredecessorMap(swm);
        CppDijkstraVisitor::EdgeCostMap ecm = CppDijkstraVisitor::getEdgeCostMap(swm);
        CppDijkstraVisitor::relax(e, ecm, vcm, vpm);
        swm = CppDijkstraVisitor::putVertexPredecessorMap(vpm, CppDijkstraVisitor::putVertexCostMap(vcm, swm));
    };


    CppDijkstraVisitor::_treeEdge CppDijkstraVisitor::treeEdge;

    CppDijkstraVisitor::PriorityQueue CppDijkstraVisitor::_update::operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::Vertex& a, const CppDijkstraVisitor::PriorityQueue& pq) {
        return __priority_queue.update(pm, a, pq);
    };


    CppDijkstraVisitor::_update CppDijkstraVisitor::update;

    priority_queue<CppDijkstraVisitor::Vertex, CppDijkstraVisitor::Cost, CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::_get> CppDijkstraVisitor::__priority_queue;

    CppDijkstraVisitor::_get CppDijkstraVisitor::get;

    CppDijkstraVisitor::Color CppDijkstraVisitor::_gray::operator()() {
        return __color_marker.gray();
    };


    CppDijkstraVisitor::_gray CppDijkstraVisitor::gray;

    CppDijkstraVisitor::VertexCostMap CppDijkstraVisitor::_initMap::operator()(const CppDijkstraVisitor::VertexList& kl, const CppDijkstraVisitor::Cost& v) {
        return __read_write_property_map1.initMap(kl, v);
    };

    CppDijkstraVisitor::VertexPredecessorMap CppDijkstraVisitor::_initMap::operator()(const CppDijkstraVisitor::VertexList& kl, const CppDijkstraVisitor::Vertex& v) {
        return __read_write_property_map2.initMap(kl, v);
    };

    CppDijkstraVisitor::EdgeCostMap CppDijkstraVisitor::_initMap::operator()(const CppDijkstraVisitor::EdgeList& kl, const CppDijkstraVisitor::Cost& v) {
        return __read_write_property_map.initMap(kl, v);
    };

    CppDijkstraVisitor::ColorPropertyMap CppDijkstraVisitor::_initMap::operator()(const CppDijkstraVisitor::VertexList& kl, const CppDijkstraVisitor::Color& v) {
        return __read_write_property_map0.initMap(kl, v);
    };


    CppDijkstraVisitor::_initMap CppDijkstraVisitor::initMap;

    CppDijkstraVisitor::VertexCostMap CppDijkstraVisitor::_put::operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::Vertex& k, const CppDijkstraVisitor::Cost& v) {
        return __read_write_property_map1.put(pm, k, v);
    };

    CppDijkstraVisitor::VertexPredecessorMap CppDijkstraVisitor::_put::operator()(const CppDijkstraVisitor::VertexPredecessorMap& pm, const CppDijkstraVisitor::Vertex& k, const CppDijkstraVisitor::Vertex& v) {
        return __read_write_property_map2.put(pm, k, v);
    };

    CppDijkstraVisitor::EdgeCostMap CppDijkstraVisitor::_put::operator()(const CppDijkstraVisitor::EdgeCostMap& pm, const CppDijkstraVisitor::Edge& k, const CppDijkstraVisitor::Cost& v) {
        return __read_write_property_map.put(pm, k, v);
    };

    CppDijkstraVisitor::ColorPropertyMap CppDijkstraVisitor::_put::operator()(const CppDijkstraVisitor::ColorPropertyMap& pm, const CppDijkstraVisitor::Vertex& k, const CppDijkstraVisitor::Color& v) {
        return __read_write_property_map0.put(pm, k, v);
    };


    CppDijkstraVisitor::_put CppDijkstraVisitor::put;

    CppDijkstraVisitor::Color CppDijkstraVisitor::_white::operator()() {
        return __color_marker.white();
    };


    CppDijkstraVisitor::_white CppDijkstraVisitor::white;

} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp