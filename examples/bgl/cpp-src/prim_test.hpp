#pragma once

#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"

namespace prim {
// Prim test
using examples::bgl::mg_src::bgl_cpp::CppPrimVisitor;
typedef CppPrimVisitor::Cost Cost;
typedef CppPrimVisitor::Edge Edge;
typedef CppPrimVisitor::Graph Graph;
typedef CppPrimVisitor::Vertex Vertex;
typedef CppPrimVisitor::VertexCostMap VertexCostMap;
typedef CppPrimVisitor::EdgeCostMap EdgeCostMap;
typedef CppPrimVisitor::VertexDescriptor VertexDescriptor;
typedef CppPrimVisitor::VertexIterator VertexIterator;
typedef CppPrimVisitor::VertexPredecessorMap VertexPredecessorMap;

auto makeEdge = CppPrimVisitor::makeEdge;
auto get = CppPrimVisitor::get;
auto put = CppPrimVisitor::put;
auto toEdgeDescriptor = CppPrimVisitor::toEdgeDescriptor;
auto toVertexDescriptor = CppPrimVisitor::toVertexDescriptor;


inline void testGraphWithCost(Graph &g, EdgeCostMap &ecm) {
    std::list<CppPrimVisitor::Edge> edges;
    ecm = CppPrimVisitor::emptyECMap();

    edges.push_back(makeEdge(0, 2));
    edges.push_back(makeEdge(1, 1));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(1, 4));
    edges.push_back(makeEdge(2, 1));
    edges.push_back(makeEdge(2, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(4, 0));
    edges.push_back(makeEdge(2, 0));
    edges.push_back(makeEdge(3, 1));
    edges.push_back(makeEdge(4, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(3, 2));
    edges.push_back(makeEdge(4, 3));
    edges.push_back(makeEdge(0, 4));

    g = Graph(edges.begin(), edges.end(),
              boost::graph_traits<Graph>::vertices_size_type(5));

    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(2, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(1, g), g), 2);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(3, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(4, g), g), 2);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(2, g), toVertexDescriptor(1, g), g), 7);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(2, g), toVertexDescriptor(3, g), g), 3);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(3, g), toVertexDescriptor(4, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(4, g), toVertexDescriptor(0, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(2, g), toVertexDescriptor(0, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(3, g), toVertexDescriptor(1, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(4, g), toVertexDescriptor(1, g), g), 2);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(2, g), g), 7);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(3, g), toVertexDescriptor(2, g), g), 3);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(4, g), toVertexDescriptor(3, g), g), 1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(4, g), g), 1);
}

inline void primTest() {
    std::cout << "Prim test:" << std::endl;

    Graph g;
    EdgeCostMap ecm;
    testGraphWithCost(g, ecm);

    VertexDescriptor start = toVertexDescriptor(0, g);
    Cost start_cost = 0;
    Cost other_vertex_base_cost = 100.0;

    VertexIterator itr;
    CppPrimVisitor::vertices(g, itr);

    VertexCostMap vcm = CppPrimVisitor::initMap(
        itr, other_vertex_base_cost);

    VertexPredecessorMap vpm;

    CppPrimVisitor::primMinimumSpanningTree(g, start, vcm, ecm, start_cost, vpm);

    for (auto i = 0; i < 5; ++i) {
        std::cout << "Parent of " << i << ": " << get(vpm, i) << std::endl;
    }
}
}
