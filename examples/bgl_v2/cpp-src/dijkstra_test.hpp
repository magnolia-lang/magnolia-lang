#pragma once

#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl_v2/mg-src/bgl_v2-cpp.hpp"

namespace dijkstra {
// Dijkstra test
using examples::bgl_v2::mg_src::bgl_v2_cpp::CppDijkstraVisitor;
typedef CppDijkstraVisitor::Cost Cost;
typedef CppDijkstraVisitor::Edge Edge;
typedef CppDijkstraVisitor::Graph Graph;
typedef CppDijkstraVisitor::Vertex Vertex;
typedef CppDijkstraVisitor::VertexCostMap VertexCostMap;
typedef CppDijkstraVisitor::EdgeCostMap EdgeCostMap;
typedef CppDijkstraVisitor::VertexDescriptor VertexDescriptor;
typedef CppDijkstraVisitor::VertexIterator VertexIterator;
typedef CppDijkstraVisitor::VertexPredecessorMap VertexPredecessorMap;

auto makeEdge = CppDijkstraVisitor::makeEdge;
auto get = CppDijkstraVisitor::get;
auto put = CppDijkstraVisitor::put;
auto toEdgeDescriptor = CppDijkstraVisitor::toEdgeDescriptor;
auto toVertexDescriptor = CppDijkstraVisitor::toVertexDescriptor;


inline void testGraphWithCost(Graph &g, EdgeCostMap &ecm) {
    std::list<CppDijkstraVisitor::Edge> edges;
    ecm = CppDijkstraVisitor::emptyECMap();

    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));

    g = Graph(edges.begin(), edges.end(),
              boost::graph_traits<Graph>::vertices_size_type(5));

    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(1, g), g), 0.5);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(2, g), g), 5.6);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(3, g), g), 0.2);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(3, g), toVertexDescriptor(4, g), g), 0.1);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(4, g), g), 3.2);

    // Some more tests
    //edges.push_back(makeEdge(0, 4));
    ////put(ecm, makeEdge(0, 4), 0.6);

    //edges.push_back(makeEdge(4, 2));
    ////put(ecm, makeEdge(4, 2), 1.5);
}

inline void dijkstraTest() {
    std::cout << "Dijkstra test:" << std::endl;

    Graph g;
    EdgeCostMap ecm;
    testGraphWithCost(g, ecm);

    VertexDescriptor start = toVertexDescriptor(0, g);
    Cost start_cost = 0;
    Cost other_vertex_base_cost = 100.0;

    VertexIterator itr;
    CppDijkstraVisitor::vertices(g, itr);
    
    VertexCostMap vcm = CppDijkstraVisitor::initMap(
        itr, other_vertex_base_cost);

    VertexPredecessorMap vpm;

    CppDijkstraVisitor::dijkstraShortestPaths(g, start, vcm, ecm, start_cost,
                                              vpm);

    for (auto i = 0; i < 5; ++i) {
        std::cout << "Distance to " << i << ": " << get(vcm, i) << std::endl; 
    }
}
}
