#pragma once

#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"

namespace bellmanFord {
// BellmanFord test
using examples::bgl::mg_src::bgl_cpp::CppBellmanFord;
typedef CppBellmanFord::Cost Cost;
typedef CppBellmanFord::Edge Edge;
typedef CppBellmanFord::Graph Graph;
typedef CppBellmanFord::Vertex Vertex;
typedef CppBellmanFord::VertexCostMap VertexCostMap;
typedef CppBellmanFord::EdgeCostMap EdgeCostMap;
typedef CppBellmanFord::VertexDescriptor VertexDescriptor;
typedef CppBellmanFord::VertexIterator VertexIterator;
typedef CppBellmanFord::VertexPredecessorMap VertexPredecessorMap;

auto makeEdge = CppBellmanFord::makeEdge;
auto get = CppBellmanFord::get;
auto put = CppBellmanFord::put;
auto toEdgeDescriptor = CppBellmanFord::toEdgeDescriptor;
auto toVertexDescriptor = CppBellmanFord::toVertexDescriptor;


inline void testGraphWithCost(Graph &g, EdgeCostMap &ecm) {
    std::list<CppBellmanFord::Edge> edges;
    ecm = CppBellmanFord::emptyECMap();

    int u = 0, v = 1, x = 2, y = 3, z = 4;

    edges.push_back(makeEdge(u, y));
    edges.push_back(makeEdge(u, x));
    edges.push_back(makeEdge(u, v));
    edges.push_back(makeEdge(v, u));
    edges.push_back(makeEdge(x, y));
    edges.push_back(makeEdge(x, v));
    edges.push_back(makeEdge(y, v));
    edges.push_back(makeEdge(y, z));
    edges.push_back(makeEdge(z, u));
    edges.push_back(makeEdge(z, x));

    g = Graph(edges.begin(), edges.end(),
              boost::graph_traits<Graph>::vertices_size_type(5));

    put(ecm, toEdgeDescriptor(toVertexDescriptor(u, g), toVertexDescriptor(y, g), g), -4.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(u, g), toVertexDescriptor(x, g), g), 8.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(u, g), toVertexDescriptor(v, g), g), 5.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(v, g), toVertexDescriptor(u, g), g), -2.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(x, g), toVertexDescriptor(y, g), g), 9.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(x, g), toVertexDescriptor(v, g), g), -3.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(y, g), toVertexDescriptor(v, g), g), 7.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(y, g), toVertexDescriptor(z, g), g), 2.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(z, g), toVertexDescriptor(u, g), g), 6.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(z, g), toVertexDescriptor(x, g), g), 7.0);
}

inline void bellmanFordTest() {
    std::cout << "Bellman-Ford test:" << std::endl;

    Graph g;
    EdgeCostMap ecm;
    testGraphWithCost(g, ecm);

    int z = 4;
    VertexDescriptor start = toVertexDescriptor(z, g);
    Cost start_cost = 0;
    Cost other_vertex_base_cost = 100.0;

    VertexIterator itr;
    CppBellmanFord::vertices(g, itr);

    VertexCostMap vcm = CppBellmanFord::initMap(
        itr, other_vertex_base_cost);
    put(vcm, start, start_cost);

    VertexPredecessorMap vpm;

    CppBellmanFord::Bool allMinimized;
    CppBellmanFord::Unit u = CppBellmanFord::unit();

    CppBellmanFord::bellmanFordShortestPaths(g, vcm, ecm, u, vpm, allMinimized);

    for (auto i = 0; i < 5; ++i) {
        std::cout << "Distance to " << i << ": " << get(vcm, i) << std::endl;
    }
}
}
