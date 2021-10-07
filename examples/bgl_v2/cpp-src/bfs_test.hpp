#pragma once

#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl_v2/mg-src/bgl_v2.hpp"

namespace bfs {
// BFS test
using examples::bgl_v2::mg_src::bgl_v2::CppBFSTestVisitor;
typedef CppBFSTestVisitor::Graph Graph;
typedef CppBFSTestVisitor::Vertex Vertex;

auto makeEdge = CppBFSTestVisitor::makeEdge;
auto emptyVertexList = CppBFSTestVisitor::emptyVertexList;
auto head = CppBFSTestVisitor::head;
auto tail = CppBFSTestVisitor::tail;

inline std::list<CppBFSTestVisitor::Edge> testEdges() {
    std::list<CppBFSTestVisitor::Edge> edges;
    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(4, 5));
    edges.push_back(makeEdge(3, 6));

    return edges;
}

inline void bfsTest() {
    std::cout << "BFS test:" << std::endl;
    Graph g(testEdges());
    Vertex start = 0;

    auto bfsResult =
        CppBFSTestVisitor::breadthFirstSearch(g, 0, emptyVertexList());

    // Nodes are returned in the wrong order
    bfsResult.reverse();

    for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;
}
}
