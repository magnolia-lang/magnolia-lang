#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include "base.hpp"
#include "gen/examples/bgl_v2/mg-src/bgl_v2-cpp.hpp"

namespace bfs {
// BFS test
using examples::bgl_v2::mg_src::bgl_v2_cpp::CppBFSTestVisitor;
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

inline std::list<CppBFSTestVisitor::Edge> lotsOfEdges() {
    int nb_vertices = 8000, nb_edges = 80000;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type>
        range_obj(0, nb_vertices);

    std::list<CppBFSTestVisitor::Edge> edges;

    for (auto i = 0; i < nb_edges; ++i) {
        edges.push_back(makeEdge(range_obj(rng), range_obj(rng)));
    }

    return edges;
}

inline void bfsTest() {
    std::cout << "BFS test:" << std::endl;
    Graph g(testEdges());
    Vertex start = 0;

    auto bfsResult =
        CppBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexList());

    // Nodes are returned in the wrong order
    bfsResult.reverse();

    for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;
}

inline void bfsPerfTest() {
    std::cout << "BFS perf test:" << std::endl;
    Graph g(lotsOfEdges());
    Vertex start = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto bfsResult =
        CppBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexList());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // Nodes are returned in the wrong order
    bfsResult.reverse();

    for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
}


}
