//#include "base.hpp"
#include <iostream>
#include <utility>
#include <vector>
#include "gen/examples/bgl/mg-src/bgl.hpp"

using namespace examples::bgl::mg_src::bgl;
typedef IncidenceGraphWalk::StringVertex Vertex;
typedef IncidenceGraphWalk::VertexCollection VertexCollection;
typedef IncidenceGraphWalk::Graph Graph;
typedef IncidenceGraphWalk::Edge Edge;
typedef IncidenceGraphWalk::EdgeCollection EdgeCollection;

Graph parse_graph() {
    unsigned int nb_edges;
    EdgeCollection edges;

    std::cout << "Number of edges to parse: ";
    std::cin >> nb_edges;

    for (unsigned int i = 0; i < nb_edges; ++i) {
        Vertex source, target;
        std::cout << "Space-separated source and target: ";
        std::cin >> source >> target;

        edges = IncidenceGraphWalk::addToCollection(
            edges, IncidenceGraphWalk::make_edge(source, target));
    }

    return Graph(edges);
}

int main(int argc, char **argv) {
    auto graph = parse_graph();
    Vertex start;

    std::cout << "Start vertex: ";
    std::cin >> start;

    examples::bgl::mg_src::bgl::IncidenceGraphWalk::bfs(graph, start);
    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.
    //examples::fizzbuzz::mg_src::fizzbuzz::MyFizzBuzzProgram P;

    //std::cout << "Fizz my buzz: " << P.doFizzBuzz(atoi(argv[1])) << std::endl;
    return 0;
}
