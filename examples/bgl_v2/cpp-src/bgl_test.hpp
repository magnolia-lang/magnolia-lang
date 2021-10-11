#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/pending/indirect_cmp.hpp>
#include <boost/range/irange.hpp>

#include <chrono>
#include <iostream>
#include <random>

#include "testcase.hpp"

namespace bgl {

using namespace boost;
template < typename TimeMap > class bfs_time_visitor:public default_bfs_visitor {
  typedef typename property_traits < TimeMap >::value_type T;
public:
  bfs_time_visitor(TimeMap tmap, T & t):m_timemap(tmap), m_time(t) { }
  template < typename Vertex, typename Graph >
    void discover_vertex(Vertex u, const Graph & g) const
  {
    put(m_timemap, u, m_time++);
  }
  TimeMap m_timemap;
  T & m_time;
};


inline void
testBgl()
{
  using namespace boost;
  // Select the graph type we wish to use
  typedef adjacency_list < vecS, vecS, directedS > graph_t;
  // Set up the vertex IDs and names
  // Specify the edges in the graph
  typedef std::pair < int, int >E;

  auto test_case = gen_test_case();

  int N = test_case.first;
  std::list<E> edge_list;
  
  for (auto pair_it = test_case.second.begin();
       pair_it != test_case.second.end();
       ++pair_it) {
    edge_list.push_back(E(pair_it->first, pair_it->second));
  }
  // Create the graph object
  const int n_edges = edge_list.size();
  typedef graph_traits<graph_t>::vertices_size_type v_size_t;
  graph_t g(edge_list.begin(), edge_list.end(), v_size_t(N));

  // Typedefs
  typedef graph_traits < graph_t >::vertex_descriptor Vertex;
  typedef graph_traits < graph_t >::vertices_size_type Size;
  typedef Size* Iiter;

  // a vector to hold the discover time property for each vertex
  std::vector < Size > dtime(num_vertices(g));

  for (auto i = 0; i < num_vertices(g); ++i) {
    dtime[i] = -1;
  }

  Size time = 0;
  bfs_time_visitor < Size * >vis(&dtime[0], time);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  breadth_first_search(g, vertex(0, g), visitor(vis));

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // Use std::sort to order the vertices by their discover time
  std::vector<graph_traits<graph_t>::vertices_size_type > discover_order(N);
  integer_range < int >range(0, N);
  std::copy(range.begin(), range.end(), discover_order.begin());
  std::sort(discover_order.begin(), discover_order.end(),
            indirect_cmp < Iiter, std::less < Size > >(&dtime[0]));

  std::cout << "order of discovery: ";
  for (int i = 0; i < N; ++i) {
    //if (dtime[discover_order[i]] == -1) std::cout << "lol OK";
    if (dtime[discover_order[i]] != -1) std::cout << discover_order[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}
}
