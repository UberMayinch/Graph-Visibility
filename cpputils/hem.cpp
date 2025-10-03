#include <bits/stdc++.h>
using namespace std;


struct Edge { int to; double w; };
using Adj = vector<vector<Edge>>; // undirected adjacency list

struct Graph {
    int n = 0;              // number of vertices
    size_t m_undirected = 0; // number of undirected edges
    Adj adj;                // adjacency list (each undirected edge stored twice)

    Graph() = default;
    explicit Graph(int n_) : n(n_), adj(n_) {}

    void add_edge(int u, int v, double w) {
        if (u == v) return; // skip self-loops for HEM
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        ++m_undirected;
    }
};

struct CoarseResult {
    vector<int> match;      // match[u] = v or -1
    vector<int> coarse_id;  // map original vertex -> coarse vertex id
    int num_coarse = 0;     // |V'|

    // Coarse graph stored as edge list (u', v', w') to avoid duplicating adjacency
    struct CEdge { int u, v; double w; };
    vector<CEdge> edges;    // undirected, unique pairs
};

// Utility: fast hash for pair<int,int>
struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        // 64-bit mix
        uint64_t x = (uint64_t)(unsigned)p.first << 32 | (uint32_t)(unsigned)p.second;
        x ^= (x >> 33); x *= 0xff51afd7ed558ccdULL;
        x ^= (x >> 33); x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= (x >> 33);
        return (size_t)x;
    }
};

// -------------------------------------------
// Heavy Edge Matching
// -------------------------------------------

CoarseResult heavy_edge_matching(const Graph& G, uint64_t rng_seed = 42) {
    const int n = G.n;
    CoarseResult R;
    R.match.assign(n, -1);

    // Random vertex order for better matching quality
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(rng_seed);
    shuffle(order.begin(), order.end(), rng);

    // Greedy HEM: for each unmatched u in random order, match to its heaviest unmatched neighbor
    for (int u : order) {
        if (R.match[u] != -1) continue;
        const auto& Nu = G.adj[u];
        int best_v = -1; double best_w = -numeric_limits<double>::infinity();
        // Scan neighbors once
        for (const Edge& e : Nu) {
            int v = e.to; if (R.match[v] != -1) continue;
            if (e.w > best_w) { best_w = e.w; best_v = v; }
        }
        if (best_v != -1) {
            R.match[u] = best_v;
            R.match[best_v] = u;
        }
    }

    // Build coarse vertex IDs
    R.coarse_id.assign(n, -1);
    int cid = 0;
    for (int u = 0; u < n; ++u) {
        if (R.coarse_id[u] != -1) continue;
        int v = R.match[u];
        R.coarse_id[u] = cid;
        if (v != -1 && R.coarse_id[v] == -1) {
            R.coarse_id[v] = cid; // pair collapsed into same coarse vertex
        }
        ++cid;
    }
    R.num_coarse = cid;

    // Aggregate edges to form coarse graph E'
    // Use hash map keyed by (min(a,b), max(a,b)) to sum weights
    unordered_map<pair<int,int>, double, PairHash> agg;
    agg.reserve(G.m_undirected * 2 + 1);

    for (int u = 0; u < n; ++u) {
        int a = R.coarse_id[u];
        for (const Edge& e : G.adj[u]) {
            int v = e.to; if (u < v) { // process each undirected edge once
                int b = R.coarse_id[v];
                if (a == b) continue; // becomes self-loop after contraction: drop
                auto key = (a < b) ? make_pair(a,b) : make_pair(b,a);
                agg[key] += e.w; // sum weights between supernodes
            }
        }
    }

    R.edges.reserve(agg.size());
    for (auto& kv : agg) {
        R.edges.push_back({kv.first.first, kv.first.second, kv.second});
    }

    return R;
}

// -------------------------------------------
// Utility: random graph generator (Erdos-Renyi G(n, p) sparse) with positive weights
// Ensures no duplicate edges and no self-loops.
// -------------------------------------------
Graph make_random_graph(int n, size_t m, uint64_t seed = 123) {
    Graph G(n);
    std::mt19937_64 rng(seed);
    // We target exactly m edges by sampling without replacement using a hash set
    // For large n and relatively small m, this is efficient.
    struct PairHash64 { size_t operator()(const pair<int,int>& p) const noexcept { return ((uint64_t)p.first<<32) ^ (uint32_t)p.second; } };
    unordered_set<pair<int,int>, PairHash64> used;
    used.reserve(m*2+1);

    uniform_int_distribution<int> U(0, n-1);
    uniform_real_distribution<double> W(0.0, 1.0); // weights in (0,1)

    while (G.m_undirected < m) {
        int u = U(rng), v = U(rng);
        if (u == v) continue;
        int a = min(u,v), b = max(u,v);
        if (used.insert({a,b}).second) {
            double w = W(rng) + 0.001; // avoid zero
            G.add_edge(u, v, w);
        }
    }
    return G;
}

// -------------------------------------------
// Timing helpers
// -------------------------------------------
struct Timer {
    chrono::high_resolution_clock::time_point t0;
    void tic() { t0 = chrono::high_resolution_clock::now(); }
    double toc_ms() const {
        auto t1 = chrono::high_resolution_clock::now();
        return chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// -------------------------------------------
// Basic sanity check on a tiny graph
// -------------------------------------------
void tiny_test() {
    cout << "\n--- Tiny sanity test ---\n";
    Graph G(6);
    // Form a small graph with obvious heavy edges
    G.add_edge(0,1, 5.0);
    G.add_edge(0,2, 1.0);
    G.add_edge(1,3, 2.0);
    G.add_edge(2,3, 4.0);
    G.add_edge(4,5, 10.0);

    auto R = heavy_edge_matching(G, 7);

    cout << "n=" << G.n << ", m=" << G.m_undirected << "\n";
    int matched_pairs = 0, unmatched = 0;
    vector<char> seen(G.n, 0);
    for (int u = 0; u < G.n; ++u) {
        if (seen[u]) continue;
        int v = R.match[u];
        if (v != -1) { matched_pairs++; seen[u]=seen[v]=1; }
        else { unmatched++; seen[u]=1; }
    }
    cout << "matched_pairs=" << matched_pairs << ", unmatched_vertices=" << unmatched << "\n";
    cout << "coarse |V'|=" << R.num_coarse << ", |E'|=" << R.edges.size() << "\n";
}

// -------------------------------------------
// Main: build a random graph, run HEM, print stats (and a few edges)
// Usage examples:
//   ./hem --n 100000 --m 300000 --seed 1 --rng 42
// -------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n = 20000;          // default vertices
    size_t m = 80000;       // default undirected edges
    uint64_t seed_graph = 1;
    uint64_t seed_match = 42;
    bool run_tiny = true;   // also run tiny test

    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        auto need = [&](const char* flag){ return s == flag && i+1 < argc; };
        if (need("--n")) { n = stoi(argv[++i]); }
        else if (need("--m")) { m = (size_t)stoll(argv[++i]); }
        else if (need("--seed")) { seed_graph = stoull(argv[++i]); }
        else if (need("--rng")) { seed_match = stoull(argv[++i]); }
        else if (s == "--no-tiny") { run_tiny = false; }
    }

    if (run_tiny) tiny_test();

    cout << "\n--- Random graph benchmark ---\n";
    cout << "Generating G with n=" << n << ", m=" << m << " ..." << flush;
    Timer t; t.tic();
    Graph G = make_random_graph(n, m, seed_graph);
    double t_gen = t.toc_ms();
    cout << " done in " << t_gen << " ms\n";

    cout << "Running HEM ..." << flush;
    t.tic();
    auto R = heavy_edge_matching(G, seed_match);
    double t_hem = t.toc_ms();
    cout << " done in " << t_hem << " ms\n";

    // Stats
    vector<char> seen(n, 0);
    int pairs = 0, singletons = 0;
    for (int u = 0; u < n; ++u) if (!seen[u]) {
        int v = R.match[u];
        if (v != -1) { pairs++; seen[u]=seen[v]=1; }
        else { singletons++; seen[u]=1; }
    }

    cout.setf(std::ios::fixed); cout<<setprecision(3);
    cout << "Input:  |V|=" << n << ", |E|=" << m << "\n";
    cout << "Match:  pairs=" << pairs << ", singletons=" << singletons 
         << ", matched_ratio=" << (2.0*pairs)/n << "\n";
    cout << "Coarse: |V'|=" << R.num_coarse << ", |E'|=" << R.edges.size() << "\n";

    // Print a few coarse edges
    cout << "Sample coarse edges (up to 10):\n";
    for (size_t i = 0; i < min<size_t>(10, R.edges.size()); ++i) {
        const auto& e = R.edges[i];
        cout << "  (" << e.u << ", " << e.v << ") w=" << e.w << "\n";
    }

    return 0;
}
