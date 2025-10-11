#include <bits/stdc++.h>
#include <omp.h>
#include <cstdint>
using namespace std;

struct GraphMetrics {
    double entropy;
    double avg_path_length;
    vector<int> degree_sequence;
    int num_nodes;
    int num_edges;
    double density;
    double clustering_coefficient;
};

class GraphAnalyzer {
private:
    vector<vector<int>> adj_list;
    int n;
    
    // Calculate degree distribution entropy
    double calculateDegreeEntropy() {
        vector<int> degrees(n);
        map<int, int> degree_count;
        
        for (int i = 0; i < n; i++) {
            degrees[i] = adj_list[i].size();
            degree_count[degrees[i]]++;
        }
        
        double entropy = 0.0;
        for (const auto& pair : degree_count) {
            double p = (double)pair.second / n;
            if (p > 0) {
                entropy -= p * log2(p);
            }
        }
        
        return entropy;
    }
    
    // BFS for shortest path calculation
    vector<int> bfs(int start) {
        vector<int> dist(n, -1);
        queue<int> q;
        
        dist[start] = 0;
        q.push(start);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v : adj_list[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        
        return dist;
    }
    
    // Calculate average path length using optimized sampling
    double calculateAvgPathLength(int sample_size = 1000) {
        if (n <= 1) return 0.0;
        
        double total_path_length = 0.0;
        int valid_paths = 0;
        
        // For small graphs, compute exact average path length
        if (n <= 100) {
            for (int i = 0; i < n; i++) {
                vector<int> distances = bfs(i);
                for (int j = 0; j < n; j++) {
                    if (j != i && distances[j] != -1) {
                        total_path_length += distances[j];
                        valid_paths++;
                    }
                }
            }
        } else {
            // For large graphs, use parallel random sampling
            int actual_samples = min(sample_size, n / 4); // Reduced sampling for efficiency
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dis(0, n - 1);
            for (int i = 0; i < actual_samples; i++) {
                int start = dis(gen);
                vector<int> distances = bfs(start);
                for (int j = 0; j < n; j++) {
                    if (j != start && distances[j] != -1) {
                        total_path_length += distances[j];
                        valid_paths++;
                    }
                }
            }
        }
        
        return valid_paths > 0 ? total_path_length / valid_paths : 0.0;
    }
    
    // Calculate clustering coefficient - optimized and parallelized
    double calculateClusteringCoefficient() {
        double total_clustering = 0.0;
        int nodes_with_degree_gt_1 = 0;
        
    for (int i = 0; i < n; i++) {
            int degree = adj_list[i].size();
            if (degree < 2) continue;
            
            nodes_with_degree_gt_1++;
            
            // Count triangles involving node i - optimized with sorted adjacency
            int triangles = 0;
            const vector<int>& neighbors_i = adj_list[i];
            
            // Use sorted adjacency lists for faster intersection
            for (size_t j = 0; j < neighbors_i.size(); j++) {
                for (size_t k = j + 1; k < neighbors_i.size(); k++) {
                    int neighbor1 = neighbors_i[j];
                    int neighbor2 = neighbors_i[k];
                    
                    // Binary search for faster neighbor lookup (adjacency lists should be sorted)
                    const vector<int>& neighbors_1 = adj_list[neighbor1];
                    if (binary_search(neighbors_1.begin(), neighbors_1.end(), neighbor2)) {
                        triangles++;
                    }
                }
            }
            
            // Local clustering coefficient for node i
            double possible_edges = degree * (degree - 1) / 2.0;
            if (possible_edges > 0) {
                total_clustering += triangles / possible_edges;
            }
        }
        
        return nodes_with_degree_gt_1 > 0 ? total_clustering / nodes_with_degree_gt_1 : 0.0;
    }

public:
    GraphAnalyzer(const string& graph_file) {
        loadGraph(graph_file);
    }
    
    void loadGraph(const string& graph_file) {
        // Try binary formats first (UGB1/WGB1)
        ifstream bfile(graph_file, ios::binary);
        if (!bfile.is_open()) {
            throw runtime_error("Cannot open graph file: " + graph_file);
        }
        
        map<int, set<int>> temp_adj;
        int max_node = -1;
        
        char magic[4] = {0};
        bfile.read(magic, 4);
        if (bfile && (strncmp(magic, "UGB1", 4) == 0 || strncmp(magic, "WGB1", 4) == 0)) {
            // Binary graph: read edge_count and edges
            uint64_t edge_count = 0;
            bfile.read(reinterpret_cast<char*>(&edge_count), sizeof(edge_count));
            for (uint64_t i = 0; i < edge_count; ++i) {
                int32_t u, v;
                bfile.read(reinterpret_cast<char*>(&u), sizeof(u));
                bfile.read(reinterpret_cast<char*>(&v), sizeof(v));
                if (!bfile) break;
                // If WGB1, skip weight
                if (strncmp(magic, "WGB1", 4) == 0) {
                    double w;
                    bfile.read(reinterpret_cast<char*>(&w), sizeof(w));
                }
                temp_adj[u].insert(v);
                temp_adj[v].insert(u);
                max_node = max(max_node, max((int)u, (int)v));
            }
            bfile.close();
        } else {
            // Fallback to CSV reader
            bfile.close();
            ifstream file(graph_file);
            if (!file.is_open()) {
                throw runtime_error("Cannot open graph file: " + graph_file);
            }
            string line;
            getline(file, line); // Skip header
            while (getline(file, line)) {
                stringstream ss(line);
                string node1_str, node2_str, weight_str;
                if (!getline(ss, node1_str, ',') || !getline(ss, node2_str, ',')) {
                    continue;
                }
                getline(ss, weight_str);
                try {
                    int node1 = stoi(node1_str);
                    int node2 = stoi(node2_str);
                    temp_adj[node1].insert(node2);
                    temp_adj[node2].insert(node1);
                    max_node = max(max_node, max(node1, node2));
                } catch (const exception& e) {
                    cerr << "Error parsing line: " << line << " - " << e.what() << endl;
                    continue;
                }
            }
            file.close();
        }
        
        if (max_node == -1) {
            throw runtime_error("No valid edges found in graph file");
        }
        
        n = max_node + 1;
        adj_list.resize(n);
        
        // Build adjacency lists and sort them for binary search optimization
        for (const auto& pair : temp_adj) {
            int node = pair.first;
            adj_list[node].assign(pair.second.begin(), pair.second.end());
            // Lists are already sorted since we used set<int>
        }
    }
    
    GraphMetrics calculateMetrics() {
        GraphMetrics metrics;
        
        metrics.num_nodes = n;
        metrics.num_edges = 0;
        metrics.degree_sequence.resize(n);
        
        // Calculate basic metrics
        for (int i = 0; i < n; i++) {
            metrics.degree_sequence[i] = adj_list[i].size();
            metrics.num_edges += adj_list[i].size();
        }
        metrics.num_edges /= 2; // Each edge counted twice
        
        metrics.density = n > 1 ? (2.0 * metrics.num_edges) / (n * (n - 1)) : 0.0;
        
        // Calculate advanced metrics
        metrics.entropy = calculateDegreeEntropy();
        metrics.avg_path_length = calculateAvgPathLength();
        metrics.clustering_coefficient = calculateClusteringCoefficient();
        
        return metrics;
    }
    
    void saveMetrics(const GraphMetrics& metrics, const string& output_file) {
        // Save as MET1 binary: magic(4), uint32 item_count, then items: uint16 key_len, key bytes, double value
        ofstream file(output_file, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Cannot create output file: " + output_file);
        }
        const char magic[4] = {'M','E','T','1'};
        file.write(magic, 4);
        
        // Prepare key-value pairs
        vector<pair<string, double>> items;
        items.push_back({"num_nodes", (double)metrics.num_nodes});
        items.push_back({"num_edges", (double)metrics.num_edges});
        items.push_back({"density", metrics.density});
        items.push_back({"degree_entropy", metrics.entropy});
        items.push_back({"avg_path_length", metrics.avg_path_length});
        items.push_back({"clustering_coefficient", metrics.clustering_coefficient});
        if (!metrics.degree_sequence.empty()) {
            vector<int> degrees = metrics.degree_sequence;
            sort(degrees.begin(), degrees.end());
            double mean_degree = accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
            double median_degree = degrees.size() % 2 == 0 ? 
                (degrees[degrees.size()/2 - 1] + degrees[degrees.size()/2]) / 2.0 :
                degrees[degrees.size()/2];
            items.push_back({"mean_degree", mean_degree});
            items.push_back({"median_degree", median_degree});
            items.push_back({"min_degree", (double)degrees.front()});
            items.push_back({"max_degree", (double)degrees.back()});
        }
        
        uint32_t item_count = static_cast<uint32_t>(items.size());
        file.write(reinterpret_cast<const char*>(&item_count), sizeof(item_count));
        for (const auto& kv : items) {
            const string& key = kv.first;
            double value = kv.second;
            uint16_t key_len = static_cast<uint16_t>(key.size());
            file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
            file.write(key.data(), key.size());
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        file.close();
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <graph_file> <output_file> [sample_size]" << endl;
        cout << "Calculate comprehensive graph metrics including entropy and path lengths" << endl;
        return 1;
    }
    
    string graph_file = argv[1];
    string output_file = argv[2];
    
    try {
        cout << "Loading graph from: " << graph_file << endl;
        
        GraphAnalyzer analyzer(graph_file);
        
        cout << "Calculating metrics..." << endl;
        GraphMetrics metrics = analyzer.calculateMetrics();
        
        cout << "Graph Analysis Results:" << endl;
        cout << "  Nodes: " << metrics.num_nodes << endl;
        cout << "  Edges: " << metrics.num_edges << endl;
        cout << "  Density: " << metrics.density << endl;
        cout << "  Degree Entropy: " << metrics.entropy << endl;
        cout << "  Avg Path Length: " << metrics.avg_path_length << endl;
        cout << "  Clustering Coefficient: " << metrics.clustering_coefficient << endl;
        
    analyzer.saveMetrics(metrics, output_file);
    cout << "Metrics saved to: " << output_file << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
