#include <bits/stdc++.h>
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
    
    // Calculate average path length using random sampling
    double calculateAvgPathLength(int sample_size = 1000) {
        if (n <= 1) return 0.0;
        
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n - 1);
        
        double total_path_length = 0.0;
        int valid_paths = 0;
        
        // Use random sampling for efficiency on large graphs
        int actual_samples = min(sample_size, n * (n - 1) / 2);
        
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
        
        return valid_paths > 0 ? total_path_length / valid_paths : 0.0;
    }
    
    // Calculate clustering coefficient
    double calculateClusteringCoefficient() {
        double total_clustering = 0.0;
        int nodes_with_degree_gt_1 = 0;
        
        for (int i = 0; i < n; i++) {
            int degree = adj_list[i].size();
            if (degree < 2) continue;
            
            nodes_with_degree_gt_1++;
            
            // Count triangles involving node i
            int triangles = 0;
            set<int> neighbors(adj_list[i].begin(), adj_list[i].end());
            
            for (int j = 0; j < adj_list[i].size(); j++) {
                for (int k = j + 1; k < adj_list[i].size(); k++) {
                    int neighbor1 = adj_list[i][j];
                    int neighbor2 = adj_list[i][k];
                    
                    // Check if neighbor1 and neighbor2 are connected
                    if (find(adj_list[neighbor1].begin(), adj_list[neighbor1].end(), neighbor2) 
                        != adj_list[neighbor1].end()) {
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
        ifstream file(graph_file);
        if (!file.is_open()) {
            throw runtime_error("Cannot open graph file: " + graph_file);
        }
        
        string line;
        getline(file, line); // Skip header
        
        map<int, set<int>> temp_adj;
        int max_node = -1;
        
        while (getline(file, line)) {
            stringstream ss(line);
            string node1_str, node2_str, weight_str;
            
            if (!getline(ss, node1_str, ',') || 
                !getline(ss, node2_str, ',')) {
                continue;
            }
            
            // Weight column is optional (for unweighted graphs)
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
        
        if (max_node == -1) {
            throw runtime_error("No valid edges found in graph file");
        }
        
        n = max_node + 1;
        adj_list.resize(n);
        
        for (const auto& pair : temp_adj) {
            int node = pair.first;
            for (int neighbor : pair.second) {
                adj_list[node].push_back(neighbor);
            }
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
        ofstream file(output_file);
        if (!file.is_open()) {
            throw runtime_error("Cannot create output file: " + output_file);
        }
        
        file << "metric,value" << endl;
        file << "num_nodes," << metrics.num_nodes << endl;
        file << "num_edges," << metrics.num_edges << endl;
        file << "density," << metrics.density << endl;
        file << "degree_entropy," << metrics.entropy << endl;
        file << "avg_path_length," << metrics.avg_path_length << endl;
        file << "clustering_coefficient," << metrics.clustering_coefficient << endl;
        
        // Degree statistics
        if (!metrics.degree_sequence.empty()) {
            vector<int> degrees = metrics.degree_sequence;
            sort(degrees.begin(), degrees.end());
            
            double mean_degree = accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
            double median_degree = degrees.size() % 2 == 0 ? 
                (degrees[degrees.size()/2 - 1] + degrees[degrees.size()/2]) / 2.0 :
                degrees[degrees.size()/2];
            
            file << "mean_degree," << mean_degree << endl;
            file << "median_degree," << median_degree << endl;
            file << "min_degree," << degrees[0] << endl;
            file << "max_degree," << degrees.back() << endl;
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
