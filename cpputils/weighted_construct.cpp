#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
#include <cstdint>
using namespace std;
using namespace std::chrono;


void WeightedVisibilityGraphDQ(vector<double>& y, int l, int r, vector<vector<pair<int, double>>>&G){
    // Use iterative approach with explicit stack to avoid stack overflow
    struct Range { int l, r; };
    stack<Range> ranges;
    ranges.push({l, r});
    
    while (!ranges.empty()) {
        Range current = ranges.top();
        ranges.pop();
        
        if (current.l >= current.r) {
            continue;
        }
        
        // Find the maximum element in range [l, r] - optimized for vectorization
        double mx = -INFINITY;
        int idx = current.l;
        
        // Use restrict and alignment hints for better optimization
        const double* __restrict__ y_data = y.data();
        
        // Find maximum without SIMD reduction due to index tracking complexity
        for(int i = current.l; i <= current.r; i++){
            if(y_data[i] > mx){
                mx = y_data[i];
                idx = i;
            }
        }
        
        // Scan left from the peak
        double min_slope = INFINITY;
        double slope;
        for(int i = idx-1; i >= current.l; i--){
            slope = abs((y[idx] - y[i]) / (idx - i));
            if(slope < min_slope){
                G[idx].push_back({i, slope});
                G[i].push_back({idx, slope});
                min_slope = slope;
            }
        }
        
        // Scan right from the peak
        min_slope = INFINITY;
        for(int i = idx+1; i <= current.r; i++){  
            slope = abs((y[i] - y[idx]) / (i - idx));
            if(slope < min_slope){
                G[idx].push_back({i, slope});
                G[i].push_back({idx, slope});
                min_slope = slope;
            }
        }
        
        // Add child ranges to stack instead of recursive calls
        if (current.l < idx - 1) {
            ranges.push({current.l, idx - 1});
        }
        if (idx + 1 < current.r) {
            ranges.push({idx + 1, current.r});
        }
    }
}


int main(int argc __attribute__((unused)), char** argv)
{
    auto start = high_resolution_clock::now();
    string data_dir = string(argv[1]);
    vector<string> files;
    
    // Get all output*.bin files
    files.reserve(1000); // Pre-allocate to avoid reallocations
    for (const auto& entry : filesystem::directory_iterator(data_dir)) {
        const string& filename = entry.path().filename().string();
        if (filename.size() >= 10 && filename.substr(0, 6) == "output" && 
            filename.substr(filename.length() - 4) == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    
    
    // Limit parallelism to prevent memory explosion and thread thrashing
    int max_threads = min(4, (int)files.size());  // Max 4 threads for memory safety
    omp_set_num_threads(max_threads);
    cout << "Processing " << files.size() << " files using " << max_threads << " threads" << endl;
    
    // Process files in parallel with controlled thread count
    #pragma omp parallel for schedule(dynamic)
    for (size_t file_idx = 0; file_idx < files.size(); ++file_idx) {
        const string& file = files[file_idx];
        
        // Read TSB1 binary: magic(4), uint32 cols, uint32 rows, then rows of 3 x float64
        ifstream infile(file, ios::binary);
        if (!infile) continue;
        
        char magic[4];
        infile.read(magic, 4);
        if (strncmp(magic, "TSB1", 4) != 0) { infile.close(); continue; }
        uint32_t cols = 0, rows = 0;
        infile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        infile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (cols < 3) { infile.close(); continue; }
        
        vector<double> y;
        y.reserve(min<size_t>(rows, 200000));
        const size_t MAX_SERIES_SIZE = 200000;
        
        for (uint32_t i = 0; i < rows; ++i) {
            double t_val, u_val, v_val;
            infile.read(reinterpret_cast<char*>(&t_val), sizeof(t_val));
            infile.read(reinterpret_cast<char*>(&u_val), sizeof(u_val));
            infile.read(reinterpret_cast<char*>(&v_val), sizeof(v_val));
            if (!infile) break;
            y.push_back(v_val);
            if (y.size() >= MAX_SERIES_SIZE) {
                cout << "Warning: Truncating time series at " << MAX_SERIES_SIZE << " points for memory safety" << endl;
                break;
            }
        }
        infile.close();

        if (y.empty()) continue;

        // Construct visibility graph
        int n = y.size();
        vector<vector<pair<int, double>>> graph(n);
        WeightedVisibilityGraphDQ(y, 0, n-1, graph);

        // Write visibility graph to binary file (WGB1)
        string graph_filename = data_dir + "/weighted_graph" + 
                               filesystem::path(file).filename().stem().string().substr(6) + ".bin";
        
        ofstream graph_file(graph_filename, ios::binary);
        if (!graph_file.is_open()) continue;
        const char gmagic[4] = {'W','G','B','1'};
        graph_file.write(gmagic, 4);
        // Count edges (each undirected edge once where i < j)
        uint64_t edge_count = 0;
        for (int i = 0; i < n; i++) {
            for (const auto& neighbor : graph[i]) {
                if (i < neighbor.first) edge_count++;
                else break;
            }
        }
        graph_file.write(reinterpret_cast<const char*>(&edge_count), sizeof(edge_count));
        // Write edges
        for (int i = 0; i < n; i++) {
            for (const auto& neighbor : graph[i]) {
                if (i < neighbor.first) {
                    int32_t a = i;
                    int32_t b = neighbor.first;
                    double w = neighbor.second;
                    graph_file.write(reinterpret_cast<const char*>(&a), sizeof(a));
                    graph_file.write(reinterpret_cast<const char*>(&b), sizeof(b));
                    graph_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
                } else {
                    break;
                }
            }
        }
        graph_file.close();
    }

    // Compute time
    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();
    cout << "Processed " << files.size() << " files in " << elapsed_ms << " ms" << endl;
    cout << "Average: " << (files.empty() ? 0 : elapsed_ms / files.size()) << " ms per file" << endl;
    
    return 0;
}
