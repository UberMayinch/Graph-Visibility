#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
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
    
    // Get all output*.csv files
    files.reserve(1000); // Pre-allocate to avoid reallocations
    for (const auto& entry : filesystem::directory_iterator(data_dir)) {
        const string& filename = entry.path().filename().string();
        if (filename.size() >= 10 && filename.substr(0, 6) == "output" && 
            filename.substr(filename.length() - 4) == ".csv") {
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
        
        // Fast CSV reading with memory optimization
        ifstream infile(file);
        if (!infile) continue;
        
        vector<double> y;
        y.reserve(200000); // Reduced size to prevent memory explosion
        
        // Memory safety check
        const size_t MAX_SERIES_SIZE = 200000;  // Limit time series size
        
        string line;
        line.reserve(64); // Reserve space for line string
        
        // Skip header if present
        getline(infile, line);
        
        // Parse CSV: time,u,v - use v column (third column)
        while (getline(infile, line)) {
            size_t first_comma = line.find(',');
            if (first_comma == string::npos) continue;
            
            size_t second_comma = line.find(',', first_comma + 1);
            if (second_comma == string::npos) continue;
            
            // Use the v column (third column) - faster parsing
            const char* v_start = line.c_str() + second_comma + 1;
            char* endptr;
            double v_val = strtod(v_start, &endptr);
            if (endptr != v_start) {
                y.push_back(v_val);
                // Memory safety: prevent excessive memory usage
                if (y.size() >= MAX_SERIES_SIZE) {
                    cout << "Warning: Truncating time series at " << MAX_SERIES_SIZE << " points for memory safety" << endl;
                    break;
                }
            }
        }
        infile.close();

        if (y.empty()) continue;

        // Construct visibility graph
        int n = y.size();
        vector<vector<pair<int, double>>> graph(n);
        WeightedVisibilityGraphDQ(y, 0, n-1, graph);

        // Write visibility graph to file - optimized I/O
        string graph_filename = data_dir + "/weighted_graph" + 
                               filesystem::path(file).filename().stem().string().substr(6) + ".csv";
        
        ofstream graph_file(graph_filename);
        graph_file.rdbuf()->pubsetbuf(nullptr, 0); // Disable buffering for immediate write
        graph_file << "node,neighbor,weight\n";
        
        // Write edges more efficiently
        for (int i = 0; i < n; i++) {
            for (const auto& neighbor : graph[i]) {
                if (i < neighbor.first) { // Only write each edge once
                    graph_file << i << ',' << neighbor.first << ',' << neighbor.second << '\n';
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
