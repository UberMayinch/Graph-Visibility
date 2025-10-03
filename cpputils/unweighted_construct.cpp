#include <bits/stdc++.h>
using namespace std;
#include <chrono>
using namespace std::chrono;


void VisibilityGraphDQ(vector<double>& y, int l, int r, vector<vector<int>>&G){
    if(l >= r){
        return;
    }
    
    // Find the maximum element in range [l, r]
    double mx = -INFINITY;
    int idx = l;
    for(int i = l; i <= r; i++){
        if(y[i] > mx){
            mx = y[i];
            idx = i;
        }
    }
    
    // Scan left from the peak
    double min_slope = INFINITY;
    double slope;
    for(int i = idx-1; i >= l; i--){
        slope = (y[idx] - y[i]) / (idx - i);
        if(slope < min_slope){
            G[idx].push_back(i);
            G[i].push_back(idx);
            min_slope = slope;
        }
    }
    
    // Scan right from the peak
    min_slope = -INFINITY;
    for(int i = idx+1; i <= r; i++){  
        slope = (y[i] - y[idx]) / (i - idx);  // Fixed: slope calculation
        if(slope > min_slope){
            G[idx].push_back(i);
            G[i].push_back(idx);
            min_slope = slope;
        }
    }
    
    // Recursively process left and right parts
    VisibilityGraphDQ(y, l, idx-1, G);
    VisibilityGraphDQ(y, idx+1, r, G);
}

int main(int argc, char** argv)
{
auto start = high_resolution_clock::now();
    string data_dir = string(argv[1]);
    vector<string> files;
    
    // Get all output*.csv files
    for (const auto& entry : filesystem::directory_iterator(data_dir)) {
        string filename = entry.path().filename().string();
        if (filename.substr(0, 6) == "output" && filename.substr(filename.length() - 4) == ".csv") {
            files.push_back(entry.path().string());
        }
    }
    
    
    for (const string& file : files) {
        ifstream infile(file);
        string line;
        vector<double> t, y;
        
        // Skip header if present
        getline(infile, line);
        
        // Read data
        // Parse CSV: time,u,v - use v column for visibility graph
        while (getline(infile, line)) {
            stringstream ss(line);
            string time_str, u_str, v_str;
            
            if (getline(ss, time_str, ',') && 
                getline(ss, u_str, ',') && 
                getline(ss, v_str, ',')) {
                // Use the v column (third column)
                y.push_back(stof(v_str));
            }
        }
        infile.close();

        // Construct visibility graph and calculate metrics
        int n = y.size();
        vector<vector<int>> graph(n);
        VisibilityGraphDQ(y, 0, n-1, graph);

        // Write visibility graph to file format -> unweighted_graph_{init_conds}.csv

        string graph_filename = data_dir + "/unweighted_graph" + filesystem::path(file).filename().stem().string().substr(6) + ".csv";
        ofstream graph_file(graph_filename);
        graph_file << "node,neighbor" << endl;
        for (int i = 0; i < n; i++) {
            for (int neighbor : graph[i]) {
            if (i < neighbor) { // Only write each edge once
                graph_file << i << "," << neighbor << endl;
            }
            else break;
            }
        }
        graph_file.close();

    }

// compute time
auto end = high_resolution_clock::now();
cout << "Elapsed: " << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

// this whole script took 8335620 ms to run last time ~2 hours 18 min