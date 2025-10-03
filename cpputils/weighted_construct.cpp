#include <bits/stdc++.h>
using namespace std;
#include <chrono>
using namespace std::chrono;


void WeightedVisibilityGraphDQ(vector<double>& y, int l, int r, vector<vector<pair<int, double>>>&G){
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
        slope = abs((y[idx] - y[i]) / (idx - i));
        if(slope < min_slope){
            G[idx].push_back({i, slope});
            G[i].push_back({idx, slope});
            min_slope = slope;
        }
    }
    
    // Scan right from the peak
    min_slope = INFINITY;
    for(int i = idx+1; i <= r; i++){  
        slope = abs((y[i] - y[idx]) / (i - idx));
        if(slope < min_slope){
            G[idx].push_back({i, slope});
            G[i].push_back({idx, slope});
            min_slope = slope;
        }
    }
    
    // Recursively process left and right parts
    WeightedVisibilityGraphDQ(y, l, idx-1, G);
    WeightedVisibilityGraphDQ(y, idx+1, r, G);
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
        // run the swap uv script before processing so that v or y is in the second column
        // Makes reading much faster.
        while (getline(infile, line)) {
            stringstream ss(line);
            string time_str, u_str, v_str;
            //this is the original order, but v_str will come first after modification.
            
            if (getline(ss, v_str, ',')) {
                y.push_back(stof(v_str));
            }
        }
        infile.close();

        // Construct visibility graph and calculate metrics
        int n = y.size();
        vector<vector<pair<int, double>>> graph(n);
        WeightedVisibilityGraphDQ(y, 0, n-1, graph);

        // Write visibility graph to file
        string graph_filename = data_dir + "/weighted_graph" + filesystem::path(file).filename().stem().string().substr(6) + ".csv";
        ofstream graph_file(graph_filename);
        graph_file << "node,neighbor, weight" << endl;
        for (int i = 0; i < n; i++) {
            for (auto neighbor : graph[i]) {
            if (i < neighbor.first) { // Only write each edge once
                graph_file << i << "," << neighbor.first << "," << neighbor.second << endl;
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
