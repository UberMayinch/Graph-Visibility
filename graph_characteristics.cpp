#include <bits/stdc++.h>
using namespace std;
#include <chrono>
using namespace std::chrono;


unsigned long MaxDegree(vector<vector<int>>&G)
{
    unsigned long mx = 0;
    for (auto node : G)
    {
        mx = max(mx, node.size());
    }
    return mx;
}

double ClusteringCoeff(vector<vector<int>>& G)
{
    double c_glob = 0;
    for (auto node : G)
    {
        double edge_tot = 0;
        for (auto edge : node)
        {
            edge_tot += (double)(G[edge].size() / G.size());
        }
        double c_loc = edge_tot / node.size();
        c_glob += c_loc / G.size();
    }
    return c_glob;
}

double AvgDegree(vector<vector<int>>& G)
{
    double avg = 0;
    for (auto node : G)
    {
        avg += node.size();
    }
    avg /= G.size();
    return avg;
}

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
        vector<vector<int>> graph(n);
        VisibilityGraphDQ(y, 0, n-1, graph);

        // Write visibility graph to file
        string graph_filename = data_dir + "/graph" + filesystem::path(file).filename().stem().string().substr(6) + ".csv";
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

        // Extract parameter value from filename (assuming format like "output_param.csv")
        string filename = filesystem::path(file).filename().string();
        string param = filename.substr(7, filename.length() - 11); // Remove "output_" and ".csv"
        
        // Write to CSV (open in append mode)
        ofstream csvfile(data_dir + "graph_metrics.csv", ios::app);
        if (csvfile.tellp() == 0) {
            // Write header if file is empty
            csvfile << "parameter,max_degree,avg_degree" << endl;
        }
        csvfile << param << "," << MaxDegree(graph) << "," << AvgDegree(graph) << endl;
        csvfile.close();
    }

// compute time
auto end = high_resolution_clock::now();
cout << "Elapsed: " << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

// this whole script took 8335620 ms to run last time ~2 hours 18 min