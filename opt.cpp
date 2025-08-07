#include <bits/stdc++.h>
using namespace std;

vector<vector<int>> VisibilityGraphNaive(vector<double>&y, vector<double>&t)
{
    int n = y.size();
    vector<vector<int>> G(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double max_slope = -INFINITY;
        double slope = 0;
        for (int j = i + 1; j < n; j++)
        {
            slope = (y[j] - y[i]) / (j - i);
            if (slope > max_slope)
            {
                #pragma omp critical
                {
                    G[i].push_back(j);
                    G[j].push_back(i);
                }
                max_slope = slope; 
            }
        }
    }
    return G;
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

int main(){
    // Test with smaller dataset first
    int n = 10000000;
    vector<double> y(n), t(n);
    srand(42); // Fixed seed for reproducible results
    
    for(int i = 0; i < n; i++){
        t[i] = i;
        y[i] = rand() % 100000 + 1;
    }
    
    // Print first few values for debugging
    cout << "First 10 y values: ";
    for(int i = 0; i < min(10, n); i++){
        cout << y[i] << " ";
    }
    cout << endl;
    
    // // Benchmark naive approach
    // auto start = chrono::high_resolution_clock::now();
    // vector<vector<int>> G1 = VisibilityGraphNaive(y, t);
    // auto end = chrono::high_resolution_clock::now();
    // auto naive_time = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // Benchmark divide and conquer approach
    vector<vector<int>> G2(n);
    auto start = chrono::high_resolution_clock::now();
    VisibilityGraphDQ(y, 0, n-1, G2);  // Fixed: should be n-1, not n
    auto end = chrono::high_resolution_clock::now();
    auto dq_time = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // // Compare graphs
    // bool graphs_identical = true;
    // for(int i = 0; i < n; i++){
    //     set<int> s1(G1[i].begin(), G1[i].end());
    //     set<int> s2(G2[i].begin(), G2[i].end());
    //     if(s1 != s2){
    //         graphs_identical = false;
    //         cout << "Difference at node " << i << ":" << endl;
    //         cout << "Naive: ";
    //         for(int x : s1) cout << x << " ";
    //         cout << "\nDQ: ";
    //         for(int x : s2) cout << x << " ";
    //         cout << endl;
    //         break;
    //     }
    // }
    
    // cout << "Graphs are " << (graphs_identical ? "identical" : "different") << endl;
    // cout << "Naive approach time: " << naive_time.count() << " ms" << endl;
    cout << "Divide & Conquer time: " << dq_time.count() << " ms" << endl;
    
    return 0;
}