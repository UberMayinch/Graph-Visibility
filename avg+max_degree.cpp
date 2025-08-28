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
