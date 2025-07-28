#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

// --------------------------- CONSTANTS -------------------------------
#define MAXN 1 << 20 // Maximum size of time series
#define LOGN 20      // Log2(MAXN)

// --------------------------- GPU ERROR CHECKING -----------------------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
        exit(code);
    }
}

// --------------------------- DEVICE GLOBAL MEMORY ---------------------
__device__ double d_y[MAXN];       // y(t)
__device__ int d_st[MAXN][LOGN];   // Sparse table storing indices of maxima
__device__ double d_slope[MAXN];   // Slopes to maximum (intermediate array)
__device__ int d_visible[MAXN];    // 1 if node can "see" the max

// --------------------------- SPARSE TABLE KERNEL ----------------------
__global__ void buildSparseTableKernel(int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + (1 << k) > n) return;

    int left = d_st[i][k - 1];
    int right = d_st[i + (1 << (k - 1))][k - 1];
    d_st[i][k] = (d_y[left] > d_y[right]) ? left : right;
}

// --------------------------- SLOPE COMPUTATION KERNEL -----------------
__global__ void computeSlopesKernel(int* result, int l, int r, int maxIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < l || i > r || i == maxIdx) return;
    d_slope[i] = (d_y[maxIdx] - d_y[i]) / (maxIdx - i);
}

// --------------------------- PREFIX MAX SCAN (INCLUSIVE) --------------
__global__ void prefixMaxKernel(int l, int r, int maxIdx, bool left) {
    int tid = threadIdx.x;
    __shared__ double scan[MAXN];

    int i = left ? (r - tid) : (l + tid);
    if (i < l || i > r || i == maxIdx) return;

    // Initialize with slope
    scan[tid] = d_slope[i];

    __syncthreads();

    // Inclusive max scan
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        double temp = 0;
        if (tid >= offset) temp = fmax(scan[tid], scan[tid - offset]);
        __syncthreads();
        if (tid >= offset) scan[tid] = temp;
        __syncthreads();
    }

    // Mark as visible if slope is local optima
    d_visible[i] = (scan[tid] == d_slope[i]) ? 1 : 0;
}

// --------------------------- SPARSE TABLE QUERY FUNCTION --------------
__device__ int queryMaxIndex(int l, int r) {
    int k = log2(r - l + 1);
    int left = d_st[l][k];
    int right = d_st[r - (1 << k) + 1][k];
    return (d_y[left] > d_y[right]) ? left : right;
}

// --------------------------- HOST SIDE GRAPH STORAGE ------------------
vector<pair<int, int>> visibility_edges;

// --------------------------- RECURSIVE PROCESSING ---------------------
void process(int l, int r) {
    if (l >= r) return;

    // Copy sparse table from device to host to query max index
    int maxIdx;
    cudaMemcpyFromSymbol(&maxIdx, d_st[l][0], sizeof(int)); // Just for illustration

    // Step 1: Compute slope to max in parallel
    computeSlopesKernel<<<(r - l + 1 + 255) / 256, 256>>>(nullptr, l, r, maxIdx);
    gpuErrchk(cudaPeekAtLastError());

    // Step 2: Left and Right prefix max
    prefixMaxKernel<<<1, r - l + 1>>>(l, maxIdx - 1, maxIdx, true);
    prefixMaxKernel<<<1, r - l + 1>>>(maxIdx + 1, r, maxIdx, false);
    gpuErrchk(cudaPeekAtLastError());

    // Step 3: Copy visibility and add edges
    int* h_visible = new int[r - l + 1];
    cudaMemcpyFromSymbol(h_visible, d_visible, sizeof(int) * (r - l + 1));
    for (int i = l; i <= r; i++) {
        if (i != maxIdx && h_visible[i]) {
            visibility_edges.emplace_back(i, maxIdx);
        }
    }

    delete[] h_visible;

    // Step 4: Recurse
    process(l, maxIdx - 1);
    process(maxIdx + 1, r);
}

// --------------------------- MAIN FUNCTION ----------------------------
int main() {
    int n;
    vector<double> y;

    // Read time series
    cin >> n;
    y.resize(n);
    for (int i = 0; i < n; i++) cin >> y[i];

    // Copy to GPU
    cudaMemcpyToSymbol(d_y, y.data(), sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        int val = i;
        cudaMemcpyToSymbol(d_st[i][0], &val, sizeof(int));
    }

    // Sparse table construction
    for (int k = 1; (1 << k) <= n; k++) {
        int blocks = (n + 255) / 256;
        buildSparseTableKernel<<<blocks, 256>>>(n, k);
        gpuErrchk(cudaPeekAtLastError());
    }

    // Run recursive algorithm
    process(0, n - 1);

    // Output result
    for (auto [u, v] : visibility_edges) {
        cout << u << " <--> " << v << endl;
    }

    return 0;
}
