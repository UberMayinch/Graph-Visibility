#include <bits/stdc++.h>
#include <cstdint>
using namespace std;
const double pi = M_PI;

// Global time series buffers (reserved in main)
static vector<double> G_TIME;
static vector<double> G_U;
static vector<double> G_V;

static const uint32_t DEFAULT_RESERVE_STEPS = 200000; // pre-allocate on program start
struct _GlobalInit {
    _GlobalInit() {
        G_TIME.reserve(DEFAULT_RESERVE_STEPS);
        G_U.reserve(DEFAULT_RESERVE_STEPS);
        G_V.reserve(DEFAULT_RESERVE_STEPS);
    }
} _globalInit;


// moved to main to reduce function call overhead
// pair<double, double> func(double u, double v, double t, unordered_map<string, double>& params){
//     double du_dt = u - params["gamma"] * sinh(params["rho"]*u) - v + params["A"] * sin(2 * pi * params["omega"] * t);
//      double dv_dt = params["delta"]*(u + params["a"] - params["b"]*v);

//     return {du_dt, dv_dt};
// }

// pair<double, double> adv(double u, double v, double t, unordered_map<string, double>& params, double h){
//     auto k1 = func(u, v, t, params);
//     auto k2 = func(u + k1.first * h /2 , v + k1.second * h/2, t + h/2, params);
//     auto k3 = func(u + k2.first * h /2 , v + k2.second * h/2, t + h/2, params);
//     auto k4 = func(u + k3.first * h , v + k3.second * h, t + h, params);

//     double u_adv = u + h * (k1.first + 2*k2.first + 2 * k3.first + k4.first)/6;
//     double v_adv = v + h * (k1.second + 2*k2.second + 2 * k3.second + k4.second)/6;

//     return {u_adv, v_adv};
// }

int main(int argc, char** argv){

    unordered_map<string, double> params;

    params["a"] = 0.7;
    params["b"] = 0.8;
    params["delta"] = 0.1;
    params["gamma"] = 0.0001 * 2.682; 
    params["omega"] = 0.17;
    params["rho"] = 4.0485;

    if(argc < 4){
        cout << "Usage ./a.out <A_value> <u0_value> <v0_value> [num_steps]" << endl;
        return 1;
    }
    params["A"] = atof(argv[1]);
    double u0=atof(argv[2]);
    double v0=atof(argv[3]);

    // Default number of steps, can be overridden by command line argument
    int num_steps = 200000;
    if(argc >= 5){
        num_steps = atoi(argv[4]);
        if(num_steps <= 0){
            cout << "Error: num_steps must be positive" << endl;
            return 1;
        }
    }
    string output_directory = "data/fhn";

    if (!filesystem::exists(output_directory)){
        cerr << "Error: 'data' directory does not exist" << endl;
        return 1;
    }
    string filename = output_directory + "/" + string(argv[2]) + "_" + string(argv[3]) + "/output_" + string(argv[1]) + ".bin";
    
    // Reset global buffers (capacity is already reserved at program start)
    G_TIME.clear(); G_U.clear(); G_V.clear();
    
    double t = 0.0;
    const double h = 0.01;  
    
    // Extract frequently used parameters to avoid map lookups
    const double a = params["a"];
    const double b = params["b"];
    const double delta = params["delta"];
    const double gamma = params["gamma"];
    const double omega = params["omega"];
    const double rho = params["rho"];
    const double A = params["A"];
    
    // Optimized integration loop
    for(int i = 0; i < num_steps; i++){
        // Direct calculation without function calls for better performance
        const double sin_term = A * sin(2 * pi * omega * t);
        const double du_dt = u0 - gamma * sinh(rho * u0) - v0 + sin_term;
        const double dv_dt = delta * (u0 + a - b * v0);
        
        // RK4 integration - optimized
        const double k1u = du_dt;
        const double k1v = dv_dt;
        
        const double u_mid1 = u0 + k1u * h * 0.5;
        const double v_mid1 = v0 + k1v * h * 0.5;
        const double t_mid = t + h * 0.5;
        const double sin_mid = A * sin(2 * pi * omega * t_mid);
        
        const double k2u = u_mid1 - gamma * sinh(rho * u_mid1) - v_mid1 + sin_mid;
        const double k2v = delta * (u_mid1 + a - b * v_mid1);
        
        const double u_mid2 = u0 + k2u * h * 0.5;
        const double v_mid2 = v0 + k2v * h * 0.5;
        
        const double k3u = u_mid2 - gamma * sinh(rho * u_mid2) - v_mid2 + sin_mid;
        const double k3v = delta * (u_mid2 + a - b * v_mid2);
        
        const double u_end = u0 + k3u * h;
        const double v_end = v0 + k3v * h;
        const double t_end = t + h;
        const double sin_end = A * sin(2 * pi * omega * t_end);
        
        const double k4u = u_end - gamma * sinh(rho * u_end) - v_end + sin_end;
        const double k4v = delta * (u_end + a - b * v_end);
        
        u0 += h * (k1u + 2*k2u + 2*k3u + k4u) / 6.0;
        v0 += h * (k1v + 2*k2v + 2*k3v + k4v) / 6.0;
        t += h;
        
        // Append to global buffers
        G_TIME.push_back(t);
        G_U.push_back(u0);
        G_V.push_back(v0);
    }
    
    // Write all buffered rows to binary TSB1 file
    ofstream outfile(filename, ios::binary);
    if (!outfile.is_open()) {
        cerr << "Error: Cannot open output file for writing: " << filename << endl;
        return 1;
    }
    const char magic[4] = {'T','S','B','1'};
    outfile.write(magic, 4);
    uint32_t cols = 3;
    uint32_t rows = static_cast<uint32_t>(G_TIME.size());
    outfile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    outfile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    for (uint32_t i = 0; i < rows; ++i) {
        const double tval = G_TIME[i];
        const double uval = G_U[i];
        const double vval = G_V[i];
        outfile.write(reinterpret_cast<const char*>(&tval), sizeof(tval));
        outfile.write(reinterpret_cast<const char*>(&uval), sizeof(uval));
        outfile.write(reinterpret_cast<const char*>(&vval), sizeof(vval));
    }
    outfile.close();

}