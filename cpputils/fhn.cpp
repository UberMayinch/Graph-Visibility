#include <bits/stdc++.h>
using namespace std;
const double pi = M_PI;


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
    string filename = output_directory + "/" + string(argv[2]) + "_" + string(argv[3]) + "/output_" + string(argv[1]) + ".csv";
    
    // Pre-allocate output buffer for better I/O performance
    vector<string> output_buffer;
    output_buffer.reserve(num_steps + 1);
    output_buffer.push_back("time,u,v");
    
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
        
        // Buffer output instead of writing immediately
        output_buffer.push_back(to_string(t) + "," + to_string(u0) + "," + to_string(v0));
    }
    
    // Write all output at once for better I/O performance
    ofstream outfile(filename);
    for (const string& line : output_buffer) {
        outfile << line << '\n';
    }
    outfile.close();

}