#include <bits/stdc++.h>
using namespace std;
const double pi = M_PI;


pair<double, double> func(double x, double y, double t, unordered_map<string, double>& params){
    double dx_dt = y; 
     double dy_dt = -params["alpha"]*x * y+ -params["beta"] * x * x * x- params["gamma"]*x + params["f"]*sin(params["omega"]*t) ;

    return {dx_dt, dy_dt};
}

pair<double, double> adv(double u, double v, double t, unordered_map<string, double>& params, double h){
    auto k1 = func(u, v, t, params);
    auto k2 = func(u + k1.first * h /2 , v + k1.second * h/2, t + h/2, params);
    auto k3 = func(u + k2.first * h /2 , v + k2.second * h/2, t + h/2, params);
    auto k4 = func(u + k3.first * h , v + k3.second * h, t + h, params);

    double u_adv = u + h * (k1.first + 2*k2.first + 2 * k3.first + k4.first)/6;
    double v_adv = v + h * (k1.second + 2*k2.second + 2 * k3.second + k4.second)/6;

    return {u_adv, v_adv};
}

int main(int argc, char** argv){

    unordered_map<string, double> params;

    params["alpha"] = 0.45;
    params["beta"] = 0.5;
    params["gamma"] = -0.5; 
    params["f"] = 0.2; 

    if(argc < 4){
        cout << "Usage ./a.out <omega_value> <x0_value> <y0_value> [num_steps]" << endl;
        return 1;
    }
    params["omega"] = atof(argv[1]);
    
    double x0=atof(argv[2]);
    double y0=atof(argv[3]);

    // Default number of steps, can be overridden by command line argument
    int num_steps = 10000;
    if(argc >= 5){
        num_steps = atoi(argv[4]);
        if(num_steps <= 0){
            cout << "Error: num_steps must be positive" << endl;
            return 1;
        }
    }
    string output_directory = "data/linard";

    if (!filesystem::exists(output_directory)) {
        cerr << "Error: 'data/linard' directory does not exist" << endl;
        return 1;
    }


    string filename = output_directory + "/" + string(argv[2]) + "_" + string(argv[3]) + "/output_" + string(argv[1]) + ".csv";
    
    // Pre-allocate output buffer for better I/O performance
    vector<string> output_buffer;
    output_buffer.reserve(num_steps + 1);
    output_buffer.push_back("time,x,y");
    
    double t = 0.0;
    const double h = 0.01;  
    
    // Extract frequently used parameters to avoid map lookups
    const double alpha = params["alpha"];
    const double beta = params["beta"];
    const double gamma = params["gamma"];
    const double f = params["f"];
    const double omega = params["omega"];
    
    // Optimized integration loop
    for(int i = 0; i < num_steps; i++){
        // Direct calculation without function calls for better performance
        const double dx_dt = y0;
        const double dy_dt = -alpha * x0 * y0 - beta * x0 * x0 * x0 - gamma * x0 + f * sin(omega * t);
        
        // RK4 integration - optimized
        const double k1x = dx_dt;
        const double k1y = dy_dt;
        
        const double x_mid1 = x0 + k1x * h * 0.5;
        const double y_mid1 = y0 + k1y * h * 0.5;
        const double t_mid = t + h * 0.5;
        
        const double k2x = y_mid1;
        const double k2y = -alpha * x_mid1 * y_mid1 - beta * x_mid1 * x_mid1 * x_mid1 - gamma * x_mid1 + f * sin(omega * t_mid);
        
        const double x_mid2 = x0 + k2x * h * 0.5;
        const double y_mid2 = y0 + k2y * h * 0.5;
        
        const double k3x = y_mid2;
        const double k3y = -alpha * x_mid2 * y_mid2 - beta * x_mid2 * x_mid2 * x_mid2 - gamma * x_mid2 + f * sin(omega * t_mid);
        
        const double x_end = x0 + k3x * h;
        const double y_end = y0 + k3y * h;
        const double t_end = t + h;
        
        const double k4x = y_end;
        const double k4y = -alpha * x_end * y_end - beta * x_end * x_end * x_end - gamma * x_end + f * sin(omega * t_end);
        
        x0 += h * (k1x + 2*k2x + 2*k3x + k4x) / 6.0;
        y0 += h * (k1y + 2*k2y + 2*k3y + k4y) / 6.0;
        t += h;
        
        // Buffer output instead of writing immediately
        output_buffer.push_back(to_string(t) + "," + to_string(x0) + "," + to_string(y0));
    }
    
    // Write all output at once for better I/O performance
    ofstream outfile(filename);
    for (const string& line : output_buffer) {
        outfile << line << '\n';
    }
    outfile.close();
    
    return 0;

}