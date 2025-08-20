#include <bits/stdc++.h>
using namespace std;
const double pi = M_PI;


pair<double, double> func(double u, double v, double t, unordered_map<string, double>& params){
    double du_dt = u - params["gamma"] * sinh(params["rho"]*u) - v + params["A"] * sin(2 * pi * params["omega"] * t);
     double dv_dt = params["delta"]*(u + params["a"] - params["b"]*v);

    return {du_dt, dv_dt};
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

    params["a"] = 0.7;
    params["b"] = 0.8;
    params["delta"] = 0.1;
    params["gamma"] = 0.0001 * 2.682; 
    params["omega"] = 0.17;
    params["rho"] = 4.0485;

    if(argc < 4){
        cout << "Usage ./a.out <A_value> <u0_value> <v0_value>" << endl;
    }
    params["A"] = atof(argv[1]);
    double u0=atof(argv[2]);
    double v0=atof(argv[3]);

    int num_steps=10000;
    string output_directory = "data/fhn";

    if (!filesystem::exists(output_directory)){
        cerr << "Error: 'data' directory does not exist" << endl;
        return 1;
    }
    ofstream outfile(output_directory + "/" + string(argv[2]) + "_" + string(argv[3]) + "_output_" + string(argv[1]) + ".csv");
    outfile << "time,u,v" << endl;  
    double t = 0.0;
    double h = 0.01;  
    
    for(int i = 0; i < num_steps; i++){
        outfile << t << "," << u0 << "," << v0 << endl;
        auto result = adv(u0, v0, t, params, h);
        u0 = result.first;
        v0 = result.second;
        t += h;
    }
    
    outfile.close();

}