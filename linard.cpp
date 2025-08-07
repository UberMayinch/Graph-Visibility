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

    if(argc < 2){
        cout << "Usage ./a.out <omega_value> <x0_value> <y0_value>" << endl;
    }
    params["omega"] = atof(argv[1]);
    
    double x0=atof(argv[2]);
    double y0=atof(argv[3]);

    int num_steps=100000;
    string output_directory = "data/linard";

    if (!filesystem::exists(output_directory)) {
        cerr << "Error: 'data/linard' directory does not exist" << endl;
        return 1;
    }

    ofstream outfile(output_directory + "/output_" + string(argv[1]) + "_" + string(argv[2]) + "_" + string(argv[3]) + ".csv");
    outfile << "time,x,y" << endl;  
    double t = 0.0;
    double h = 0.01;  
    
    for(int i = 0; i < num_steps; i++){
        outfile << t << "," << x0 << "," << y0 << endl;
        auto result = adv(x0, y0, t, params, h);
        x0 = result.first;
        y0 = result.second;
        t += h;
    }
    
    outfile.close();

}