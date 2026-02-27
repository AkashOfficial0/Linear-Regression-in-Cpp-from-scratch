#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>
#include<matplotlibcpp.h>

using namespace std;
namespace plt = matplotlibcpp;

int main() {
    ifstream file("dataSet.csv");
    if(!file.is_open()){
        cout << "Error opening file";
        return 1;
    }
    string line;
    vector<double> x_values;
    vector<double> y_values;

    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string x_str, y_str;

        getline(ss, x_str, ',');
        getline(ss, y_str, ',');

        x_values.push_back(stod(x_str));
        y_values.push_back(stod(y_str));
    }

    file.close();

    int n = x_values.size();

    double m = 0;
    double c = 0;
    
    int epochs = 5000;
    double learning_rate = 0.0001;

    for(int epoch = 0; epoch < epochs; epoch++){
        double dm = 0;
        double dc = 0;
        for(int i=0; i<n; i++){
            double y_pred = m * x_values[i] + c;
            double error = y_pred - y_values[i];
            dm += (2.0/n) * x_values[i] * error;
            dc += (2.0/n) * error;
        }
        m -= learning_rate * dm;
        c -= learning_rate * dc;

        double total_loss = 0;
        for(int i = 0; i < n; i++){
            double y_pred = m * x_values[i] + c;
            total_loss += pow(y_pred - y_values[i], 2);
}
    total_loss /= n;

    if(epoch % 100 == 0){
        cout << "Epoch: " << epoch << " Loss: " << total_loss << endl;
}
    }
    cout << m << " " << c << endl;

    double min_x = *min_element(x_values.begin(), x_values.end());
    double max_x = *max_element(x_values.begin(), x_values.end());

    vector<double> x_line = {min_x, max_x};
    vector<double> y_line = {m * min_x + c, m * max_x + c};

    plt::scatter(x_values, y_values, 10.0);      // Original data
    plt::plot(x_line, y_line, "r-"); // Predicted regression line
    plt::show();

    return 0;
}
