//
//  main.cpp
//  Biostat615_SGD
//
//  Created by chenkf on 12/7/16.
//  Copyright © 2016 chenkf. All rights reserved.
//

#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void update(VectorXd &theta, VectorXd &theta_new, MatrixXd &X, VectorXd &Y, double rate, int j);

int main(int argc, const char * argv[]) {
    ifstream myfile;
    string file;
    double diff_value;
    double rate;            // learning rate, or step size
    double tol = 1e-6;      // tolerence, or stopping criterion
    bool flag = 1;
    int N, M;               // input data dimension: N row * M column
    int sliding_window = 10;// size of sliding window
    int max_iter = 1000;    // maximum number of iterations
    int num = 0;            // number of iterations
    
    // arguments
    file = argv[1];
    N = atoi(argv[2]);
    M = atoi(argv[3]);
    if (argc==5) tol = atof(argv[4]);
    rate = 10.0 / N;
    
    myfile.open(file.c_str());
    while(!myfile.is_open()) {
        myfile.clear();
        cout << "Incorrect filename: " << file << endl;
        abort();
    }
    
    // Create variables
    MatrixXd X(N,M-1);
    VectorXd Y(N);
    VectorXd theta(M-1);    // estimated coefficients
    VectorXd theta_new(M-1);
    VectorXd P(N);          // probabilities
    VectorXd SE(M-1);       // standard errors of estimated coefficients
    vector<double> diff;    // vector of differences of updated and previous theta
    
    // read data into X and Y
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++ ) {
            if (j==0) myfile >> Y(i);
            else myfile >> X(i,j-1);
        }
    }
    
    // Coordinate Descent
    while (flag) {
        num++;
        if(num >= max_iter) {
            cerr << "Algorithm cannot converge within " << max_iter << " iterations" << endl;
            return -1;
        }
        for (int j = 0; j < theta.size(); j++) {
            theta = theta_new;
            update(theta, theta_new, X, Y, rate, j);
            
            // compute l2 norm of difference of updated and previous theta
            diff_value = 0;
            for (int j = 0; j < theta.size(); j++) {
                diff_value += pow((theta_new(j) - theta(j)), 2);
            }
            diff_value = sqrt(diff_value);
            // store l2 norm into the sliding window
            diff.push_back(diff_value);
            if (diff.size() > sliding_window) diff.erase(diff.begin(),diff.begin()+diff.size()-sliding_window);
            
            cout << "moving average of differences is: " << accumulate(diff.begin(), diff.end(), 0.0) / diff.size() << endl;
            // stop SGD if the average of l2 norms in the sliding window is samller than tolerance
            if (accumulate(diff.begin(), diff.end(), 0.0) / diff.size() < tol) {
                flag = 0;
                break;
            }
            // print theta
            cout << "theta is: ";
            for (int j = 0; j < theta.size(); j++) {
                cout << theta_new(j) << " ";
            }
            cout << endl;
        }
        // decrease learning rate
        rate /= 1 + num / double(N);
    }
    cout << endl;
    
    // compute standard error of the coefficients
    if (N<=1000) {
        for (int i = 0; i < X.rows(); i++) {
            double sum = X.row(i) * theta;
            double p = 1 / (1 + exp(-sum));
            P(i) = p*(1-p);
        }
        MatrixXd W(N,N);        // weights matrix
        W = P.asDiagonal();
        SE = (X.transpose() * W * X).inverse().diagonal().array().sqrt();
    }
    
    // print number of iterations
    cout << "number of iterations is: " << num << endl;
    // print theta
    cout << "theta is: ";
    for (int j = 0; j < theta.size(); j++) {
        cout << theta_new(j) << " ";
    }
    cout << endl;
    
    // print SE
    if (N<=1000) {
        cout << "standard error is: ";
        for (int j = 0; j < theta.size(); j++) {
            cout << SE(j) << " ";
        }
        cout << endl;
    }
    
    return 0;
}

// update one coordinate/parameter at a time using whole data
void update(VectorXd &theta, VectorXd &theta_new, MatrixXd &X, VectorXd &Y, double rate, int j) {
    double gradient = 0;
    for (int i = 0; i < X.rows(); i++) {
        double sum = X.row(i) * theta;
        double p = 1 / (1 + exp(-sum));
        double error = Y(i) - p;
        gradient += X(i,j) * error;
    }
    theta_new(j) = theta(j) + rate * gradient;
}
