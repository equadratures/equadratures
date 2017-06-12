//
//  recurrencecoefficients.hpp
//  EQuad
//
//  Created by Pranay Seshadri on 6/12/17.
//  Copyright © 2017 Pranay Seshadri. All rights reserved.
//

#ifndef recurrencecoefficients_hpp
#define recurrencecoefficients_hpp
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#endif /* recurrencecoefficients_hpp */

class RecurrenceCoefficients{
public:
    
    // Global variables
    int order;
    double lower, upper, shape_parameter_A, shape_parameter_B;
    string parameter_type;
    int N=8000;
    
    // Constructor!
    RecurrenceCoefficients(int input_order, double input_lower, double input_upper, double input_shape_parameter_A, double input_shape_parameter_B, string input_parameter_type);

    // Public methods!
    MatrixXd getRecurrenceCoefficients();

private:
    // Custom & Jacobi recurrence coefficients!
    MatrixXd getCustomRecurrenceCoefficients(MatrixXd xw);
    MatrixXd getJacobiRecurrenceCoefficients();
    
    // The distributions!
    MatrixXd getPDF_BetaDistribution();
    MatrixXd getPDF_UniformDistribution();
    MatrixXd getPDF_GaussianDistribution(double mu, double sigma);
    MatrixXd getPDF_TruncatedGaussianDistribution(double mu, double sigma, double left, double right);
    MatrixXd getPDF_ExponentialDistribution(double lambda);
    MatrixXd getPDF_CauchyDistribution(double x0, double gamma_value);
    MatrixXd getPDF_GammaDistribution(double k, double theta);
    MatrixXd getPDF_WeibullDistribution(double lambda, double k);
    
    // A beta function!
    double beta(double a, double b);
    
    // Gaussian CDF!
    double GaussianCDF(double constant, double mu, double sigma);
};
