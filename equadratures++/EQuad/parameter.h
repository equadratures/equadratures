//
//  parameter.h
//  EQuad
//
//  Created by Pranay Seshadri on 6/9/17.
//  Copyright © 2017 Pranay Seshadri. All rights reserved.
//

#ifndef parameter_h
#define parameter_h
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#endif /* parameter_h */

class Parameter{
public:
    
    // Global variables
    int order;
    double lower, upper, shape_parameter_A, shape_parameter_B;
    string parameter_type;
    int N=8000;
    
    // Constructor!
    Parameter();
    //~Parameter();
    Parameter(int o, double l, double u, double A, double B, string type);
    //Parameter(int order, double lower, double upper, double shape_parameter_A, double shape_parameter_B, string parameter_type);
    
    
    // Set methods
    //void virtual init();
    //void setRecurrenceCoefficients();
    
    // Get methods
    //double getMean();
    //RowVectorXd getUnivariateQuadraturePoints(int order);
    //RowVectorXd getUnivariateQuadratureWeights(int order);
    //MatrixXd getRecurrenceCoefficients();
    /*
    Matrix2d getCDF(int N);
    Matrix2d getPDF(int N);
    RowVectorXd getInverseCDF(RowVectorXd x);
    MatrixXd getJacobiMatrix(int order);
    MatrixXd getJacobiEigenvectors(int order);
    MatrixXd getOrthogonalPolynomialMatrix(int order, RowVectorXd points);
    MatrixXd getDerivativeOfOrthogonalPolynomialMatrix(int order, RowVectorXd points);
    RowVectorXd getOrthogonalPolynomial(int order, RowVectorXd points);
*/

private:
    // Analytical distributions!
    //MatrixXd getPDF_BetaDistribution(double shape_parameter_A, double shape_parameter_B, double lower, double upper);
    //MatrixXd getPDF_UniformDistribution(double lower, double upper);

    //MatrixXd getJacobiRecurrenceCoefficients(double shape_parameter_A, double shape_parameter_B);
    //Matrix2d getPDF_GaussianDistribution(int N, double shape_parameter_A, double shape_parameter_B, double lower, double upper);
    //Matrix2d getPDF_ExponentialDistribution(int N, double lambda);
    //Matrix2d getPDF_CauchyDistribution(int N, double xo, double gamma);
    //Matrix2d getPDF_GammaDistribution(int N, double k, double theta);
    //Matrix2d getPDF_WeibullDistribution(int N, double lambda, double k);
    //Matrix2d getPDF_TruncatedDistribution(int N, double mu, double sigma, double a, double b);
    
    // Recurrence coefficients
    //Matrix2d getJacobiRecurrenceCoefficients(int order, double shape_parameter_A, double shape_parameter_B);
    
    //*/
    

};
