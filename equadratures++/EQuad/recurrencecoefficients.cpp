//
//  recurrencecoefficients.cpp
//  EQuad
//
//  Created by Pranay Seshadri on 6/12/17.
//  Copyright © 2017 Pranay Seshadri. All rights reserved.
//

#include "recurrencecoefficients.hpp"

// Empty constructor
RecurrenceCoefficients::RecurrenceCoefficients(int input_order, double input_lower, double input_upper, double input_shape_parameter_A, double input_shape_parameter_B, string input_parameter_type)
{
    order = input_order;
    lower = input_lower;
    upper = input_upper;
    shape_parameter_A = input_shape_parameter_A;
    shape_parameter_B = input_shape_parameter_B;
    parameter_type = input_parameter_type;
}


MatrixXd RecurrenceCoefficients::getRecurrenceCoefficients(){
    
    // Allocate memory
    MatrixXd ab = MatrixXd::Zero(order, 2);
    MatrixXd xw(N,2);
    
    
    // Get the xw values from the distribution!
    if (parameter_type.compare("Beta") == 0){
        xw = getPDF_BetaDistribution();
        ab = getCustomRecurrenceCoefficients(xw);
        
    }
    else if(parameter_type.compare("Uniform") == 0){
        xw = getJacobiRecurrenceCoefficients();
    }
    else if(parameter_type.compare("Gaussian") == 0 ){
        double mu = shape_parameter_A;
        double sigma = sqrt(shape_parameter_B);
        xw = getPDF_GaussianDistribution(mu, sigma);
        ab = getCustomRecurrenceCoefficients(xw);
    }
    else if(parameter_type.compare("Exponential") == 0){
        double lambda = shape_parameter_A;
        xw = getPDF_ExponentialDistribution(lambda);
        ab = getCustomRecurrenceCoefficients(xw);
    }
    else if(parameter_type.compare("Cauchy") == 0){
        double x0 = shape_parameter_A;
        double gamma_value = shape_parameter_B;
        xw = getPDF_CauchyDistribution(x0, gamma_value);
        ab = getCustomRecurrenceCoefficients(xw);
    }
    else if(parameter_type.compare("Gamma") == 0 ){
        double k = shape_parameter_A;
        double theta = shape_parameter_B;
        xw = getPDF_GammaDistribution(k, theta);
        ab = getCustomRecurrenceCoefficients(xw);
    }
    else if(parameter_type.compare("Weibull") == 0 ){
        double lambda = shape_parameter_A;
        double k = shape_parameter_B;
        xw = getPDF_WeibullDistribution(lambda, k);
        ab = getCustomRecurrenceCoefficients(xw);
    }
    else if(parameter_type.compare("Truncated-Gaussian") == 0 ) {
        double mu = shape_parameter_A;
        double sigma = sqrt(shape_parameter_B);
        xw = getPDF_TruncatedGaussianDistribution(mu, sigma, lower, upper);
        ab = getCustomRecurrenceCoefficients(xw);
    }
    else{
        //throw "Unknown parameter!";
        cout<<"Her!"<<endl;
    }
    return xw;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *  Calculate recurrence coefficients!
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
// Routine to compute the reucrrence coefficients
MatrixXd RecurrenceCoefficients::getCustomRecurrenceCoefficients(MatrixXd xw){
    
    // Allocate memory
    MatrixXd ab = MatrixXd::Zero(order, 2);
    RowVectorXd w(N), p1, p2, nonzero_indices(N);
    double s;
    int counter=0;
    
    // Normalize the elements of w!
    s = xw.col(1).sum();
    for(int i=0; i<order; i++){
        xw(i,1) = xw(i,1)/s;
    }
    
    // Get the non-zero indices!
    for(int i=0; i<order; i++){
        if (xw(i,1) != 0) {
            nonzero_indices(counter) = i;
            counter++;
        }
        w(i) = xw(i,1);
    }
    
    // Now we copy only the non-zero indices into xw_nonzero_indices!
    MatrixXd xw_nonzero_indices(counter,2);
    for(int i=0; i<counter; i++){
        xw_nonzero_indices(i,0) = xw(nonzero_indices(i), 0);
        xw_nonzero_indices(i,1) = w(nonzero_indices(i));
    }
    
    ab(0,0) = xw.col(0).dot(xw.col(1));
    ab(0,1) = s;
    cout<<ab(0,0)<<endl;
    cout<<ab(0,1)<<endl;
    if (order == 1){
        return ab;
    }
    
    p1 = RowVectorXd::Zero(counter);
    p2 = RowVectorXd::Zero(counter);
    
    
    //cout<<ab<<endl;
    return ab;
}



// Routine to compute the reucrrence coefficients
MatrixXd RecurrenceCoefficients::getJacobiRecurrenceCoefficients(){
    double a0, b2a2;
    MatrixXd ab = MatrixXd::Zero(order, 2);
    
    // Preliminaries!
    a0 = (shape_parameter_B - shape_parameter_A)/(shape_parameter_A + shape_parameter_B + 2.0);
    b2a2 = pow(shape_parameter_B, 2) - pow(shape_parameter_A, 2);
    
    if(order > 0){
        ab(0,0) = a0;
        ab(0,1) = (pow(2.0, shape_parameter_A + shape_parameter_B + 1) * tgamma(shape_parameter_A + 1.0) * tgamma(shape_parameter_B + 1.0))/(tgamma(shape_parameter_A + shape_parameter_B + 2.0));
    }
    
    for(int k=1; k<order; k++){
        int temp = k+1;
        ab(k,0) = b2a2 /( ( 2*(temp - 1.0) + shape_parameter_A + shape_parameter_B ) * (2.0 * temp + shape_parameter_A + shape_parameter_B) );
        if (k == 1){
            ab(k,1) = (4.0 * (temp - 1.0) * (temp - 1.0 + shape_parameter_A) * (temp - 1.0 + shape_parameter_B)  )/( pow( 2 * (temp - 1.0) + shape_parameter_A + shape_parameter_B , 2) * (2 * (temp - 1.0) + shape_parameter_A + shape_parameter_B + 1) );
        }
        else{
            ab(k,1) = (4.0 * (temp - 1.0) * (temp - 1.0 + shape_parameter_A) * (temp - 1.0 + shape_parameter_B) * (temp - 1.0 + shape_parameter_A + shape_parameter_B ) )/( pow(2 * (temp - 1.0) + shape_parameter_A + shape_parameter_B, 2) * (2.0*(temp - 1.0) + shape_parameter_A + shape_parameter_B + 1.0) * (2.0 * (temp - 1.0) + shape_parameter_A + shape_parameter_B - 1.0) );
        }
        
    }
    return ab;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *  Analytical PDFs
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
 // Beta distribution
MatrixXd RecurrenceCoefficients::getPDF_BetaDistribution(){
    
    VectorXd x = VectorXd::LinSpaced(N,0.0,1.0) ;
    VectorXd xdim = VectorXd::LinSpaced(N,lower,upper) ;
    MatrixXd xw(N,2);
    double a, b;
    
    
    for(int i=0; i<N; i++){
        // For the nodes!
        a = xdim(i);
        
        // For the weights!
        b = (pow(x(i), (shape_parameter_A - 1.0) ) * pow( 1 - x(i), (shape_parameter_B - 1.0)  ))/(beta(shape_parameter_A, shape_parameter_B)) * 1.0/(upper - lower);
        
        xw(i,0) = a;
        xw(i,1) = b;
    };
    return xw;
}

// Uniform distribution
MatrixXd RecurrenceCoefficients::getPDF_UniformDistribution(){
    
    VectorXd x = VectorXd::LinSpaced(N,lower,upper) ;
    MatrixXd xw(N,2);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = 1.0/(upper - lower);
    };
    return xw;
}

// Weibull distribution
MatrixXd RecurrenceCoefficients::getPDF_WeibullDistribution(double lambda, double k){
    
    VectorXd x = VectorXd::LinSpaced(N,0.0, 15.0/k) ;
    MatrixXd xw(N,2);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = k/lambda * pow(x(i)/lambda, k-1) * exp(-1.0 * pow( (x(i)/lambda) , k) );
    };
    return xw;
}

// Gamma distribution
MatrixXd RecurrenceCoefficients::getPDF_GammaDistribution(double k, double theta){
    
    VectorXd x = VectorXd::LinSpaced(N,0.0, k * theta * 10.0) ;
    MatrixXd xw(N,2);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = 1.0/(tgamma(k) * pow(theta, k)) * pow(x(i), k-1) * exp(-1.0 * x(i)/theta ) ;
    };
    return xw;
}

// Exponential distribution
MatrixXd RecurrenceCoefficients::getPDF_ExponentialDistribution(double lambda){
    
    VectorXd x = VectorXd::LinSpaced(N,0.0, 20.0*lambda) ;
    MatrixXd xw(N,2);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = lambda * exp(-lambda * x(i));
    };
    return xw;
}

// Cauchy distribution
MatrixXd RecurrenceCoefficients::getPDF_CauchyDistribution(double x0, double gamma_value){
    
    VectorXd x = VectorXd::LinSpaced(N,-16.0*gamma_value, 16.0*gamma_value) ;
    MatrixXd xw(N,2);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = 1.0/( M_PI * gamma_value * (1 +  pow((x(i) - x0)/(gamma_value), 2 ) ) );
    };
    return xw;
}

// Gaussian distribution
MatrixXd RecurrenceCoefficients::getPDF_GaussianDistribution(double mu, double sigma){
    
    VectorXd x = VectorXd::LinSpaced(N,-16.0*sigma, 16.0*sigma) ;
    MatrixXd xw(N,2);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = 1.0/( sqrt(2 * pow(sigma, 2) * M_PI ) ) * exp(-1.0 * pow(x(i) - mu, 2) * 1.0/(2 * pow(sigma, 2))  ) ;
    };
    return xw;
}

// Truncated Gaussian distribution
MatrixXd RecurrenceCoefficients::getPDF_TruncatedGaussianDistribution(double mu, double sigma, double left, double right){
    
    VectorXd x = VectorXd::LinSpaced(N, left, right) ;
    MatrixXd xw(N,2);
    double first_term = GaussianCDF(right, mu , sigma);
    double second_term = GaussianCDF(left, mu, sigma);
    
    for(int i=0; i<N; i++){
        xw(i,0) = x(i);
        xw(i,1) = 1.0/( sqrt(2 * pow(sigma, 2) * M_PI ) ) * exp(-1.0 * pow(x(i) - mu, 2) * 1.0/(2 * pow(sigma, 2))  ) ;
        xw(i,1) = 1.0/sigma * xw(i,1);
        xw(i,1) = xw(i,1) / (first_term - second_term);
    };
    
    return xw;
}

// Beta function
double RecurrenceCoefficients::beta(double a, double b){
    return (tgamma(a) * tgamma(b))/(tgamma(a+b));
}

// Gaussian Cumulative density function
double RecurrenceCoefficients::GaussianCDF(double constant, double mu, double sigma){
    double w = 1.0/(2.0) * (1 + erf((constant - mu)/(sigma * sqrt(2))));
    return w;
}
