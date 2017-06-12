//
//  collectionofparameters.hpp
//  EQuad
//
//  Created by Pranay Seshadri on 6/10/17.
//  Copyright © 2017 Pranay Seshadri. All rights reserved.
//

#ifndef collectionofparameters_hpp
#define collectionofparameters_hpp

#include <stdio.h>
#include "parameter.h"
#endif /* collectionofparameters_hpp */

class CollectionOfParameters{
public:
    
    
    // Global variables
    int number_of_parameters, permitted_number_of_function_calls;
    string method, indexset;
    RowVector2d coefficients;

    
    // Constructor!
    CollectionOfParameters(VectorXd Parameters);
    
    //
    // basis = Tensor, Sparse, Total order, Hyperbolic, Sobol sequence
    //
    // sampling = Quadrature, Subsampling:{Uniform-Lattice Chebyshev-Lattice, Christofel-Lattice, Tensor-Lattice, Induced-Lattice}, Given-Dataset
    //
    // approach = direct (default), parallel, iterative
    //
    // method = Integration.Tensor
    //          Integration.Sparse
    //          LeastSquares.Default
    //          LeastSquares.QRP
    //          LeastSquares.LU
    //          LeastSquares.SVD-LU
    //          ConvexOptimization.DeterminantMaximization
    //          CompressedSensing.L1NormMinimizer
    //
    // ratio >= 1.0
    

    // The required set methods!
    void setBasis(string basis);
    void setSampling(string sampling);
    void setMethod(string method);
    
    void setRegressionDataSet(MatrixXd XY);
    void setFunctionEvaluations(RowVector2d fun);
    void setGradientEvaluations(RowVector2d grads);
    
    // An optional method for parallelization / memory saving!
    void setComputationApproach(string approach); // not required by default!
    
    
    
    
    
    // Get methods
    MatrixXd getEvaluationPoints();
    RowVector2d getCoefficients(string basis, string sampling, string method, string approach, RowVector2d fun);
    double getMean(RowVector2d coefficients, MatrixXd multi_indices);
    int numberOfBasisTerms();
    
};
