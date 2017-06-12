//
//  main.cpp
//  EQuad
//
//  Created by Pranay Seshadri on 6/9/17.
//  Copyright © 2017 Pranay Seshadri. All rights reserved.
//

#include <iostream>
#include "recurrencecoefficients.hpp"

int main(int argc, const char * argv[]) {
    // insert code here...
    //std::cout << "Hello, World!\n";
    //return 0;
    
    RecurrenceCoefficients* x1 = new RecurrenceCoefficients(5, -1.0, 1.0, 0.0, 0.0, "Uniform");
    MatrixXd toby;
    toby = x1->getRecurrenceCoefficients();
    cout<<toby;
    
    return 1;
};
