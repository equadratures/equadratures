//
//  parameter.cpp
//  EQuad
//
//  Created by Pranay Seshadri on 6/9/17.
//  Copyright © 2017 Pranay Seshadri. All rights reserved.
//

#include "parameter.h"
#include <cmath>

// Empty constructor
Parameter::Parameter()
{
    order = 3;
    lower = -1.0;
    upper = 1.0;
    shape_parameter_A = 0.0;
    shape_parameter_B = 0.0;
    parameter_type = "Uniform";
}

// Loaded constructor
Parameter::Parameter(int o, double l, double u, double A, double B, string type)
{
    order = o;
    lower = l;
    upper = u;
    shape_parameter_A = A;
    shape_parameter_B = B;
    parameter_type = type;
}


