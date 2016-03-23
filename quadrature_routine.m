% QUADRATURE_ROUTINE Compute univariate quadrature points and weights for a 
%particular sampling strategy
%
% [p,w] = quadrature_routine(quadrature_type, order)
%
% Returns a vector of quadrature points and a vector of the corresponding
% quadrature weights for numerical integration over [-1,1]
%
% Inputs:
%   type     : quadrature type
%   order    : number of quadrature points + 1
%
% Outputs:
%   p     : a vector of quadrature points
%   w     : a vector of quadrature weights
%
% Example:
%   [p,w] = quadrature_routine('Monte Carlo', 20)
%   [p,w] = quadrature_routine('Clenshaw-Curtis', 17)
%   [p,w] = quadrature_routine('Gauss-Legendre', 32)
%
%
% Copyright 2015 Pranay Seshadri (pranay.seshadri@gmail.com)
%
% History
% -------
% :2015-11-18: Initial release

function [p,w] = quadrature_routine(quadrature_type, order)

switch lower(quadrature_type)
    case 'mc'
        p = rand(order,1).*2 - 1; 
        w = ones(order,1)./(order);
    case 'gl'
        s = parameter('Legendre', -1 ,1 );
        [p,w] = gaussian_quadrature(s, order) ;
    case 'cc'
         [p,w] = chebpts(order);
    otherwise
        error('Unrecognized quadrature type: %s',quadrature_type);
end


end