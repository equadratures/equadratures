%--------------------------------------------------------------------------
% Codes for replicating Figure 2 from the paper. 
%
% This code requires pmpack
%
% Copyright (c) 2016 by Pranay Seshadri
%
%--------------------------------------------------------------------------
% Main inputs (feel free to change these):
fun_type = 'exp';
quadrature_type = 'GL';
switch lower(fun_type)
    case 'exp'
        fun = @(x) exp(x(1));
        climits = [-15 0];
    case 'power'
        fun = @(x) x(1).^10;
        climits = [-15 0];
    case 'abs'
        fun = @(x) abs(x(1) - 0.2).^3;
        climits = [-8 0];
    case 'sine'
        fun = @(x) sin(5.*(x(1) - 0.5) );
        climits = [-15 0];   
    otherwise
        error('Unrecognized function type: %s',fun_type);
end

order = 80; % maximum number of random samples
s = parameter('Legendre', -1, 1); % uniformly dist. random var. [-1,1]
%--------------------------------------------------------------------------
% Code to generate a big matrix A, which is square
[points,weights] = quadrature_routine(quadrature_type, order); % quadrature points and weights
W = diag(sqrt(weights));
P = evaluate_ops(s, order, points);
A = W' * P'; % this is our "big" A
f = funceval(fun, points); % evaluate function at all points
b = W' * f;
X = pseudospectral(fun, s, order);

% For setting up the error
x_grid = linspace(-1,1,200)';
y_grid = funceval(fun, x_grid);

% Two loops to construct an m x n matrix A_T, where m>=n.
for j = 1 : order % # of samples
    for i = 1 : order % # of basis terms
   
        % We only want a tall matrix!
        if(j >= i)
            
            % Subselect j random quadrature points
            v = sort(randperm(order, j));
            samples = points(v); weights_small = weights(v); % Key! we are literally sampling from the large matrix!
            W_tall = diag(sqrt(weights_small));
            P_tall = evaluate_ops(s, i, samples); %--> this implies our order is increasing!!!!
            A_tall = W_tall' * P_tall';
            f_tall = funceval(fun, samples);
            b_tall = W_tall' * f_tall; 
            x_estimate = A_tall\b_tall;
            Basis = evaluate_ops(s,i,x_grid);
            y_estimate = Basis' * x_estimate;
            
            % Compute 
            coefficient_error(j,i) = norm(x_estimate - (X.coefficients(1:i))', 2);
            function_error(j,i) = norm(y_estimate - y_grid, 2);
            condi(j,i) = cond(A_tall);
        else
            coefficient_error(j,i) = NaN;
            function_error(j,i) = NaN;
            condi(j,i) = NaN;
            
        end
    end
end


%--------------------------------------------------------------------------
% Figures
close all;

figure3_string = strcat('cond_', quadrature_type);
climits = [-15 0];
figure1 = figure;
set(gca, 'FontSize', 22, 'LineWidth', 2); hold on; grid on; box on;
surf(log10(coefficient_error)); shading interp;
xlim([1 order-10]); ylim([1 order-10]); caxis(climits);
xlabel('$m$', 'Interpreter', 'latex'); ylabel('$l$', 'Interpreter', 'latex'); colormap jet; 
colorbar;
hold off;

figure3 = figure;
set(gca, 'FontSize', 22, 'LineWidth', 2); hold on; grid on; box on;
surf(log10(abs(condi))); shading interp;
xlim([1 order-10]); ylim([1 order-10]); caxis([0 15]);
xlabel('$m$', 'Interpreter', 'latex'); ylabel('$l$', 'Interpreter', 'latex'); colormap jet;
colorbar;
hold off;
