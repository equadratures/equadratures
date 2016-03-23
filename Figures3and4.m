%--------------------------------------------------------------------------
% Codes for replicating Figures 3 and 4 from the paper. 
% For m = l (figure 3) use offset_value = 1
% For m = 1.2l (figure 4) use offset_value = 1.2
%
% This code requires pmpack and ciplot.
%
% Copyright (c) 2016 by Pranay Seshadri
%
%--------------------------------------------------------------------------
clear; close all; clc;
offset_value = 1.2; % must always be >= 1
%---------------------------Basic Setup------------------------------------
s = parameter('Legendre', -1, 1); % parameter is Legendre
n_big = 80; % number of quadrature points
spacing = 2; % spacing for plotting
[p,w] = gaussian_quadrature(s,n_big); % Gauss-Legendre quadrature points
fun = @(x) exp(x(1)); % function -- you can replace this.
P = evaluate_ops(s,n_big,p); % Set up "P" and call A = P-transpose
W = diag(sqrt(w)); % diagonal matrix of sqrt(weights)
g = funceval(fun, p); % function eval'd at quadrature points
A =  W' * P'; % the design matrix
y = W' * g; % weighted function eval's
%----------Compute errors from randomized and optimal quadrature-----------
kk = 1;
n_stop = n_big;
begin = 3;
for k = begin : spacing : n_stop
    n = k; m = round(n/offset_value);
    [error_qr(kk), condition_QR(kk), mean_error_rand(kk), max_error_rand(kk), min_error_rand(kk), ...
        mean_cond_rand(kk), max_cond_rand(kk), min_cond_rand(kk)] = compute_errors(A, p, w, g, y, m, n, n_big);
    kk = kk + 1;
end
%-----------------------Plot the graphs!-----------------------------------
% Coefficient errors plot
samples = begin : spacing : n_stop ;
figure1 = figure;
axes1 = axes('Parent',figure1,'LineWidth',2,'YScale','log','FontSize',18);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'on');
ciplot(min_error_rand, max_error_rand,samples, [0.729411780834198 0.831372559070587 0.95686274766922])
plot(samples, mean_error_rand, 'b-', 'LineWidth', 4, 'DisplayName', 'Random');
plot(samples, error_qr, 'r-', 'LineWidth', 4, 'DisplayName', 'QR pivoting');
xlabel('Number of rows in square matrix', 'Interpreter', 'Latex');
ylabel('Error $\epsilon$', 'Interpreter', 'Latex');
legend('Min-max of randomized quadrature', 'Mean of randomized quadrature', 'Optimal quadrature');
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.16853996155754 0.811620795107036 0.382285466269841 0.0596330275229358],...
    'Orientation','vertical',...
    'EdgeColor',[1 1 1],...
    'FontSize',18, 'Interpreter', 'Latex');
xlim([begin 70]);
ylim([10^(-17) 10^10]);
hold off;
% Condition numbers plot
figure2 = figure;
set(gca, 'FontSize', 18, 'LineWidth', 2, 'Yscale', 'log'); hold on; grid on; box on;
ciplot(min_cond_rand, max_cond_rand,samples, [0.729411780834198 0.831372559070587 0.95686274766922])
plot(samples, mean_cond_rand, 'b-', 'LineWidth', 4,  'DisplayName', 'Random');
plot(samples, condition_QR, 'r-', 'LineWidth', 4,  'DisplayName', 'QR pivoting');
xlabel('Subsampled points $l$', 'Interpreter', 'Latex');
ylabel('$\kappa$', 'Interpreter', 'Latex' );
xlim([begin 70]);

