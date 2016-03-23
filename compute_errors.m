function [error_qr, condition_QR, mean_error_rand, max_error_rand, min_error_rand, mean_cond_rand, max_cond_rand, min_cond_rand] = compute_errors(A, points, weights, f_tall, y, m, n, n_big)
% m = basis terms
% n = number of points
s = parameter('Legendre', -1, 1);
A_hat = A(:,1:m); % First select number of basis terms
x_real = A\y; x_real2 = x_real(1:m);
[~,~,P] = qr(A_hat', 'vector'); P = P(1:n);
b_sub = y(P);
A_sub = A_hat(P, :);
x_qr = A_sub \ b_sub;
condition_QR = cond(A_sub);
%% Randomized sampling
% We need to average this result
for i = 1 : 100%0
    v = sort(randperm(n_big,n));
    samples = points(v); weights_small = weights(v);
    W_tall = diag(sqrt(weights_small));
    P_tall = evaluate_ops(s, m, samples); %--> this implies our order is increasing!!!!
    A_tall = W_tall' * P_tall';
    %size(A_tall)
    b_tall = W_tall' * f_tall(v); 
    x_rand = A_tall \ b_tall;
    % NEW WAY
      % Key! we are literally sampling from the large matrix!
            ;
    
    % OLD WAY
    % A_star = A(rows_to_use, 1:m); 
    % x_rand = A_star \ y(rows_to_use);
    condition_rand(i) = cond(A_tall);
    error_rand(i) = norm(x_rand - x_real2 , 2);
end
max_cond_rand = max(condition_rand);
min_cond_rand = min(condition_rand);
mean_cond_rand = mean(condition_rand);

max_error_rand = max(error_rand);
min_error_rand = min(error_rand);
mean_error_rand = mean(error_rand);

%% Norms
error_qr = norm(x_real2 - x_qr, 2);

end