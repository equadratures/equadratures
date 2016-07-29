% Beta test.
alpha = 4; betav = 2;
s = parameter('Jacobi',  0, 1, betav - 1, alpha-1);
N = 5000;
t = linspace(0,1,N);
wt = (t.^(alpha - 1) .* (1 - t).^(betav - 1) )./(beta(alpha, betav) );
wt = wt./sum(wt);
[p,w] = gaussian_quadrature(s, N);
xw = [t; wt]';
ab = stieltjes(5,xw)
ab_lan = lanczos(5,xw)
ab_test = s.recur(5)

% Gaussian test.
close all; clc;
mu = 0; sigma = sqrt(0.5);
t = linspace(-sigma*10,sigma*10,N);
s = parameter('hermite');
[p,w] = gaussian_quadrature(s, N);

wt = 1/(sqrt(2*sigma^2 * pi)) * exp( -(t - mu).^2 .* 1/(2*sigma^2) );
wt = wt./sum(wt);
figure1 = figure; plot(t, wt); xlim([-10 10]);
figure2 = figure; plot(p, w); xlim([-10 10]);
ab = s.recur(5)
ab_stie = stieltjes(5, [t;wt]')

