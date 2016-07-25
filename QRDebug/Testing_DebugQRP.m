clear all; close all; clc;

A = [3 2 1 5; 3 2 12 31; 12 2 -1 -3; -6 -7 13 -21; 1 0 -2 52];
[Q, R, pivots] = qr_MGS_pivoting_debug(A)