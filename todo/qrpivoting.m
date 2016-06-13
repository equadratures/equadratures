% MODIFIED GRAM-SCHMIDT QR COLUMN PIVOTING ALGORITHM:
% See:
% 1. Gene Golub & Charles Van Loan, "Matrix Computations" (2003)
% 2. Aciya Dax, "A modified Gram-schmidt algorithm with iterative
% orthogonalization and column pivoting" (2000)
%
% Copyright (c) 2016 by Pranay Seshadri


function [Q,R, pivots] = qr_householderpivotingIII(A)
[m,n] = size(A); % Size of "A" -- can set as input!
column_norms = zeros(n,1); % Initialize column norms vector
epsilon = zeros(n,1); % epsilon array
pivots = 1 : n; % 
%---------------------------------------------------------------------
% Step 0:
%---------------------------------------------------------------------
% 1. Compute all the column norms -- this computation is expensive and
% ideal only for the first iteration. [Change me later -- to Pythogras!]
for j = 1 : n
    column_norms(j,1) = norm(A(1:m, j),2)^2;
    epsilon(j,1) = eps * column_norms(j,1);
end

% Used for pythogras updating!
tau = min(eps^(1/4), 0.001);

% Now loop!
for k = 1 : n
    
    %---------------------------------------------------------------------
    % Step 0:
    %---------------------------------------------------------------------
    % 2. Find the "j*" column index with the highest column norm
    [~,j_star] = max(column_norms(k:n,1));
    j_star = j_star + (k - 1);

    
    % 3. If j* = k, skip to step 1, else swap columns
    if(k ~= j_star)
        
        % Swamp columns in A
        temp = A(1:m,k);
        A(1:m,k) = A(1:m,j_star);
        A(1:m,j_star) = temp;
        
        % Swap columns in R
        for i = 1 : k - 1
            temp = R(i,k);
            R(i,k) = R(i,j_star);
            R(i,j_star) = temp;
        end
        
        % Swap pivots
        temp = pivots(k);
        pivots(k) = pivots(j_star);
        pivots(j_star) = temp;
        
    end
    
    %---------------------------------------------------------------------
    % Step 1: Reorthogonalization
    %---------------------------------------------------------------------
    if( k~=1 )
        for i = 1 : k - 1
            alpha(i) = Q(1:m,i)' * A(1:m,k);
            R(i,k) = R(i,k) + alpha(i);
            A(1:m,k) = A(1:m,k) - alpha(i)*Q(1:m,i);
        end
    end
    
    %---------------------------------------------------------------------
    % Step 2: Normalization
    %---------------------------------------------------------------------
    R(k,k) =  norm(A(1:m,k),2);
    Q(1:m,k) = A(1:m,k)/R(k,k);
    
    %---------------------------------------------------------------------
    % Step 3: Orthogonalization
    %---------------------------------------------------------------------
    if(k ~= n)
        for j = k + 1 : n
            R(k,j) = Q(1:m,k)' * A(1:m,j);
            A(1:m,j) = A(1:m,j) - R(k,j)* Q(1:m,k);
            %column_norms(j,1) = column_norms(j,1) - R(k,j)^2  ;
            %actual = norm(A(1:m, j),2)^2;
            %V = [column_norms(j,1), actual];
            %disp(V)
            
%             % To address difficulty in "Pythogras' updating"
%             %disp('got here----');
%             if(column_norms(j,1) < epsilon(j) / tau)
%                 disp('here')
%                 column_norms(j,1) = sum(A(1:m,j).^2);
%                 epsilon(j,1) = eps * column_norms(j,1);
%             end
            
            % ---- Seems to be pretty similar to MATLAB -------    
            column_norms(j,1) = norm(A(1:m, j),2)^2;
        end
    end
    
    
end


