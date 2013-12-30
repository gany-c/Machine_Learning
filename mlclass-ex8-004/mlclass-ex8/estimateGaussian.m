function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%



for j = 1:n
 colVector = X(:,j);
 j_mu = (sum(colVector))/m;
 onesVector = ones(m,1);
 meanVector = onesVector * j_mu;
 colVector = colVector .- meanVector;
 colVector = colVector .^ 2;
 variance  = (sum(colVector))/m;
 mu(j) = j_mu;
 sigma2(j) = variance; 
endfor

disp(" mu = ");
disp(mu);

disp("sigma2 = ");
disp(sigma2);


% =============================================================


end
