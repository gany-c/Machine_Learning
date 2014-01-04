function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

%disp("X");
%disp(size(X));

%disp("y");
%disp(size(y));

%disp("theta");
%disp(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

H = X * theta;

difference = H - y;

sqDiff = difference .^ 2;

J = sum(sqDiff)/(2*m);



% =========================================================================

end
