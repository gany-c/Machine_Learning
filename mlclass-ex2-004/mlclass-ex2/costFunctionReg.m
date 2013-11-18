function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


nonRegCost = costFunction(theta,X,y);

regPenalty = (lambda/(2*m)) * (sum(theta .^ 2)- theta(1)^2);

J = nonRegCost + regPenalty;

H = X * theta;
sigmoidV = sigmoid(H);
tempgrad = ((sigmoidV - y)' * X)/m;
disp("--start of tempgrad");
disp(tempgrad);
disp("---------");
term1 = tempgrad(1);
disp("---start of term1");
disp(term1);
disp("---------");
regvector = (lambda/m)*theta;
disp("----start of regVector");
disp(regvector);
disp("------");
tempgrad =tempgrad+regvector';
disp("----start of tempgrad regvector");
disp(tempgrad);
disp("----");
tempgrad(1) = term1;
grad = tempgrad;

% =============================================================



end
