function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];

H = sigmoid(X * Theta1');

l = size(H, 1);
H = [ones(l,1) H];


H2 = sigmoid(H * Theta2');

onesmat = ones(size(H2));

logsig = (-1)*log(H2);

logonesig = (-1)*log(onesmat - H2);

yAsBinaryMatrix = zeros(m,num_labels); 

for i = 1:m,
  yAsBinaryMatrix(i,y(i)) =1 ;
end


nonRegCost = 0;

for j = 1:m,
  rowsum = 0;
  for k = 1:num_labels,
  	rowsum = rowsum +(yAsBinaryMatrix(j,k) * logsig(j,k) + (1 - yAsBinaryMatrix(j,k)) * logonesig(j,k));
  end	 	
  nonRegCost = nonRegCost + rowsum;
end

Temp_Theta1 = Theta1;
Temp_Theta2 = Theta2;

Theta1(:,[1]) = [];
Theta2(:,[1]) = [];

Theta1 = Theta1 .* Theta1;
Theta2 = Theta2 .* Theta2; 


nb_sum1 = sum(Theta1);
scalar_sum1 = sum(nb_sum1);

nb_sum2 = sum(Theta2);
scalar_sum2 = sum(nb_sum2);

regPenalty = (scalar_sum1+scalar_sum2);
regPenalty = (lambda/(2*m))*regPenalty;

J = nonRegCost/m + regPenalty;

Theta1 = Temp_Theta1;
Theta2 = Temp_Theta2;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1:m,
 a_row = X(t,:);
 y_row = yAsBinaryMatrix(t,:); 
 h2_row = sigmoid(a_row * Theta1');

 h2_row = [ones(1,1) h2_row];
 h3_row = sigmoid(h2_row * Theta2');

 delta_row =   h3_row - y_row;

 z2_row = a_row * Theta1';
 z2_row = [1 z2_row];
 z_row_sg = sigmoidGradient(z2_row);

 second_delta_row = (delta_row * Theta2) .* z_row_sg;

 second_delta_row = second_delta_row(2:end);

  Theta2_grad = Theta2_grad + delta_row' * h2_row;
  Theta1_grad = Theta1_grad + second_delta_row' * a_row; 
 
end 

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


regmat1 = Theta1;
regmat2 = Theta2;

regmat1(:,[1]) = [];
regmat2(:,[1]) = [];


l = size(regmat1, 1);
regmat1 = [zeros(l,1) regmat1];

l = size(regmat2, 1);
regmat2 = [zeros(l,1) regmat2];

regmat1 = (lambda/m) * regmat1;
regmat2 = (lambda/m) * regmat2;

Theta1_grad = Theta1_grad + regmat1;
Theta2_grad = Theta2_grad + regmat2;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
