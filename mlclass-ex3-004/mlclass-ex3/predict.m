function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

disp("Theta1");
disp(size(Theta1));
disp("Theta2");
disp(size(Theta2));
disp("X");
disp(size(X));

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];
disp("X");
disp(size(X));

H = sigmoid(X * Theta1');
disp("H =");
disp(size(H));

l = size(H, 1);
H = [ones(l,1) H];

disp("H = ");
disp(size(H));

H2 = sigmoid(H * Theta2');
disp("H2 = ");
disp(size(H2));

[discriminants,p] = max(H2, [],2);


disp("p = ");
disp(size(p));
% =========================================================================


end
