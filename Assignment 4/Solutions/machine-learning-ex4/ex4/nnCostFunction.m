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

% forward propagation
% adding the bias term. and moving the number of training examples into columns
X = [ones(size(X, 1), 1) X];
a1 = X';   % 401 * 5000

z2 = Theta1 * a1;    % (25 * 401) * (401 * 5000) = (25 * 5000)
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)) ; a2]; % it becomes 26 * 5000

z3 = Theta2 * a2;   % (10 * 26) * (26 * 5000) = (10 * 5000)
a3 = sigmoid(z3);   % which is the final output of the neural network

y_new = zeros(num_labels, m);
% note that y is a 5000 length vecttor and y(i) is the output of the ith value of that
for i=1:m
    y_new(y(i), i) = 1;
end
% we made the y into a 10 * 5000 matrix that corresponds with our output.
% sum of matrix returns the same of all columns in the form of a row vector.
% sum of that row vector gives the total sum

J = (-1/m) * sum(sum(y_new.*log(a3) + (1-y_new).*log(1-a3)));

% we also have to penalise the the parameters using regularization
% but we do not regularize the bias term parameter out of convention
% this would be the first column of our theta matrices.

reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));

J = J + reg;

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
%

% backpropagation is done for each trainig example separately after the forward propagation for it is done.
% the forward propagation part we did in one go already.

for i=1:m
   % we can use the already caluculated values in the forward propagation during the calculation of J
   % but we will do it again as it is meant to be forward and backward for each training data

   % forward propagation
   a1 = (X(i, :))';     % 401 * 1
   z2 = Theta1 * a1;    %(25 * 401) * (401 * 1) = (25 * 1)
   a2 = sigmoid(z2);

   a2 = [1 ; a2];        % adding the bias term
   z3 = Theta2 * a2;    % (10 * 26) * (26 * 1) = (10 * 1)
   a3 = sigmoid(z3);    % the final layer

   % calulation of delta
   delta3 = a3 - y_new(:, i);
   z2 = [1 ; z2];    %bias added for taking matrix mult with theta'
   delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);

   delta2 = delta2(2:end);  %remove the one that came from bias

   Theta2_grad = Theta2_grad + delta3 * a2';
   Theta1_grad = Theta1_grad + delta2 * a1';

end
% without any regularizaion as of now.
% Theta2_grad = (1/m) * Theta2_grad; % avg over all examples.
% Theta1_grad = (1/m) * Theta1_grad; % avg over all examples.

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% implementing regularization for the non-0 thetas in Theta
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
