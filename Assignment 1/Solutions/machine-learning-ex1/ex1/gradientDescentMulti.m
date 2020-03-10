function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
num_parameters = length(theta)
J_history = zeros(num_iters, 1);
temp_theta = zeros(num_parameters, 1)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % First we will update the theta0 and then theta1, but they have to be simultaneous

    for i = 1:num_parameters
        temp_theta(i) = theta(i) - alpha * sum(((X * theta) - y) .* X(:, i)) / m;

    theta = temp_theta;

    % temp_theta_0 = theta(1) - alpha * sum((X * theta) - y) / m;
    % temp_theta_1 = theta(2) - alpha * sum(((X * theta) - y) .* X(:, 2)) / m;
    % temp_theta_2 = theta(3) - alpha * sum(((X * theta) - y) .* X(:, 3)) / m;
    
    % theta = [temp_theta_0;
    %         temp_theta_1;
    %         temp_theta_2];

    % ============================================================

    % Save the cost J in every iteration    

    J_history(iter) = computeCost(X, y, theta);

end

end
