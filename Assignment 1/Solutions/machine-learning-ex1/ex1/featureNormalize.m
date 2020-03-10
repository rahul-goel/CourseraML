function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
num_parameters = size(X, 2);
mu = zeros(1, size(X, 2));
% size(X, 2) is the same as size(X)(2)
% size returns 1x2 matrix, and we are accessing second element
% of that i.e. no. of columns
sigma = zeros(1, size(X, 2));
% X_norm is the normalised X, mu is the mean value and sigma is
% standard deviation. 
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

for i = 1:num_parameters
    mu(:, i) = mean(X(:, i))
    sigma(:, i) = std(X(:, i))
    X_norm(:, i) = (X(:, i) - mu(:, i)) / sigma(:, i)


% mu(:, 1) = mean(X(:, 1))
% mu(:, 2) = mean(X(:, 2)) 

% sigma(:, 1) = std(X(:, 1)) 
% sigma(:, 2) = std(X(:, 2))

% X_norm(:, 1) = (X(:, 1) - mu(:, 1)) / sigma(:, 1)
% X_norm(:, 2) = (X(:, 2) - mu(:, 2)) / sigma(:, 2)


% ============================================================

end
