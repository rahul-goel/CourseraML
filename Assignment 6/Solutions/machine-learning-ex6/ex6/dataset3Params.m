function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

results = zeros(size(c_range, 2)*size(sigma_range, 2), 3);
counter = 1;

for c = c_range
    for sig = sigma_range
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
        cur_prediction = svmPredict(model, Xval);
        cur_error = mean(double(cur_prediction ~= yval));
        
        results(counter, : ) = [cur_error, c, sig];
        fprintf("Current iteration %d for C = %d and sigma = %d.\n", counter, c, sig);
        counter += 1;
    end
end

results = sortrows(results, 1);
% sorts by error.

C = results(1, 2);
sigma = results(1, 3);
% =========================================================================

end
