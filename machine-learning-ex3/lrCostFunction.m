function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

y1(:,1) = -y;
y1(:,2) = -(1 - y);

h =  sigmoid(X * theta);

X1(1,:) = log(h);
X1(2,:) = log(1-h);

J = y1 * X1;
J = trace(J)
J = J ./ m;

% computing the regularization
reg_theta = theta;
reg_theta(1) = 0;
reg_theta = reg_theta .^ 2;

J = J + (lambda / (2 * m)) * sum(reg_theta);



% computing the gradient

error = h - y;

grad = error' * X;
grad = grad ./ m;

% computing the regularization in gradient
reg_theta = theta;
reg_theta(1) = 0;
reg_theta = reg_theta .* (lambda / m);

grad = grad + reg_theta';

% =============================================================

grad = grad(:);

end
