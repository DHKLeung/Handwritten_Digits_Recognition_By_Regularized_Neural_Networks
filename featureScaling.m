function [X_scaled, mu, stddev] = featureScaling(X)

%   Initialization
n = size(X, 2); %   number of features
X_scaled = zeros(size(X));  %   X after being feature-scaled
mu = zeros(1, n);   %   mean of each feature
stddev = zeros(1, n);   %   standard deviation of each feature

%   Compute X_scaled, mu, stddev
for i = 1: n
    mu(i) = mean(X(:, i));
    stddev(i) = std(X(:, i));
    X_scaled(:, i) = (X(:, i) - mu(i)) ./ stddev(i);
end

end