function p = predictions(X_test, mu, stddev, Theta1, Theta2)

for i = 1:size(X_test, 2)
    X_test(:, i) = (X_test(:, i) - mu(i)) ./ stddev(i);
end
m = size(X_test, 1);
a1_t = [ones(m, 1), X_test];
z2_t = a1_t * Theta1';
a2_t = [ones(m, 1), sigmoid(z2_t)];
z3_t = a2_t * Theta2';
a3_t = softmax(z3_t);
[temp, p] = max(a3_t, [], 2);   %   find out the index of that max value and save in predict
p = p - 1;  %   matlab's indexing starts from 1, but my code considers index 1 as digit 0

end