function p = predictions(X, Theta1, Theta2, Theta3)

m = size(X, 1);
a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(m, 1), sigmoid(z3)];
z4 = a3 * Theta3';
a4 = softmax(z4);
[temp, p] = max(a4, [], 2);   %   find out the index of that max value and save in predict
p = p - 1;  %   matlab's indexing starts from 1, but my code considers index 1 as digit 0

end