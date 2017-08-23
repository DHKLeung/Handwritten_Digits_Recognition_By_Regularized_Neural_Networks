function g_diff = sigmoid_diff(z)

%   Compute the differentiated sigmoid of z
g_diff = sigmoid(z) .* (1 - sigmoid(z));

end