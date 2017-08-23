function s = softmax(z)

%   Initialization
m = size(z, 1);
n = size(z, 2);
s = zeros(size(z));

%   Compute Softmax
for i = 1:m
    for j = 1:n
        s(i, j) = exp(z(i, j)) / sum(exp(z(i, :)));
    end
end

end