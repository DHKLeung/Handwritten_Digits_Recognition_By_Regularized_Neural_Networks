function [Theta1, Theta2, J_History] = FeedForwardBackProp(Theta1, Theta2, epoch, alpha, lambda, X, y, s1, s2, num_labels)

%   Initialization
m = size(X, 1); %   number of dataset
J_History = zeros(epoch, 1);    %   record the cost of each iteration
Theta1grad = zeros(size(Theta1));   %   save the gradient of Theta1
Theta2grad = zeros(size(Theta2));   %   save the gradient of Theta2
a1 = [ones(m, 1), X];   %   add intercept terms to the input layer

%   Process in Neural Network training
for i = 1:epoch

    %   Feed Forward
    z2 = a1 * Theta1';
    a2 = [ones(m, 1), sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = softmax(z3);

    %   Compute Regularized Cost
    for k = 0:num_labels - 1    %   compute the non-regularized cost
        yOfk = double(y == k);
        J_History(i) = J_History(i) + (1 / m) * sum(((-yOfk) .* log(a3(:, k + 1))) - (1 - yOfk) .* log(1 - a3(:, k + 1)));
    end
    SumOfTheta1_R = sum(sum(Theta1(:, 2:s1 + 1) .^ 2)); %   compute all the terms for regularization
    SumOfTheta2_R = sum(sum(Theta2(:, 2:s2 + 1) .^ 2));
    J_History(i) = J_History(i) + (lambda / (2 * m)) * (SumOfTheta1_R + SumOfTheta2_R); %   regularize the cost

    %   Back Propagation and Compute regularized gradient of each weight
    temp = 0:num_labels - 1;
    for t = 1:m    %    back propagate
        yt = double(temp == y(t));
        delta3 = a3(t, :) - yt;
        delta2 = delta3 * Theta2 .* [1, sigmoid_diff(z2(t, :))];
        delta2 = delta2(2:end);
        Theta1grad = Theta1grad + delta2' * a1(t, :);
        Theta2grad = Theta2grad + delta3' * a2(t, :);
    end
    Theta1grad = Theta1grad ./ m;   %   compute non-regularized gradient of each weight
    Theta2grad = Theta2grad ./ m;
    Theta1grad(:, 2:end) = Theta1grad(:, 2:end) + (lambda / m) .* Theta1(:, 2:end);    %   compute regularized gradient of each weight
    Theta2grad(:, 2:end) = Theta2grad(:, 2:end) + (lambda / m) .* Theta2(:, 2:end);

    %   Gradient Descent
    Theta1 = Theta1 - alpha .* Theta1grad;
    Theta2 = Theta2 - alpha .* Theta2grad;
 
    %   print the training status
    if mod(i, 10) == 0
        fprintf('Iteration: %d, Cost: %f\n', i, J_History(i));       
    end
end

end