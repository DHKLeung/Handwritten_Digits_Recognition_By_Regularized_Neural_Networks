function [Theta1, Theta2, Theta3, Theta4, J_History] = FeedForwardBackProp(Theta1, Theta2, Theta3, Theta4, epoch, alpha, lambda, X, y, s1, s2, s3, s4, num_labels)

%   Initialization
m = size(X, 1); %   number of dataset
J_History = zeros(epoch, 1);    %   record the cost of each iteration
Theta1grad = zeros(size(Theta1));   %   save the gradient of Theta1
Theta2grad = zeros(size(Theta2));   %   save the gradient of Theta2
Theta3grad = zeros(size(Theta3));   %   save the gradient of Theta3
Theta4grad = zeros(size(Theta4));   %   save the gradient of Theta4
a1 = [ones(m, 1), X];   %   add intercept terms to the input layer

%   Process in Neural Network training
for i = 1:epoch

    %   Feed Forward
    z2 = a1 * Theta1';
    a2 = [ones(m, 1), sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = [ones(m, 1), sigmoid(z3)];
    z4 = a3 * Theta3';
    a4 = [ones(m, 1), sigmoid(z4)];
    z5 = a4 * Theta4';
    a5 = softmax(z5);
    
    %   Compute Regularized Cost
    for k = 0:num_labels - 1    %   compute the non-regularized cost
        yOfk = double(y == k);
        J_History(i) = J_History(i) + (1 / m) * sum(((-yOfk) .* log(a5(:, k + 1))) - (1 - yOfk) .* log(1 - a5(:, k + 1)));
    end
    SumOfTheta1_R = sum(sum(Theta1(:, 2:s1 + 1) .^ 2)); %   compute all the terms for regularization
    SumOfTheta2_R = sum(sum(Theta2(:, 2:s2 + 1) .^ 2));
    SumOfTheta3_R = sum(sum(Theta3(:, 2:s3 + 1) .^ 2));
    SumOfTheta4_R = sum(sum(Theta4(:, 2:s4 + 1) .^ 2));
    J_History(i) = J_History(i) + (lambda / (2 * m)) * (SumOfTheta1_R + SumOfTheta2_R + SumOfTheta3_R + SumOfTheta4_R); %   regularize the cost
    
    %   Back Propagation and Compute regularized gradient of each weight
    temp = 0:num_labels - 1;
    for t = 1:m    %    back propagate
        yt = double(temp == y(t));
        delta5 = a5(t, :) - yt;
        delta4 = delta5 * Theta4 .* [1, sigmoid_diff(z4(t, :))];
        delta4 = delta4(2:end);
        delta3 = delta4 * Theta3 .* [1, sigmoid_diff(z3(t, :))];
        delta3 = delta3(2:end);
        delta2 = delta3 * Theta2 .* [1, sigmoid_diff(z2(t, :))];
        delta2 = delta2(2:end);
        Theta1grad = Theta1grad + delta2' * a1(t, :);
        Theta2grad = Theta2grad + delta3' * a2(t, :);
        Theta3grad = Theta3grad + delta4' * a3(t, :);
        Theta4grad = Theta4grad + delta5' * a4(t, :);
    end
    Theta1grad = Theta1grad ./ m;   %   compute non-regularized gradient of each weight
    Theta2grad = Theta2grad ./ m;
    Theta3grad = Theta3grad ./ m;
    Theta4grad = Theta4grad ./ m;
    Theta1grad(:, 2:end) = Theta1grad(:, 2:end) + (lambda / m) .* Theta1(:, 2:end);    %   compute regularized gradient of each weight
    Theta2grad(:, 2:end) = Theta2grad(:, 2:end) + (lambda / m) .* Theta2(:, 2:end);
    Theta3grad(:, 2:end) = Theta3grad(:, 2:end) + (lambda / m) .* Theta3(:, 2:end);
    Theta4grad(:, 2:end) = Theta4grad(:, 2:end) + (lambda / m) .* Theta4(:, 2:end);
    
    %   Gradient Descent
    Theta1 = Theta1 - alpha .* Theta1grad;
    Theta2 = Theta2 - alpha .* Theta2grad;
    Theta3 = Theta3 - alpha .* Theta3grad;
    Theta4 = Theta4 - alpha .* Theta4grad;
    
    %   print the training status
    if mod(i, 10) == 0
        fprintf('Iteration: %d, Cost: %f\n', i, J_History(i));       
    end
end

end