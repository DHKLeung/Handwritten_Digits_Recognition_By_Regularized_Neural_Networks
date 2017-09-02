function [Theta1, Theta2, Theta3, J_History, J_T_History, Acc_History, Acc_T_History] = FeedForwardBackProp(Theta1, Theta2, Theta3, epoch, alpha, lambda, X, y, s1, s2, s3, num_labels, X_test, mu, stddev, y_test)

%   Initialization
m = size(X, 1); %   number of dataset
m_t = size(X_test, 1);
J_History = zeros(epoch, 1);    %   record the cost on training set of each iteration
J_T_History = zeros(epoch, 1);  %   record the cost on testing set of each iteration
Acc_History = zeros(epoch, 1);  %   record the accuracy on training set of each iteration
Acc_T_History = zeros(epoch, 1);  %   record the accuracy on testing set of each iteration
Theta1grad = zeros(size(Theta1));   %   save the gradient of Theta1
Theta2grad = zeros(size(Theta2));   %   save the gradient of Theta2
Theta3grad = zeros(size(Theta3));
Theta1accugrad = zeros(size(Theta1));
Theta2accugrad = zeros(size(Theta2));
Theta3accugrad = zeros(size(Theta3));

%   Process in Neural Network training
for i = 1:epoch

    %   Feed Forward on training set
    a1 = [ones(m, 1), X];   %   add intercept terms to the input layer on training set
    [z2, a2, z3, a3, z4, a4] = feedForward(a1, Theta1, Theta2, Theta3, m);

    %   Back Propagation, Compute regularized gradient of each weight, Accumulate gradients for Adagrad
    temp = 0:num_labels - 1;
    for t = 1:m    %    back propagate
        yt = double(temp == y(t));
        delta4 = a4(t, :) - yt;
        delta3 = delta4 * Theta3 .* [1, sigmoid_diff(z3(t, :))];
        delta3 = delta3(2:end);
        delta2 = delta3 * Theta2 .* [1, sigmoid_diff(z2(t, :))];
        delta2 = delta2(2:end);
        Theta1grad = Theta1grad + delta2' * a1(t, :);
        Theta2grad = Theta2grad + delta3' * a2(t, :);
        Theta3grad = Theta3grad + delta4' * a3(t, :);
    end
    Theta1grad = Theta1grad ./ m;   %   compute non-regularized gradient of each weight
    Theta2grad = Theta2grad ./ m;
    Theta3grad = Theta3grad ./ m;
    Theta1grad(:, 2:end) = Theta1grad(:, 2:end) + (lambda / m) .* Theta1(:, 2:end);    %   compute regularized gradient of each weight
    Theta2grad(:, 2:end) = Theta2grad(:, 2:end) + (lambda / m) .* Theta2(:, 2:end);
    Theta3grad(:, 2:end) = Theta3grad(:, 2:end) + (lambda / m) .* Theta3(:, 2:end);
    Theta1accugrad = Theta1accugrad + Theta1grad .^ 2;  %   accumulate gradients for Adagrad
    Theta2accugrad = Theta2accugrad + Theta2grad .^ 2;
    Theta3accugrad = Theta3accugrad + Theta3grad .^ 2;
    
    %   Adagrad
    Theta1 = Theta1 - (alpha ./ sqrt(Theta1accugrad)) .* Theta1grad;
    Theta2 = Theta2 - (alpha ./ sqrt(Theta2accugrad)) .* Theta2grad;
    Theta3 = Theta3 - (alpha ./ sqrt(Theta3accugrad)) .* Theta3grad;
    
    %   Feed Forward on training set
    a1 = [ones(m, 1), X];   %   add intercept terms to the input layer on training set
    [z2, a2, z3, a3, z4, a4] = feedForward(a1, Theta1, Theta2, Theta3, m);  
    
    %   Compute Regularized Cost on training set
    J_History = computeCost(J_History, i, Theta1, Theta2, Theta3, y, a4, m, num_labels, s1, s2, s3, lambda);
    
    %   Feed Forward on testing set
    a1_t = [ones(m_t, 1), X_test];   %   add intercept terms to the input layer on testing set
    [z2_t, a2_t, z3_t, a3_t, z4_t, a4_t] = feedForward(a1_t, Theta1, Theta2, Theta3, m_t);
    
    %   Compute Regularized Cost on testing set
    J_T_History = computeCost(J_T_History, i, Theta1, Theta2, Theta3, y_test, a4_t, m_t, num_labels, s1, s2, s3, lambda); 

    %   Compute accuracy on training set
    Acc_History(i) = mean(double(predictions(X, Theta1, Theta2, Theta3) == y)) * 100;
    
    %   Compute accuracy on testing set
    Acc_T_History(i) = mean(double(predictions(X_test, Theta1, Theta2, Theta3) == y_test)) * 100;

    %   print the training status
    fprintf('Iteration: %d, Cost(Training set): %f, Cost(Testing set): %f, Accuracy(Training set): %f%%, Accuracy(Testing set): %f%%\n', i, J_History(i), J_T_History(i), Acc_History(i), Acc_T_History(i));       
end

end