function J_History = computeCost(J_History, i, Theta1, Theta2, Theta3, y, olayer, m, num_labels, s1, s2, s3, lambda)

%   Compute Regularized Cost
for k = 0:num_labels - 1    %   compute the non-regularized cost
    yOfk = double(y == k);
    J_History(i) = J_History(i) + (1 / m) * sum(((-yOfk) .* log(olayer(:, k + 1))) - (1 - yOfk) .* log(1 - olayer(:, k + 1)));
end
SumOfTheta1_R = sum(sum(Theta1(:, 2:s1 + 1) .^ 2)); %   compute all the terms for regularization
SumOfTheta2_R = sum(sum(Theta2(:, 2:s2 + 1) .^ 2));
SumOfTheta3_R = sum(sum(Theta3(:, 2:s3 + 1) .^ 2)); %   compute all the terms for regularization
J_History(i) = J_History(i) + (lambda / (2 * m)) * (SumOfTheta1_R + SumOfTheta2_R + SumOfTheta3_R); %   regularize the cost

end