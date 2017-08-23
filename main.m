%%  Pen-Based Recognition of Handwritten Digits By Regularized Neural Networks
%   Programed by Daniel H. Leung 23/08/2017 (DD/MM/YYYY)
%   For detail, please refer to the comments in codes and README.md

%%  Initialization

clear;
close all;
clc;

%%  Load Data

X = load('in_train.txt');
y = load('out_train.txt');
X_test = load('in_test.txt');
y_test = load('out_test.txt');
lambda = 0.1;   %   for regularization
alpha = 2.5;   %   learning rate
epoch = 500;    %   number of times of batch gradient descent
num_labels = 10;    %   number of labels
s1 = 16;    %   number of nodes in layer 1
s2 = 32;    %   number of nodes in layer 2
s3 = 32;    %   number of nodes in layer 3
s4 = 32;    %   number of nodes in layer 4
s5 = num_labels;    %   number of nodes in layer 5, equals to num of labels

%%  Feature Scaling

[X, mu, stddev] = featureScaling(X);    %   scale the feature by gaussian normalization

%%  Randomize the Initial Weights

Theta1 = randomInitializeWeights(s1, s2);
Theta2 = randomInitializeWeights(s2, s3);
Theta3 = randomInitializeWeights(s3, s4);
Theta4 = randomInitializeWeights(s4, s5);

%%  Feed Forward, Back Propagation

[Theta1, Theta2, Theta3, Theta4, J_History] = FeedForwardBackProp(Theta1, Theta2, Theta3, Theta4, epoch, alpha, lambda, X, y, s1, s2, s3, s4, num_labels);

%%  Display Cost by Iteration and Plot graphs

figure('Name', 'Cost on the Training Data - Num of Iterations');
plot(1:numel(J_History(:, 1)), J_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Cost on the Training Data'));

%%  Predictions to testcase

for i = 1:size(X_test, 2)
    X_test(:, i) = (X_test(:, i) - mu(i)) ./ stddev(i);
end
m = size(X_test, 1);
a1_t = [ones(m, 1), X_test];
z2_t = a1_t * Theta1';
a2_t = [ones(m, 1), sigmoid(z2_t)];
z3_t = a2_t * Theta2';
a3_t = [ones(m, 1), sigmoid(z3_t)];
z4_t = a3_t * Theta3';
a4_t = [ones(m, 1), sigmoid(z4_t)];
z5_t = a4_t * Theta4';
a5_t = softmax(z5_t);
[temp, predict] = max(a5_t, [], 2);   %   find out the index of that max value and save in predict
predict = predict - 1;  %   matlab's indexing starts from 1, but my code considers index 1 as digit 0
fprintf('\nTested by test dataset.\n');
fprintf('Accuracy: %f%%\n', mean(double(predict == y_test)) * 100);
