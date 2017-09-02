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
alpha = 0.25;   %   learning rate
epoch = 200;    %   number of times of Adagrad
num_labels = 10;    %   number of labels
s1 = 16;    %   number of nodes in layer 1
s2 = 32;    %   number of nodes in layer 2
s3 = 32;    %   number of nodes in layer 3
s4 = num_labels;    %   number of nodes in layer 4

%%  Feature Scaling

%   Scale the feature by gaussian normalization
[X, mu, stddev] = featureScaling(X);

%   Scale the testing data by using the mean and standard deviation of training data
for k = 1:size(X_test, 2)
    X_test(:, k) = (X_test(:, k) - mu(k)) ./ stddev(k);
end

%%  Randomize the Initial Weights

Theta1 = randomInitializeWeights(s1, s2);
Theta2 = randomInitializeWeights(s2, s3);
Theta3 = randomInitializeWeights(s3, s4);

%%  Feed Forward, Back Propagation

[Theta1, Theta2, Theta3, J_History, J_T_History, Acc_History, Acc_T_History] = FeedForwardBackProp(Theta1, Theta2, Theta3, epoch, alpha, lambda, X, y, s1, s2, s3, num_labels, X_test, mu, stddev, y_test);

%%  Display Cost by Iteration and Plot graphs

%   Graph of Cost(Training Data) - Iterations
figure('Name', 'Cost on the Training Data - Num of Iterations');
plot(1:numel(J_History(:, 1)), J_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Cost on the Training Data'));

%   Graph of Cost(Testing Data) - Iterations
figure('Name', 'Cost on the Testing Data - Num of Iterations');
plot(1:numel(J_T_History(:, 1)), J_T_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Cost on the Testing Data'));

%   Graph of Accuracy(Training Data) - Iterations
figure('Name', 'Accuracy on the Training Data - Num of Iterations');
plot(1:numel(Acc_History(:, 1)), Acc_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Accuracy on the Training Data'));

%   Graph of Accuracy(Testing Data) - Iterations
figure('Name', 'Accuracy on the Testing Data - Num of Iterations');
plot(1:numel(Acc_T_History(:, 1)), Acc_T_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Accuracy on the Testing Data'));
