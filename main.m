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
alpha = 0.7;   %   learning rate
epoch = 160;    %   number of times of Adagrad
num_labels = 10;    %   number of labels
s1 = 16;    %   number of nodes in layer 1
s2 = 32;    %   number of nodes in layer 2
s3 = num_labels;    %   number of nodes in layer 3

%%  Feature Scaling

[X, mu, stddev] = featureScaling(X);    %   scale the feature by gaussian normalization

%%  Randomize the Initial Weights

Theta1 = randomInitializeWeights(s1, s2);
Theta2 = randomInitializeWeights(s2, s3);

%%  Feed Forward, Back Propagation

[Theta1, Theta2, J_History, Acc_History] = FeedForwardBackProp(Theta1, Theta2, epoch, alpha, lambda, X, y, s1, s2, num_labels, X_test, mu, stddev, y_test);

%%  Display Cost by Iteration and Plot graphs

%   Graph of Cost - Iterations
figure('Name', 'Cost on the Training Data - Num of Iterations');
plot(1:numel(J_History(:, 1)), J_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Cost on the Training Data'));

%   Graph of Accuracy - Iterations
figure('Name', 'Accuracy on the Testing Data - Num of Iterations');
plot(1:numel(Acc_History(:, 1)), Acc_History(:, 1), '-b', 'LineWidth', 2);
xlabel('Num of Iterations');
ylabel(strcat('Accuracy on the Testing Data'));
