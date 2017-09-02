function [z2, a2, z3, a3, z4, a4] = feedForward(a1, Theta1, Theta2, Theta3, m)

z2 = a1 * Theta1';
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(m, 1), sigmoid(z3)];
z4 = a3 * Theta3';
a4 = softmax(z4);
    
end