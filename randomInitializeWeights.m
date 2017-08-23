function weights = randomInitializeWeights(sInput, sOutput)

%   Compute the randomized weights
epsilon = sqrt(6) / sqrt(sInput + sOutput);
weights = rand(sOutput, 1 + sInput) .* 2 .* epsilon - epsilon;

end