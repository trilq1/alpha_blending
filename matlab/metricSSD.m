function scores = metricSSD(features1, features2, N1, N2, outputClass)
%metricSSD Calculating the Sum of Squared Differences between two groups of features

%   Copyright 2020 The MathWorks, Inc.

%#codegen

% Need to obtain feature vector length to perform explicit row indexing
% needed for code generation of variable sized inputs

if isempty(coder.target)
    % call optimized builtin function
    scores = zeros(N1, N2, outputClass);
    scores(:,:) = visionSSDMetric(features1,features2);
else
    if coder.internal.isTargetMATLABHost
        features1 = features1';
        features2 = features2';
        scores = vision.internal.buildable.ComputeMetricBuildable.ComputeMetric_core(features1,features2, 'ssd', N1, N2, outputClass);
    else
        % for portable C code generation
        vector_length = size(features1, 1);
        scores = zeros(N1, N2, outputClass);

        for c = 1:N2
            for r = 1:N1
                scores(r, c) = sum((features1(1:vector_length, r) - ...
                    features2(1:vector_length, c)).^2);
            end
        end
    end
end