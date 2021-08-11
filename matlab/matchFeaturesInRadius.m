function [indexPairs, matchMetric] = matchFeaturesInRadius(features1, ...
    features2, points2, centerPoints, radius, varargin)
%matchFeaturesInRadius Match features within a radius
%   indexPairs = matchFeaturesInRadius(features1, features2, points2,
%   centerPoints, radius) returns a P-by-2 matrix, indexPairs, containing
%   the indices to the features most likely to correspond between the two
%   input feature matrices satisfying spatial constraints specified by
%   centerPoints and radius.
%
%   Inputs
%   ------
%   features1          An M1-by-N matrix or a binaryFeatures object
%                      specifying the features in the first image
%
%   features2          An M2-by-N matrix or a binaryFeatures object
%                      specifying the features in the second image
%
%   points2            Feature points corresponding to features2 specified
%                      as an M2-by-2 matrix of [x, y] coordinates or an
%                      M2-element <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'pointfeaturetypes')">feature point</a> array.
%
%   centerPoints       An M1-by-2 matrix of [x, y] coordinates specifying
%                      the locations in the second image that are expected
%                      to correspond to the feature points associated with
%                      features1 in the first image
%
%   radius             A positive scalar or an M1-element array representing
%                      the searching radius associated with centerPoints
%
%   [indexPairs, matchMetric] = matchFeaturesInRadius(...) also returns
%   the metric values that correspond to the associated features indexed
%   by indexPairs in a P-by-1 matrix matchMetric.
%
%   [indexPairs, matchMetric] = matchFeaturesInRadius(..., Name, Value)
%   specifies additional name-value pairs described below:
%
%   'MatchThreshold'   A scalar T, 0 < T <= 100, that specifies the
%                      distance threshold required for a match. A pair of
%                      features are not matched if the distance between
%                      them is more than T percent from a perfect match.
%                      Increase T to return more matches.
%
%                      Default: 10.0 for binary feature vectors
%                                1.0 otherwise
%
%   'MaxRatio'         A scalar R, 0 < R <= 1, specifying a ratio threshold
%                      for rejecting ambiguous matches. Increase R to
%                      return more matches.
%
%                      Default: 0.6
%
%   'Metric'           A string used to specify the distance metric. This
%                      parameter is not applicable when features1 and
%                      features2 are binaryFeatures objects.
%
%                      Possible values are:
%                        'SAD'         : Sum of absolute differences
%                        'SSD'         : Sum of squared differences
%
%                      Default: 'SSD'
%
%                      Note: When features1 and features2 are
%                            binaryFeatures objects, Hamming distance is
%                            used to compute the similarity metric.
%
%   'Unique'           A logical scalar. Set this to true to return only
%                      unique matches between features1 and features2.
%
%                      Default: false
%
%   Notes
%   -----
%   - Use this function when the 3-D world points corresponding to features1
%     are known. centerPoints can be obtained by projecting the 3-D world
%     point onto the second image. The 3-D world points can be obtained by
%     triangulating matched image points from two stereo images.
%
%   - For each [x, y] location in centerPoints, points2 located within a
%     radius of this location are searched. The corresponding feature
%     in features1 is then matched with the features of these points.
%
%   - The range of values of matchMetric varies as a function of the feature
%     matching metric being used. Prior to computation of SAD and SSD
%     metrics, the feature vectors are normalized to unit vectors using the
%     L2 norm. The table below summarizes the metric ranges and perfect
%     match values:
%
%     Metric      Range                            Perfect Match Value
%     ----------  -------------------------------  -------------------
%     SAD         [0, 2*sqrt(size(features1, 2))]          0
%     SSD         [0, 4]                                   0
%     Hamming     [0, features1.NumBits]                   0
%
%   Class Support
%   -------------
%   features1 and features2 can be logical, int8, uint8, int16, uint16,
%   int32, uint32, single, double, or binaryFeatures object. points2 can
%   be single, double, or any point feature type. centerPoints and radius
%   can be single or double.
%
%   The output class of indexPairs is uint32. matchMetric is double when
%   features1 and features2 are double. Otherwise, it is single.
%
%   Example 1: Match features between two images
%   --------------------------------------------
%   % Load image and camera data
%   data = load('matchInRadiusData.mat');
%
%   % Convert camera pose to extrinsics
%   orientation  = data.cameraPose2.Rotation;
%   location     = data.cameraPose2.Translation;
%   [rotationMatrix, translationVector] = cameraPoseToExtrinsics(orientation, location);
%
%   % Project the 3-D world points associated with features1 onto the
%   % second image
%   centerPoints = worldToImage(data.intrinsics, rotationMatrix, ...
%       translationVector, data.worldPoints);
%
%   % Match features with spatial constraints
%   indexPairs1 = matchFeaturesInRadius(data.features1, data.features2, ...
%       data.points2, centerPoints, data.radius, 'MatchThreshold', 40, ...
%       'MaxRatio', 0.9);
%
%   % Compare with matching features without spatial constraints
%   indexPairs2 = matchFeatures(data.features1, data.features2, ...
%       'MatchThreshold', 40, 'MaxRatio', 0.9);
%
%   % Visualize the results
%   figure
%   subplot(2, 1, 1)
%   showMatchedFeatures(data.I1, data.I2, data.points1(indexPairs1(:,1)), ...
%       data.points2(indexPairs1(:,2)));
%   title(sprintf('%d pairs matched with spatial constraints', ...
%       size(indexPairs1, 1)));
%
%   subplot(2, 1, 2)
%   showMatchedFeatures(data.I1, data.I2, data.points1(indexPairs2(:,1)), ...
%       data.points2(indexPairs2(:,2)));
%   title(sprintf('%d pairs matched without spatial constraints', ...
%       size(indexPairs2, 1)));
%
%   Example 2: Monocular Visual Simultaneous Localization and Mapping
%   -----------------------------------------------------------------
%   % This example shows how to process image data from a monocular camera
%   % to build a map of an indoor environment and estimate the trajectory
%   % of the camera simultaneously.
%   % <a href="matlab:web(fullfile(docroot, 'vision/ug/monocular-visual-simultaneous-localization-and-mapping.html'))">View example</a>
%
%   See also matchFeatures, showMatchedFeatures, detectHarrisFeatures,
%       detectFASTFeatures, detectMinEigenFeatures, detectBRISKFeatures,
%       detectSURFFeatures, detectMSERFeatures, detectORBFeatures,
%       extractFeatures, binaryFeatures, estimateWorldCameraPose, worldToImage.

%   Copyright 2020 The MathWorks, Inc.
%
% References
% ----------
% [1] Fraundorfer, Friedrich, and Davide Scaramuzza. "Visual odometry: Part
%     ii: Matching, robustness, optimization, and applications."
%     IEEE Robotics & Automation Magazine 19, no. 2 (2012): 78-90.
%
% [2] David Lowe, "Distinctive image features from scale-invariant keypoints",
%     International Journal of Computer Vision, 60, 2 (2004)
%
% [3] Marius Muja and David G. Lowe, "Fast Approximate Nearest Neighbors
%     with Automatic Algorithm Configuration", in International Conference
%     on Computer Vision Theory and Applications (VISAPP'09), 2009
%
% [4] Marius Muja, David G. Lowe: "Fast Matching of Binary Features".
%     Conference on Computer and Robot Vision (CRV) 2012.

%#codegen

% Parse and check inputs
[features1, features2, centerPoints, points2, radius, metric, matchThreshold, ...
    maxRatio, uniqueMatch, outputClass, numPoints1, numPoints2] = parseInputs(...
    features1, features2, centerPoints, points2, radius, varargin{:});

vision.internal.matchFeatures.checkFeatureConsistency(features1, features2);

% Convert match threshold percent to a numeric threshold
matchThreshold = vision.internal.matchFeatures.percentToLevel( ...
    matchThreshold, size(features1, 2), metric, outputClass);

% Use transposed feature vectors to compute distance
features1 = features1';
features2 = features2';

% Cast and normalize non-binary features
if ~strcmp(metric, 'hamming')
    features1 = cast(features1, outputClass);
    features2 = cast(features2, outputClass);
    
    % Convert feature vectors to unit vectors
    features1 = vision.internal.matchFeatures.normalizeFeature(features1);
    features2 = vision.internal.matchFeatures.normalizeFeature(features2);
end


% Find feature points within the radius
allSpatialDist = vision.internal.matchFeatures.metricSSD( ...
    centerPoints', points2', numPoints1, numPoints2, class(centerPoints));
isInRadius     = allSpatialDist <= repmat(radius.^2, 1, numPoints2);

% Find strong matches
matchScores    = vision.internal.matchFeatures.exhaustiveDistanceMetrics(...
    features1, features2, numPoints1, numPoints2, outputClass, metric);

isStrongMatch  = matchScores <= matchThreshold;
disp(matchThreshold);
disp(max(matchScores, [], 'all')); % DEBUG

indexPairs     = zeros(numPoints1, 2, 'uint32');
ratioCheck     = maxRatio ~= 1;
for centerIdx  = 1:numPoints1
    neighborIdxLogical  = isInRadius(centerIdx, :);
    candidateScores     = matchScores(centerIdx, neighborIdxLogical);
    neighborIdxLinear   = find(neighborIdxLogical);
    % disp(neighborIdxLinear);
    if numel(neighborIdxLinear) == 1
        matchIndex = neighborIdxLinear(1);
        
    elseif numel(neighborIdxLinear) > 1
        
        % Pick the best two candidates within the radius
        [topTwoMetrics, topTwoIndices] = vision.internal.partialSort(candidateScores, 2, 'ascend');
        matchIndex = neighborIdxLinear(topTwoIndices(1));
%         disp(topTwoIndices);
%         disp(topTwoIndices(1));
%         disp("===========")
        % Perform ratio test
        if ratioCheck
            if  topTwoMetrics(2) < cast(1e-6, outputClass)
                ratio = cast(1, outputClass);
            else
                ratio = topTwoMetrics(1) /topTwoMetrics(2);
            end
            
            if ratio > maxRatio
                continue % Ambiguous match
            end
        end
    else
        matchIndex = 0; % For codegen
        continue % No match found
    end
    
    % Verify if the match is strong
    if isStrongMatch(centerIdx, matchIndex)
        indexPairs(centerIdx, :) = [centerIdx, matchIndex];
    end
end

isPointMatched = indexPairs(:, 1) ~= uint32(0);
% disp(size(indexPairs));
indexPairs     = indexPairs(isPointMatched, :);
% disp(size(indexPairs));
% disp(indexPairs)
% Check if the match is unique
if uniqueMatch && any(isPointMatched)
    [~, idx]      = min(matchScores(:,indexPairs(:, 2)));
    uniqueIndices = idx' == indexPairs(:, 1);
    indexPairs    = indexPairs(uniqueIndices, :);
    disp("match!");
end

% Get the corresponding match score
matchMetric = matchScores(sub2ind([numPoints1, numPoints2], indexPairs(:, 1), indexPairs(:,2)));
end

%--------------------------------------------------------------------------
% Parse and check inputs
%--------------------------------------------------------------------------
function [features1, features2, centerPoints, points2, radius, metric, ...
    matchThreshold, maxRatio, uniqueMatch, outputClass, numPoints1, numPoints2] = ...
    parseInputs(featuresIn1, featuresIn2, centerPoints, points2, radius, varargin)

fileName = 'matchFeaturesInRadius';

vision.internal.inputValidation.checkFeatures(featuresIn1, fileName, 'features1');
vision.internal.inputValidation.checkFeatures(featuresIn2, fileName, 'features2');

coder.internal.errorIf(~isequal(class(featuresIn1), class(featuresIn2)),...
    'vision:matchFeatures:featuresNotSameClass');

isBinaryFeature = isa(featuresIn1, 'binaryFeatures');

if isBinaryFeature
    features1   = featuresIn1.Features;
    features2   = featuresIn2.Features;
else
    features1   = featuresIn1;
    features2   = featuresIn2;
end

% Determine output class
if (isa(features1, 'double'))
    outputClass = 'double';
else
    outputClass = 'single';
end

numPoints1 = size(features1, 1);
numPoints2 = size(features2, 1);

points2      = checkPoints2(points2, numPoints2);
checkCenterPoints(centerPoints, numPoints1);
centerPoints = cast(centerPoints, 'like', points2);
radius       = checkRadius(radius, numPoints1);

isRadiusSearch  = true;
defaultParams   = vision.internal.matchFeatures.getDefaultParameters(isBinaryFeature, isRadiusSearch);

% Parse inputs
if isempty(coder.target)  % Simulation
    [metricTemp, matchThreshold, maxRatio, uniqueMatch] = ...
        parseOptionalInputsSimulation(defaultParams, varargin{:});
else % Code generation
    [metricTemp, matchThreshold, maxRatio, uniqueMatch] = ...
        parseOptionalInputsCodegen(defaultParams, varargin{:});
end

vision.internal.matchFeatures.checkMatchThreshold(matchThreshold, fileName);
vision.internal.matchFeatures.checkMaxRatioThreshold(maxRatio, fileName);
vision.internal.matchFeatures.checkMetric(metricTemp, fileName);
vision.internal.matchFeatures.checkUniqueMatches(uniqueMatch, fileName);

if isBinaryFeature
    metric = 'hamming';
else
    metric = lower(metricTemp);
end
end

%--------------------------------------------------------------------------
function [metric, matchThreshold, maxRatio, uniqueMatch] = ...
    parseOptionalInputsSimulation(defaultParams, varargin)
%parseOptionalInputsSimulation Parse parameters for simulation workflow

% Setup parser
parser = inputParser;
parser.addParameter('MatchThreshold', defaultParams.MatchThreshold);
parser.addParameter('MaxRatio', defaultParams.MaxRatio);
parser.addParameter('Metric', defaultParams.Metric);
parser.addParameter('Unique', defaultParams.Unique);

% Parse input
parser.parse(varargin{:});
r = parser.Results;

matchThreshold  = r.MatchThreshold;
maxRatio        = r.MaxRatio;
metric          = r.Metric;
uniqueMatch     = r.Unique;
end

%--------------------------------------------------------------------------
function [metric, matchThreshold, maxRatio, uniqueMatch] = ...
    parseOptionalInputsCodegen(defaultParams, varargin)
%parseOptionalInputsCodegen Parse parameters for codegen workflow

% Set parser inputs
params = struct( ...
    'Metric',                uint32(0), ...
    'MatchThreshold',        uint32(0), ...
    'MaxRatio',              0, ...
    'Unique',                false);

popt = struct( ...
    'CaseSensitivity', false, ...
    'StructExpand',    true, ...
    'PartialMatching', true);

optarg = eml_parse_parameter_inputs(params, popt, varargin{:});

metric = eml_get_parameter_value(optarg.Metric, ...
    defaultParams.Metric, varargin{:});
matchThreshold = eml_get_parameter_value(optarg.MatchThreshold, ...
    defaultParams.MatchThreshold, varargin{:});
maxRatio = eml_get_parameter_value(optarg.MaxRatio, ...
    defaultParams.MaxRatio, varargin{:});
uniqueMatch = eml_get_parameter_value(optarg.Unique, ...
    defaultParams.Unique, varargin{:});
end

%--------------------------------------------------------------------------
function checkCenterPoints(points, numPoints)
checkImagePointsWithKnownSize(points, numPoints, 'centerPoints')
end

%--------------------------------------------------------------------------
function pointsOut = checkPoints2(pointsIn, numPoints)
coder.internal.errorIf( ~isnumeric(pointsIn)...
    && ~vision.internal.inputValidation.isValidPointObj(pointsIn), ...
    'vision:points:ptsClassInvalid', 'points2');

if ~isnumeric(pointsIn)
    pointsOut = pointsIn.Location;
else
    pointsOut = pointsIn; % For codegen
end

checkImagePointsWithKnownSize(pointsOut, numPoints, 'points2')
end

function checkImagePointsWithKnownSize(imagePoints, numPoints, varName)
validateattributes(imagePoints, {'numeric'}, ...
    {'nonsparse', 'real', 'size', [numPoints, 2]}, ...
    'matchFeaturesInRadius', varName);
end

%--------------------------------------------------------------------------
function radius = checkRadius(r, numPoints)
if isscalar(r)
    validateattributes(r, {'single', 'double'}, ...
        {'nonnan', 'nonsparse', 'real', 'positive'}, ...
        'matchFeaturesInRadius', 'radius');
    radius = repmat(r, numPoints, 1);
else
    validateattributes(r, {'single', 'double'}, ...
        {'vector', 'nonnan', 'nonsparse', 'real', 'positive', 'numel', numPoints}, ...
        'matchFeaturesInRadius', 'radius');
    radius = r(:);
end
end