function tforms = helperRegisterImages(images, radius)
    
    params.Radius         = radius;
    params.MatchThreshold = 10;
    params.MaxRatio       = 0.6;
    params.Confidence     = 99.9;
    params.MaxDistance    = 2;
    params.MaxNumTrials   = 2000;
    
    numImages = numel(images);
    
    % Store points and features for all images.
    features = cell(1,numImages);
    points   = cell(1,numImages);
    for i = 1:numImages
        grayImage = rgb2gray(images{i});
        points{i} = detectKAZEFeatures(grayImage);
%         imshow(grayImage);
%         hold on;
%         plot(selectStrongest(points{i}, 1000));
%         hold off;
%         pause;
        [features{i}, points{i}] = extractFeatures(grayImage, points{i});
    end

    % Initialize all the transforms to the identity matrix.
    tforms(numImages) =  affine2d(eye(3));
    
    % Set the seed for reproducibility.
    rng(0);
    
    % Find the relative transformations between each image pair.
    for i = 2:numImages
        % Find correspondences between images{i} and images{i-1} using
        % constrained feature matching.
        indexPairs = matchFeaturesInRadius(features{i-1}, features{i}, points{i}.Location, ...
                                           points{i-1}.Location, params.Radius, ...
                                           "MatchThreshold", params.MatchThreshold, ...
                                           "MaxRatio", params.MaxRatio);
        
        %  Estimate the transformation between images{i} and images{i-1}.
        matchedPointsPrev = points{i-1}(indexPairs(:,1), :);    
        matchedPoints     = points{i}(indexPairs(:,2), :);
        
        % Matching 
        figure; ax = axes;
        showMatchedFeatures(images{i-1},images{i},matchedPointsPrev,matchedPoints,'Parent',ax);
        title(ax, 'Putative point matches');
        legend(ax,'Matched points 1','Matched points 2');
        
        tforms(i) = estimateGeometricTransform2D(matchedPoints, matchedPointsPrev,"similarity",...
                                                 "Confidence" , params.Confidence, ...
                                                 "MaxDistance", params.MaxDistance, ...
                                                 "MaxNumTrials",params.MaxNumTrials);
        
        % Compute the transformation that maps images{i} to the stitched
        % image as T(i)*T(i-1)*...*T(1).
        tforms(i).T = tforms(i).T*tforms(i-1).T;

    end
end