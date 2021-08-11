function [outputImage, outputView] = helperStitchImages(images, tforms) 
    
    numImages = numel(images);
    imageSize = zeros(numImages,2);
    xlim = zeros(numImages,2);
    ylim = zeros(numImages,2);
    
    % Compute the output limits for each transform.
    for i = 1:numel(images)
        imageSize(i,:) = size(images{i}, 1:2);
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), ...
                                              [1 imageSize(i,2)], ...
                                              [1 imageSize(i,1)]);
    end    

    % Find the minimum and maximum output limits.
    maxImageSize = max(imageSize);
    
    xMin = min([1; xlim(:)]);
    xMax = max([maxImageSize(2); xlim(:)]);

    yMin = min([1; ylim(:)]);
    yMax = max([maxImageSize(1); ylim(:)]);

    % Width and height of panorama.
    width  = round(xMax - xMin);
    height = round(yMax - yMin);

    % Initialize the "empty" panorama.
    outputImage = zeros([height width 3], "like", images{1});

    % Create a 2-D spatial reference object defining the size of the panorama.
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    outputView = imref2d([height width], xLimits, yLimits);

    % Step 7 - Stitch the images.
    for i = 1:numel(tforms)
        % Apply transformation.
        warpedImage = imwarp(images{i}, tforms(i), "OutputView", outputView);
        
        % Blend the images.
        outputImage = helperBlendImages(warpedImage, outputImage);
    end

end