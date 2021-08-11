function outputImage = helperBlendImages(I1, I2) 
    arguments
        I1 uint8
        I2 uint8
    end
    % Identify the image regions in the two images by masking out the black
    % regions.
    mask1 = sum(I1, 3) ~= 0;
    mask2 = sum(I2, 3) ~= 0;
    maskc = mask1 & mask2;
    
    % Compute alpha values that are proportional to the center seam of the two
    % images.
    alpha1 = ones(size(mask1,1:2));
    alpha2 = ones(size(mask2,1:2));
    dist1  = bwdist(edge(mask1));
    dist2  = bwdist(edge(mask2));
    alpha1(maskc) = double(dist1(maskc) > dist2(maskc));    
    alpha2(maskc) = double(dist1(maskc) <= dist2(maskc));
    
    I1 = double(I1);
    I2 = double(I2);        
    outputImage = alpha1.*I1 + alpha2.*I2;    
    outputImage = uint8(outputImage);
%     imshow(outputImage);
%     pause;
    
end