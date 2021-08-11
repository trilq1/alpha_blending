function x = main()

    numCameras = 8;
    bevImgs = cell(1, numCameras);
    
    for i = 2:3
        bevImgs{i} = imread('../data/test' + string(i) + '.jpg');
    end

    % Combine the first four images to get the stitched leftSideview and the
    % spatial reference object Rleft.
    
    % Note: radius could be tuned to find the sweetspot because blending
    % also depends on the order of blending???

    radius = 10;
    leftImgs = bevImgs(2:3);
    tforms = helperRegisterImages(leftImgs, radius);
    [leftSideView, Rleft] = helperStitchImages(leftImgs, tforms);
    
%     radius =  10;
%     % Combine the last four images to get the stitched rightSideView.
%     rightImgs = bevImgs(5:8);
%     tforms = helperRegisterImages(rightImgs, radius);
%     [rightSideView, Rright] = helperStitchImages(rightImgs, tforms);

    % Combine the two side views to get the 360° bird's-eye-view in
    % surroundView and the spatial reference object Rsurround
%     radius = 60;
%     imgs = {leftSideView, rightSideView};
%     tforms = helperRegisterImages(imgs, radius);
%     [surroundView, Rsurround] = helperStitchImages(imgs, tforms);
%     figure
%     imshow(surroundView);

    x = 1;

end