% An example script to compute the BF measure desrcribed in:
% SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. PAMI, 2017.
% Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla, University of Cambridge

% Please download benchmark BSDS code from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

%% Ground truth
gtroot = '/home/sunrgb_360x480/testannot_480x360/'; %In Index format i.e each pixel takes a value between 1:N classes
gt = dir(gtroot);

%% predictions
predroot = '/pa/adv_dev/DeepSegmentation/pami_submission/Dropout/sunrgb_enc_4x_Expdilation_80K/';
pp = dir(predroot);

numImages = numel(gt) - 2;
numClasses = 37;%SUNRGBD segmentation classes
maxDist = 0.0075; %Threshold for BF measure

% parallel compute
cntR = zeros(numImages,numClasses,1);
sumR = zeros(numImages,numClasses,1);
cntP = zeros(numImages,numClasses,1);
sumP = zeros(numImages,numClasses,1);
precision = zeros(numImages,numClasses,1);
recall    = zeros(numImages,numClasses,1);
F1_measure = zeros(numImages,numClasses,1);
F1_measure_im = zeros(numImages,numClasses);
avg_BF_measure = 0;
BF_measure = zeros(numImages,1);%See "What is a good evaluation measure for semantic segmentation?", Csurka & Perronin's BMVC 2013 
classcount = zeros(numImages,numClasses,1);

%% Boundary evaluation using BSDS code to compute F-measure
parfor i = 3:numel(gt)% Need parallel programming toolbox to speed up computation
    display(num2str(i));%   
    g = [gtroot gt(i).name];
    p = [predroot pp(i).name];
        
    gim = imread(g);   
    pim = imread(p);
    pim = imresize(pim,[360 480],'nearest');%compute accuracy in appropriate res     
   
    for n = 1:numClasses%for each class
        In = gim == n;        
        bmap_gim = logical(seg2bdry(In,'imageSize'));% Warning! This function which produces a binary edge map from each classes segment seems outdated and replaced by seg2bmap (not tested here). See https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
        In = pim == n;
        bmap_pim = logical(seg2bdry(In,'imageSize'));%  
    
        % compute the correspondence
        [match1,match2] = correspondPixels(double(bmap_pim),double(bmap_gim), maxDist); %See https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
        
        % compute recall
        sumR(i-2,n) = sum(bmap_gim(:));
        cntR(i-2,n) = sum(match2(:)>0); %pixels in the ground truth which have a correspondence in the predicted contour
        if sumR(i-2,n) > 0
            classcount(i-2,n) = 1;
            recall(i-2,n) = cntR(i-2,n)/sumR(i-2,n);
        end
        
        % compute precision
        sumP(i-2,n) = sum(bmap_pim(:));
        cntP(i-2,n) = sum(match1(:)>0);  %pixels in the contour which have a correspondence in the ground truth boundary
        if sumP(i-2,n) > 0
            classcount(i-2,n) = 1;
            precision(i-2,n) = cntP(i-2,n)/sumP(i-2,n);
        end
        
        %Compute F1 measure of boundary delineation quality
        if precision(i-2,n)+recall(i-2,n) > 0
            F1_measure(i-2,n) = 2*precision(i-2,n)*recall(i-2,n)/(precision(i-2,n)+recall(i-2,n)); %Berkeley contour metric
            %F1_measure_im(i-2,n) = 2*precision(i-2,n)*recall(i-2,n)/(precision(i-2,n)+recall(i-2,n)); %Berkeley contour metric for each image
        end       
        
    end
        
end

%Compute for full dataset
for i = 3:numel(gt)
    sumc = 0;
    sumf = 0;     
    for n = 1:numClasses
        sumc = sumc + classcount(i-2,n); %only classes which are present in that ground truth image (see Csurka & Perronin's BMVC 2013)
        sumf = sumf + F1_measure(i-2,n);
    end
    if sumc > 0
        BF_measure(i-2) = sumf/sumc;
        avg_BF_measure = avg_BF_measure + BF_measure(i-2);        
    end  
    
end

display('Avg F1 measure = ');
display(num2str(avg_BF_measure./(numel(gt)-2)));



