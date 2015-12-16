% if you use another test/train set change number of classes and the
% unlabeled index as well as number of iterations (needs to be equal to the test set size)

gtPath = '/SegNet/Scripts/gt'; % path to your ground truth images
predPath = '/SegNet/Scripts/predictions'; %path to your predictions (you get them after you implement saving images in the test_segmentation_camvid.py script - or you write your own)
groundTruths = dir(gtPath);
skip = 2; % first two are '.' and '..' so skip them
predictions = dir(predPath);

iter = 233;

numClasses = 11;
unknown_class = 12;

totalpoints = 0;
cf = zeros(iter,numClasses,numClasses);
globalacc = 0;

for i = 1:iter
    display(num2str(i));

    pred = imread(strcat(predPath, '/', predictions(i + skip).name)); % set this to iterate through your segnet prediction images
    pred = pred + 1; % i added this cause i labeled my classes from 0 to 11
    annot = imread(strcat(gtPath, '/', groundTruths(i + skip).name)); % set this to iterate through your ground truth annotations
    annot = annot + 1; % i added this cause i labeled my classes from 0 to 11 -> so in that case the next line will find every pixel labeled with unknown_class=12

    pixels_ignore = annot == unknown_class;
    pred(pixels_ignore) = 0;
    annot(pixels_ignore) = 0;
   
    totalpoints = totalpoints + sum(annot(:)>0);

    % global and class accuracy computation
    for j = 1:numClasses
        for k = 1:numClasses
                c1  = annot == j;
                c1p = pred == k;
                index = gather(c1 .* c1p);              
                cf(i,j,k) = cf(i,j,k) + sum(index(:));
        end
            c1  = annot == j;
            c1p = pred == j;
            index = gather(c1 .* c1p);
            globalacc = globalacc + sum(index(:));
        
    end
end

cf = sum(cf,1);
cf = squeeze(cf);

% Compute confusion matrix
conf = zeros(numClasses);
for i = 1:numClasses
    if i ~= unknown_class && sum(cf(i,:)) > 0
        conf(i,:) = cf(i,:)/sum(cf(i,:));
    end
end
globalacc = sum(globalacc)/sum(totalpoints);

% Compute intersection over union for each class and its mean
intoverunion = zeros(numClasses,1);
for i = 1:numClasses
    if i ~= unknown_class   && sum(conf(i,:)) > 0
        intoverunion(i) = (cf(i,i))/(sum(cf(i,:))+sum(cf(:,i))-cf(i,i));
    end
end

display([' Global acc = ' num2str(globalacc) ' Class average acc = ' num2str(sum(diag(conf))/(numClasses)) ' Mean Int over Union = ' num2str(sum(intoverunion)/(numClasses))]);
