%% START
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Main script file for the transfer learning network
    which we used for the analusis in the project.
%}
%% CODE

I = imread("C:\Users\umarf\Downloads\PRML\Projects\Term Project\DataSet\New Plant Diseases Dataset(Augmented)\data\train\Strawberry___Leaf_scorch\0c3c9e71-d339-4201-a32a-7a28f0556258___RS_L.Scorch 0930_flipLR.jpg");
imshow(I)

Bin =  imbinarize(I);
%imshow(Bin)

%imshowpair(I,Bin,'montage')
imshow(rgb2gray(I));


%% segmentation

V  = squeeze(I.I);
 fseed = V(:,:,seedLevel) > 75;
 bseed = V(:,:,seedLevel) == 0;
 figure; 
 imshow(fseed)


%% 2
L = imsegkmeans(I,2);
B = labeloverlay(I,L);
imshow(B)


%% 3 thresholding

level = multithresh(I);
seg_I = imquantize(I,level);

imshow(seg_I)


%%  good
threshRGB = multithresh(I,1);

% threshForPlanes = zeros(3,7);			

% for i = 1:2
%     threshForPlanes(i,:) = multithresh(I(:,:,i),7);
% end


value = [0 threshRGB(2:end) 255]; 
quantRGB = imquantize(I, threshRGB, value);

imshow(quantRGB)

%% edge detection

T = rgb2gray(I);

BW1 = edge(T,'Canny');

imshow(BW1)


