%% START
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Script file to extract the HOG best cornor features and
    plot them on the image and show
%}
%% CODE
I2 = imread('C:\Users\umarf\Downloads\PRML\Projects\Term Project\DataSet\New Plant Diseases Dataset(Augmented)\data\train\Grape___Black_rot\1b332769-d8c9-4f5b-b21e-171fcc9f939b___FAM_B.Rot 0640.jpg');
corners   = detectFASTFeatures(rgb2gray(I2));
strongest = selectStrongest(corners,100);
[hog2, validPoints,ptVis] = extractHOGFeatures(I2,strongest,'CellSize',[2,2]);
figure;
imshow(I2);
hold on;
plot(ptVis,'Color','blue');

