%% START
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Script file to extract the color histogram features from
    the images of dataset.
%}
%% CODE
img = imread('C:\Users\umarf\Downloads\PRML\Projects\Term Project\DataSet\PlantVillage-Dataset-master\PlantVillage-Dataset-master\raw\color\Potato___Early_blight\0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.jpg');

image = rgb2hsv(img);
 %Split into RGB Channels
    Red = image(:,:,1);
    Green = image(:,:,2);
    Blue = image(:,:,3);
    %Get histValues for each channel
    [yRed, x] = imhist(Red);
    [yGreen, x] = imhist(Green);
    [yBlue, x] = imhist(Blue);
    %Plot them together in one plot
    plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
