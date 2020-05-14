%% START
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: The scrpit file to visualize the FC layers and create a
    iimages which nearly resembles the class with the learned features.
%}
%% CODE
layer = 15;
name = net.Layers(layer).Name
channels = [1];
net.Layers(end).Classes(channels)
I = deepDreamImage(net,name,channels, ...
    'Verbose',false, ...
    'NumIterations',100, ...
    'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])
