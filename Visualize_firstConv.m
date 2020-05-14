%% Umar
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Script file to  visualize the convolition layers of the
    trained networks.
%}
%% Start
layer = 9;
name = net.Layers(layer).Name;

channels = 1:30; %% this number changes for the different networks depending
                    % on the number of the layers and filter sizes we used.
I = deepDreamImage(net,name,channels, ...
    'PyramidLevels',1);

figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')

%% END