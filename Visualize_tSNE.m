%% Umar
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: script file to generate the visualization of the
    actiivations of the fully connected layer for the mutidimensional
    arrays using the t-SNE visualization.
%}
%% Start
finalConvActivations = activations(net2,val,"fc_2","OutputAs","rows");
finalConvtsne = tsne(finalConvActivations);

doLegend = 'on';
markerSize = 10;
figure;
gscatter(finalConvtsne(:,1),finalConvtsne(:,2),val.Labels, ...
    [],'.',markerSize,doLegend);
title("Final FC layer activations");
%% END