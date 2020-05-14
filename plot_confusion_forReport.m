%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: The script file to plot the confusion matrices for train,
    validation & test data for the network.
    Accuuracy/ loss curve
    And analyze the network we trained
%}
%% STart

% conmattrain = confusionmatrix(test.Labels,YTest,'confusion matrix for training data');
% confusionchart(conmattrain);

% Test on the validation data


YTrain = classify(net2,train);
train_acc = mean(YTrain==train.Labels)

YVal = classify(net2,val);
val_acc = mean(YVal==val.Labels)

% Test on the Testing data
YTest = classify(net2,test);
test_acc = mean(YTest==test.Labels)

figure(1)
plotconfusion(train.Labels,YTrain);
set(findobj(gca,'type','text'),'fontsize',6)
title('confusion matrix on train data');


figure(2)
plotconfusion(val.Labels,YVal);
set(findobj(gca,'type','text'),'fontsize',6)
title('confusion matrix on validation data');


figure(3)
plotconfusion(test.Labels,YTest);
set(findobj(gca,'type','text'),'fontsize',6)
title('confusion matrix on test data');


figure(4)
plotTrainingAccuracy_All(info1,numEpochs);


analyzeNetwork(net1);
%%END
