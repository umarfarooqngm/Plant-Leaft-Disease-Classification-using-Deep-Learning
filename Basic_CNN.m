%% START
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Main script file training the network with the simple CNN
    architecture.
%}
%% CODE
addpath 'C:\Users\umarf\OneDrive\Documents\MATLAB\Examples\R2019b\nnet\TransferLearningUsingGoogLeNetExample'

addpath '/storage/home/ubm5020/Documents/MATLAB/Examples/R2019a/nnet/TransferLearningUsingGoogLeNetExample/'
dataDir= './data/';
%checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
%Symmetry_Groups = {'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___healthy','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___healthy'};
Symmetry_Groups = {'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'};
train_folder = 'train';
valid_folder  = 'valid';
% uncomment after you create the augmentation dataset
 %train_folder = 'train_rgb';
 %test_folder  = 'test_rgb';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource','foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
%[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

[train_val, test] = splitEachLabel(train_all,.8);

[train, val] =  splitEachLabel(train_val,.8);


rng('default');
numEpochs = 5; % 5 for both learning rates
batchSize = 100;
nTraining = length(train.Labels);

% Define the Network Structure, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 
layers = [
    imageInputLayer([256 256 3]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,70,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    batchNormalizationLayer;
    reluLayer();  % ReLU layer
    convolution2dLayer(5,30,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(3,50,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();
    maxPooling2dLayer(2,'Stride',2); 
    convolution2dLayer(3,40,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();
    maxPooling2dLayer(2,'Stride',1); 
    
    fullyConnectedLayer(25); % Fullly connected layer with 50 activations
    dropoutLayer(.25); % Dropout layer
    fullyConnectedLayer(38); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];
options = trainingOptions('adam','MaxEpochs',20,... 
    'InitialLearnRate',1e-5,...% learning rate
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

YTrain = classify(net,train);
train_acc = mean(YTrain==train.Labels)

YVal = classify(net,val);
val_acc = mean(YVal==val.Labels)

YTest = classify(net,test);
test_acc = mean(YTest==test.Labels)

