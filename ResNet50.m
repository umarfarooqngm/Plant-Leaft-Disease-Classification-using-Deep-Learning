
%% Umar
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Main script file to train the network using the googlenet
    architecture for the dataset.
%}
%% STart

%addpath 'C:\Users\umarf\OneDrive\Documents\MATLAB\Examples\R2019b\nnet\TransferLearningUsingGoogLeNetExample'

addpath '/storage/home/ubm5020/Documents/MATLAB/Examples/R2019a/nnet/TransferLearningUsingGoogLeNetExample/'
net = resnet50;
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


% fprintf('Loading val Filenames and Label Data...'); t = tic;
% val = imageDatastore(fullfile(dataDir,valid_folder),'IncludeSubfolders',true,'LabelSource','foldernames');
% val.Labels = reordercats(val.Labels,Symmetry_Groups);
% fprintf('Done in %.02f seconds\n', toc(t));


%%
rng('default');
numEpochs = 6; % 5 for both learning rates
batchSize = 200;
nTraining = length(train.Labels);

%layersTransfer = net.Layers(1:end-3);

% Define the Network Structure, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 

inputSize = net.Layers(1).InputSize;


if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 



%
numClasses = numel(categories(train.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% 

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


%

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
trainaug = augmentedImageDatastore(inputSize(1:2),train, ...
    'DataAugmentation',imageAugmenter);

%

valaug = augmentedImageDatastore(inputSize(1:2),val);
testaug = augmentedImageDatastore(inputSize(1:2),test);

%

miniBatchSize = batchSize;
valFrequency = floor(numel(trainaug.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',numEpochs, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',valaug, ...
    'ValidationFrequency',valFrequency);

%
 t = tic;
[net,info1]= trainNetwork(trainaug,lgraph,options);
fprintf('Trained in in %.02f seconds\n', toc(t));


% Train the network, info contains information about the training accuracy
% and loss


% Test on the validation data

YTrain = classify(net,trainaug);
train_acc = mean(YTrain==train.Labels)

YVal = classify(net,valaug);
val_acc = mean(YVal==val.Labels)

YTest = classify(net,testaug);
test_acc = mean(YTest==test.Labels)

% 
 save('transfer_learning_termproj_resnet_segdata')
% 
%% END
