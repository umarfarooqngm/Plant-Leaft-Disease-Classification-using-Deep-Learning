
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: The script will do the augmentation like rotation, sclaing
    and translation o the data given in the dataset for 4 times each
    dataset and this will be increasing the number of dataset by 4 times.
%}
%% Start
Symmetry_Groups = {'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'};


angle_values = [];
scaling_vaues = [];
translation_valueX = [];
translation_valueY = [];

for i = 1:numel(Symmetry_Groups)
    
folder = "C:\Users\umarf\Downloads\PRML\Projects\Project 4\p4_startercode\p4_startercode\data\wallpapers\train\" + Symmetry_Groups(i);
test = Symmetry_Groups(i);

for_dir = folder + "\*.png";
contents = dir(for_dir);

for j = 1:numel(contents)
 image_location  =  folder + "\" + contents(j).name; 
 
 I = imread(image_location);
 
 for k = 1:4

angle = randi(360);

angle_values = [angle_values angle];

J = imrotate(I,angle);

p = randi([100,200],1)/100;

scaling_vaues = [scaling_vaues p*100];

K = imresize(J,p);

s = randi([-50,50],1,2);

translation_valueX = [translation_valueX s(1,1)];
translation_valueY = [translation_valueY s(1,2)];

P = imtranslate(K,[s(1,1),s(1,2)]);

e = centerCropWindow2d(size(P),[128 128]);

BW = imcrop(P,e);

%imshow(P);

address = "C:\Users\umarf\Downloads\PRML\Projects\Project 4\p4_startercode\p4_startercode\data\wallpapers\train_aug\" + Symmetry_Groups(i) + "\" + Symmetry_Groups(i) + "_" + j + "_"+ k + ".png";

imwrite(BW,address);


 end
end
end

figure(1)
histogram(angle_values');


figure(2)
histogram(scaling_vaues'); 


figure(3)
histogram2(translation_valueX',translation_valueY',[10 10],'FaceColor','flat'); 
colorbar




%% ENd
