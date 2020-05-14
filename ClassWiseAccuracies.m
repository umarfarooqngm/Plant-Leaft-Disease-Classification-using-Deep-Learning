%% START
%{
    Name: Umar Farooq
    PSU Email ID: ubm5020@psu.edu
    Description: Script file to plot the class wise accuracies for the
    predictions of the network trained.
    The blue rectagle is accuracy and blue is failure percentage.
%}
%% CODE
Symmetry_Groups = {'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'};
ticcount = zeros(38,1);
ticacc = zeros(38,1);
for i = 1:38
    yoyoyo = Symmetry_Groups{i};
    i
    for j = 1:14058;
        if  test.Labels(j,1) == yoyoyo
            ticcount(i,1) = ticcount(i,1) + 1;
            if test.Labels(j,1) == YTest(j,1)
                ticacc(i,1) = ticacc(i,1) +1;
            end
        end
    end
end

ticaccu = 100*(ticacc./ticcount);

ticfail = 100- ticaccu;

bar([ticaccu, ticfail])
%% END

