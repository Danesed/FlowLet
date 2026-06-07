import os
import sys
import numpy as np
import pandas as pd

sys.path.append("./");

from libs.paths import PROJECT_ROOT
from libs.utils import extract_flowlet_model


dice_path = f"{PROJECT_ROOT}/results/dice/";
roi_path = f"{PROJECT_ROOT}/results/roi/";


models_roi = [];
models_dice = [];
uMAE = [];
iMAE = [];
uMSE = [];
iMSE = [];
uKLD = [];
iKLD = [];
dice = [];


roi_files = os.listdir(roi_path);
for file in roi_files:
    temp_path = f"{roi_path}/{file}";
    temp_df = pd.read_csv(temp_path);
    
    if file.startswith("metrics_roi_FlowLetAblation"):
        model_name = extract_flowlet_model(file);
    else:
        model_name = file[12:-4];
    models_roi.append(model_name);
    
    uMAE.append(f"{temp_df["uMAE"].mean():.2f} ± {temp_df["uMAE"].std():.2f}"); 
    iMAE.append(f"{temp_df["iMAE"].mean():.2f} ± {temp_df["iMAE"].std():.2f}"); 
    uMSE.append(f"{temp_df["uMSE"].mean():.2f} ± {temp_df["uMSE"].std():.2f}"); 
    iMSE.append(f"{temp_df["iMSE"].mean():.2f} ± {temp_df["iMSE"].std():.2f}"); 
    uKLD.append(f"{temp_df["uKLD"].mean():.2f} ± {temp_df["uKLD"].std():.2f}"); 
    iKLD.append(f"{temp_df["iKLD"].mean():.2f} ± {temp_df["iKLD"].std():.2f}"); 
    
dice_files = os.listdir(dice_path);
for file in dice_files:
    temp_path = f"{dice_path}/{file}";
    temp_df = pd.read_csv(temp_path);
    
    if file.startswith("metrics_dice_FlowLetAblation"):
        model_name = extract_flowlet_model(file);
    else:
        model_name = file[13:-4];
    models_dice.append(model_name);
    dice.append(f"{temp_df["score"].mean():.2f} ± {temp_df["score"].std():.2f}"); 


rois = pd.DataFrame();
rois["model"] = models_roi;
rois["uMAE"] = uMAE;
rois["iMAE"] = iMAE;
rois["uMSE"] = uMSE;
rois["iMSE"] = iMSE;
rois["uKLD"] = uKLD;
rois["iKLD"] = iKLD;

dices = pd.DataFrame();
dices["model"] = models_dice;
dices["dice"] = dice;

print("\nGLOBAL\n");
print(rois.head(rois.shape[0]));

print("\n");
print(dices.head(dices.shape[0]));

print("\n");



''' LOCAL '''





n_models = len(roi_files);
n_labes = 95;

uMAE = np.zeros((2, n_models, n_labes));
iMAE = np.zeros((2, n_models, n_labes));
uMSE = np.zeros((2, n_models, n_labes));
iMSE = np.zeros((2, n_models, n_labes));
uKLD = np.zeros((2, n_models, n_labes));
iKLD = np.zeros((2, n_models, n_labes));
dice = np.zeros((2, n_models, n_labes));


models_roi = [];
for model_idx, file in enumerate(roi_files):
    temp_path = f"{roi_path}/{file}";
    temp_df = pd.read_csv(temp_path);
    
    if file.startswith("metrics_roi_FlowLetAblation"):
        model_name = extract_flowlet_model(file);
    else:
        model_name = file[12:-4];
    models_roi.append(model_name);
    
    labels = pd.unique(temp_df["label"]);
    
    for label_idx, label in enumerate(labels): 
        df_label = temp_df[temp_df["label"] == label];

        uMAE[0, model_idx, label_idx] = df_label["uMAE"].mean();
        uMAE[1, model_idx, label_idx] = df_label["uMAE"].std();
        
        iMAE[0, model_idx, label_idx] = df_label["iMAE"].mean();
        iMAE[1, model_idx, label_idx] = df_label["iMAE"].std();
        
        uMSE[0, model_idx, label_idx] = df_label["uMSE"].mean();
        uMSE[1, model_idx, label_idx] = df_label["uMSE"].std();
        
        iMSE[0, model_idx, label_idx] = df_label["iMSE"].mean();
        iMSE[1, model_idx, label_idx] = df_label["iMSE"].std();
        
        uKLD[0, model_idx, label_idx] = df_label["uKLD"].mean();
        uKLD[1, model_idx, label_idx] = df_label["uKLD"].std();
        
        iKLD[0, model_idx, label_idx] = df_label["iKLD"].mean();
        iKLD[1, model_idx, label_idx] = df_label["iKLD"].std();
        
    

models_dice = [];
for model_idx, file in enumerate(dice_files):
    temp_path = f"{dice_path}/{file}";
    temp_df = pd.read_csv(temp_path);
    
    if file.startswith("metrics_dice_FlowLetAblation"):
        model_name = extract_flowlet_model(file);
    else:
        model_name = file[13:-4];
    models_dice.append(model_name);
    
    labels = pd.unique(temp_df["label"]);

    for label_idx, label in enumerate(labels): 
        df_label = temp_df[temp_df["label"] == label];
    
        dice[0, model_idx, label_idx] = df_label["score"].mean();
        dice[1, model_idx, label_idx] = df_label["score"].std();




metrics = np.zeros((2, 4, 2, n_models, n_labes));

metrics[0, 0, :, :, :] = iMAE;
metrics[0, 1, :, :, :] = iMSE;
metrics[0, 2, :, :, :] = iKLD;
metrics[0, 3, :, :, :] = dice;

metrics[1, 0, :, :, :] = uMAE;
metrics[1, 1, :, :, :] = uMSE;
metrics[1, 2, :, :, :] = uKLD;
metrics[1, 3, :, :, :] = dice;


np.save("./results/summary/summary", metrics);


print("LOCAL");

df_agg_results = pd.DataFrame();
for m,model in enumerate(models_dice):
    
    row = {"model": model};
    
    for i,type in enumerate(["i", "u"]):
    
        for j,met in enumerate(["MAE","MSE","KLD","DICE"]):
            
            row.update({
                f"{type}{met}": f"{metrics[i,j,0,m,:].mean():.2f} ± {metrics[i,j,0,m,:].std():.2f}",
            });

    if df_agg_results.shape[0] != 0:
            df_agg_results.loc[df_agg_results.shape[0]] = row;
    else:     
        df_agg_results = pd.DataFrame(row, index=[0])
    
        
print(df_agg_results.head(df_agg_results.shape[0]));


df_agg_results.to_csv("./results/summary/roi-dice_local.csv", index = False);
rois.to_csv("./results/summary/rois_global.csv", index = False);
dices.to_csv("./results/summary/dices_global.csv", index = False);

print("\n\n")