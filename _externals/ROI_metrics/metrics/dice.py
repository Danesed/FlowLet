import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
from torcheval.metrics.functional import binary_f1_score 

sys.path.append("./");

from libs.paths import PROJECT_ROOT
from libs.utils import load_mgz, get_device


'''
    
    >> python metrics/dice.py baseline dataset
    
    With:
        - baseline: name of reference dataset
        - dataset: name of dataset
    
            
'''



if len(sys.argv) == 3:
    baseline_code = sys.argv[1];
    dataset_code = sys.argv[2];
    
    print(f"\n[!] Computing DICE SCORE metric for {dataset_code} with reference: {baseline_code}\n");
else:
    raise Exception("Incorrect number of arguments.");



''' Loading all metadata '''

metadata_baseline_path = f"./data/fsMetadata_{baseline_code}.csv";
df_baseline = pd.read_csv(metadata_baseline_path);
df_baseline = df_baseline.sort_values(by = "age").reset_index(drop = True);

metadata_synt_path = f"./data/fsMetadata_{dataset_code}.csv";
df_synt = pd.read_csv(metadata_synt_path);
df_synt = df_synt.sort_values(by = "age").reset_index(drop = True);


baseline_source_root = f"{PROJECT_ROOT}/saves/{baseline_code}/";
synt_source_root = f"{PROJECT_ROOT}/saves/{dataset_code}/";



''' Check saved processing '''

save_metrics_path = f"./results/dice/metrics_dice_{dataset_code}.csv";
if os.path.isfile(save_metrics_path):
    
    print("[!] Some work was already done.");
    df_metrics = pd.read_csv(save_metrics_path);
    start_idx = len(pd.unique(df_metrics["couple_idx"]));
    print(f"ID already seen: {start_idx} || Starting from {start_idx+1}.\n");

else:
    print("[!] Starting new processing.\n");
    
    
    df_metrics = pd.DataFrame();
    start_idx = 0;
    
    

''' Extraction Couple of sample '''
eps = 1e-5;
device = get_device();
tot = df_baseline.shape[0];
for i in range(tot):
    
    print(f"\nCouple: {i+1}/{tot}");
    
    if i >= start_idx:
        
        ''' Extract and collect metadata '''
        
        data_bl = df_baseline.loc[i];
        data_synt = df_synt.loc[i];
        
        filename_bl = data_bl["original_filename"];
        filename_synt = data_synt["original_filename"];
        
        id_bl = data_bl["id"];
        id_synt = data_synt["id"];
        
        
        ''' Prepare paths '''
        
        this_baseline_seg_path = f"{baseline_source_root}/{id_bl}/mri/aparc.DKTatlas+aseg.deep.mgz";
        this_synt_seg_path = f"{synt_source_root}/{id_synt}/mri/aparc.DKTatlas+aseg.deep.mgz";
        
        
        ''' Load data '''
        
        baseline_seg = torch.from_numpy(load_mgz(this_baseline_seg_path)).to(device);
        synt_seg = torch.from_numpy(load_mgz(this_synt_seg_path)).to(device);
    
    
        ''' Iter for region '''

        couple_summary = pd.DataFrame();
        
        labels = torch.unique(baseline_seg);
 
        scores = torch.zeros(torch.numel(labels)-1);     
         
        for idx, label in enumerate(labels):            
            label = label.cpu().item();

            if label > 0:
                scores[idx - 1] = binary_f1_score((baseline_seg == label).int().flatten(), (synt_seg == label).int().flatten()).item();
        
        n = torch.numel(labels) - 1;
        couple_summary["couple_idx"] = [i+1]*n;
        couple_summary["id_bl"] = [id_bl]*n;
        couple_summary["id_synt"] = [id_synt]*n;
        couple_summary["original_filename_bl"] = [filename_bl]*n;
        couple_summary["original_filename_synt"] = [filename_synt]*n;
        couple_summary["age_bl"] = [data_bl["age"]]*n;
        couple_summary["age_synt"] = [data_synt["age"]]*n;
        couple_summary["label"] = labels[1:].cpu();
        couple_summary["score"] = scores.cpu();
        
        
        if df_metrics.shape[0] != 0:
            df_metrics = pd.concat([df_metrics, couple_summary]);
        else:       
            df_metrics = deepcopy(couple_summary);  
            
        df_metrics.to_csv(save_metrics_path, index = False);
        
        del baseline_seg, synt_seg, scores, labels;
        torch.cuda.empty_cache();
        
    else:
        print("[!] Already done.");
        
print("\n\nEND\n\n");



  

