import os
import sys
import torch
import numpy as np
import pandas as pd
from copy import deepcopy

sys.path.append("./");

from scipy.special import kl_div
from libs.paths import PROJECT_ROOT
from libs.utils import load_mgz, get_torch_normalized_hist, torch_KLD, get_device


'''
    
    >> python metrics/roi.py baseline dataset
    
    With:
        - baseline: name of reference dataset
        - dataset: name of dataset
            
'''



if len(sys.argv) == 3:
    baseline_code = sys.argv[1];
    dataset_code = sys.argv[2];
    
    print(f"\n[!] Computing roi metric for {dataset_code} with reference: {baseline_code}\n");
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

save_metrics_path = f"./results/roi/metrics_roi_{dataset_code}.csv";
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
        
        d = {
            "couple_idx": i+1,
            "id_bl": id_bl,  
            "id_synt": id_synt, 
            "original_filename_bl": filename_bl, 
            "original_filename_synt": filename_synt, 
            "age_bl": data_bl["age"], 
            "age_synt": data_synt["age"], 
        };
        
        
        ''' Prepare paths '''
        
        this_baseline_path = f"{baseline_source_root}/{id_bl}/mri/orig.mgz";
        this_synt_path = f"{synt_source_root}/{id_synt}/mri/orig.mgz";
        
        this_baseline_seg_path = f"{baseline_source_root}/{id_bl}/mri/aparc.DKTatlas+aseg.deep.mgz";
        this_synt_seg_path = f"{synt_source_root}/{id_synt}/mri/aparc.DKTatlas+aseg.deep.mgz";
        
        
        ''' Load data '''
        
        baseline = torch.from_numpy(load_mgz(this_baseline_path)).to(device);
        synt = torch.from_numpy(load_mgz(this_synt_path)).to(device);
        
        baseline_seg = torch.from_numpy(load_mgz(this_baseline_seg_path)).to(device);
        synt_seg = torch.from_numpy(load_mgz(this_synt_seg_path)).to(device);
    
    
        ''' Iter for region '''
    
        couple_summary = pd.DataFrame();
        
        mae_based = torch.abs(torch.sub(baseline,synt));
        mse_based = torch.pow(torch.sub(baseline,synt), 2);
        
        for label in torch.unique(baseline_seg):
            
            label = label.cpu().item();

            if label > 0:
                uKLD, iKLD, uMSE, iMSE, uMAE, iMAE = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan;
                
                label_dict = deepcopy(d);
                
                baseline_mask = (baseline_seg == label);
                synt_mask = (synt_seg == label);
                
                intersection = torch.logical_and(baseline_mask,synt_mask);
                union = torch.logical_or(baseline_mask,synt_mask);

                
                this_int_mae_based = mae_based[intersection];
                this_int_mse_based = mse_based[intersection];
                int_baseline = baseline[intersection];
                int_synt = synt[intersection];
                
                this_un_mae_based = mae_based[union];
                this_un_mse_based = mse_based[union];
                un_baseline = baseline[union];
                un_synt = synt[union];
                
                
                
                if torch.numel(int_baseline) != 0:  
                    iKLD = torch_KLD(
                        P = get_torch_normalized_hist(int_baseline),
                        Q = get_torch_normalized_hist(int_synt),
                        ).cpu().item();            
                    iMAE = torch.mean(this_int_mae_based).cpu().item();           
                    iMSE = torch.mean(this_int_mse_based).cpu().item();  
                    
                if torch.numel(un_baseline) != 0:                      
                    uKLD = torch_KLD(
                        P = get_torch_normalized_hist(un_baseline),
                        Q = get_torch_normalized_hist(un_synt),
                        ).cpu().item();       
                    uMAE = torch.mean(this_un_mae_based).cpu().item();
                    uMSE = torch.mean(this_un_mse_based).cpu().item();         

                
                label_dict["label"] = label;
                label_dict["intersection"] = torch.numel(int_baseline);
                label_dict["union"] = torch.numel(un_baseline);
                label_dict["uKLD"] = uKLD;
                label_dict["iKLD"] = iKLD;
                label_dict["uMAE"] = uMAE;
                label_dict["iMAE"] = iMAE;
                label_dict["uMSE"] = uMSE;
                label_dict["iMSE"] = iMSE;
                
                
                if couple_summary.shape[0] != 0:
                    couple_summary.loc[couple_summary.shape[0]] = label_dict;
                else:       
                    couple_summary = pd.DataFrame(label_dict, index=[0]);
                
                
                del this_int_mae_based, this_int_mse_based, int_baseline, int_synt, this_un_mae_based, this_un_mse_based, un_baseline, un_synt;          
                torch.cuda.empty_cache();
                
        
        if df_metrics.shape[0] != 0:
            df_metrics = pd.concat([df_metrics, couple_summary]);
        else:       
            df_metrics = deepcopy(couple_summary);  
            
        df_metrics.to_csv(save_metrics_path, index = False);

        
        del mae_based, mse_based;
        torch.cuda.empty_cache();
        
    else:
        print("[!] Already done.");
        
print("\n\nEND\n\n");



  

