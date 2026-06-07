import os
import numpy as np
import pandas as pd

from libs.utils import nii2npy, standardize, guess_age, crop, check_folder
from libs.paths import dataset_path, processed_dataset_save_path

'''
    [!] PROCESSING

    -> metadata(id, original_filename, dataset, age, site, split)
'''

#####################################################################

name_generated_dataset = "";

#####################################################################


summary_path = f"./saves/participants_{name_generated_dataset}.csv";


if os.path.isfile(summary_path):
    
    print("[!] Some work was already done.");
    summary = pd.read_csv(summary_path);
    start_idx = summary.shape[0];
    print(f"ID seen: {start_idx}/Starting from {start_idx+1}.\n");

else:
    print("[!] Starting new processing.\n");
    
    
    summary = pd.DataFrame();
    start_idx = 0;
    
dataset_path = f"{dataset_path}/{name_generated_dataset}";      
files = os.listdir(dataset_path);

processed_dataset_save_path = f"{processed_dataset_save_path}/{name_generated_dataset}";
check_folder(processed_dataset_save_path);

df = pd.read_csv("./saves/metadata/participants_merged_datasets.csv");
synt_ages = guess_age(real_ages = df["age"].to_list(),
                num_samples = len(files));

for i,file in enumerate(files):
    
    print(f"{i+1}/{len(files)}");
    
    if i >= start_idx:
        tempZipPath = f"{dataset_path}/{file}";    
        data = nii2npy(tempZipPath);
        
        ''' Metadata extraction '''
        metadata = {};
        metadata["id"] = 10000 + (i + 1);
        metadata["original_filename"] = file[:-7];
        metadata["dataset"] = name_generated_dataset;
        metadata["site"] = "synt";
        metadata["age"] = synt_ages[i];
        metadata["split"] = "train";

        tempDatasetPath = f"{processed_dataset_save_path}/sub-{metadata["id"]}_preproc-quasiraw_T1w";
            
        if data.shape != (91,109,91):
            data = crop(data, (91,109,91));
        standardized = standardize(data)
        expanded = np.expand_dims(standardized, axis = (0,1));    
        np.save(tempDatasetPath, expanded.astype(np.float32));
        
        ''' Save '''
        
        if summary.shape[0] != 0:
            summary.loc[summary.shape[0]] = metadata;
        
        else:       
            summary = pd.DataFrame(metadata, index=[0]);

        summary.to_csv(summary_path, index = False); 
    
    else:
        print("[!] Already Done.")    

        
print("\nEND\n\n");