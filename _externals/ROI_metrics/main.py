import os
import sys
import subprocess
import pandas as pd


from libs.utils import check_folder, extract_age
from libs.paths import PROJECT_ROOT, DATASET_ROOT


''' 

    >> python main.py root dataset
    
    With:
        - root: dataset path (to nii.gz)
        - dataset: name of dataset
        
'''


if len(sys.argv) == 3:
    dataset_path = sys.argv[1];
    dataset_code = sys.argv[2];
    
    print(f"\n[!] Processing {dataset_code} from {dataset_path}\n");
else:
    raise Exception("Incorrect number of arguments.");

saves_root = f"{PROJECT_ROOT}/saves/{dataset_code}/";
check_folder(saves_root);

source_root = f"{DATASET_ROOT}/{dataset_path}";
files = os.listdir(source_root);


summary_path = f"./data/fsMetadata_{dataset_code}.csv";
if os.path.isfile(summary_path):
    
    print("[!] Some work was already done.");
    summary = pd.read_csv(summary_path);
    start_idx = summary.shape[0];
    print(f"ID already investigated: {start_idx}/Starting from {start_idx+1}.\n");

else:
    print("[!] Starting new processing.\n");    
    summary = pd.DataFrame();
    start_idx = 0;
  
    
for i,file in enumerate(files):
    
    print(f"\n\n SAMPLE: {i+1}/{len(files)}\n\n");
    if i >= start_idx:
        this_file_path = f"{source_root}/{file}";
        
        id = 10000 + (i + 1);
        subprocess.run(["bash",
                        "./run.sh",
                        this_file_path,
                        saves_root,
                        str(id)]);
        
        
        ''' Save metadata '''
        
        metadata = {};
        metadata["id"] = id;
        metadata["original_filename"] = file[:-7];
        metadata["dataset"] = dataset_code;
        metadata["age"] = extract_age(file);
            
        
        if summary.shape[0] != 0:
            summary.loc[summary.shape[0]] = metadata;
        
        else:       
            summary = pd.DataFrame(metadata, index=[0]);

        summary.to_csv(summary_path, index = False); 
        
    else:
        print("[!] Already done.");

print("\nEND\n\n");