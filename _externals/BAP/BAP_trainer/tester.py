import sys
import numpy as np
import pandas as pd

from libs.classes import tester
from libs.paths import data_path, train_dataset_path, test_dataset_path
from libs.utility import set_seed, get_device, get_preTrained_DenseNet, get_dataload, check_cuda, get_training_params, save_metadata

import warnings
warnings.filterwarnings("ignore");

''' 

    [!] TESTER 
    
    ////////////////////////////////////////////////////////////////////////////////////////////
    /                                                                                          /
    /                             python tester.py modelName                                   /
    /                                                                                          /
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    With:
        - modelName: name of file in which model parameters are stored.

'''

if len(sys.argv) == 2:  
    modelName = sys.argv[1];
    
    print("\n//////////// TESTER ////////////");
    print(f" ->  Model: {modelName}\n");

else:
    sys.exit("[X] ERROR: Invalid number of parameters.");


set_seed(seed = 82);
device = get_device();

    
train_data_path = f"{data_path}/train/participants_merged_datasets.csv";
test_data_path = f"{data_path}/test/participants.csv";

save_path = f"./saves/{modelName}/data/";

model = get_preTrained_DenseNet(device = device,
                                path = f"./saves/{modelName}/data/trained_model.pt");


# !
print("\n[!] Get min and max...");

_min = np.inf;
_max = -np.inf;

df_min_max = pd.read_csv(train_data_path);
min_list = [];
max_list = [];
for file in df_min_max["id"].tolist():

    img = np.load(f"{train_dataset_path}/{file}");
    
    min_list.append(img.min());
    max_list.append(img.max());
    
_min = np.percentile(min_list,5);
_max = np.percentile(max_list,95);
print(f"MIN: {_min}");
print(f"MAX: {_max}");
# !


''' TRAIN PERFORMANCE '''

print("\n\nTRAINING SET:\n");

train_dload = get_dataload(batch_size = 1,
                          data_path = train_data_path,
                          dataset_path = train_dataset_path,
                          _min = _min,
                          _max = _max);


criterion, _, _ = get_training_params(model = model);
train_performance = tester(model = model,
                       device = device,
                       test_DataLoader = train_dload,
                       criterion = criterion,
                       no_cuda = check_cuda(device = device),
                       );
train_losses, train_ids, train_targets, train_preds = train_performance.run_tester();



''' TEST PERFORMANCE '''
print("\n\nTEST SET:\n");
test_dload = get_dataload(batch_size = 1,
                          data_path = test_data_path,
                          dataset_path = test_dataset_path,
                          _min = _min,
                          _max = _max);



criterion, _, _ = get_training_params(model = model);
test_performance = tester(model = model,
                       device = device,
                       test_DataLoader = test_dload,
                       criterion = criterion,
                       no_cuda = check_cuda(device = device),
                       );
test_losses, test_ids, test_targets, test_preds = test_performance.run_tester();



''' SAVE ALL METADATAS '''

save_metadata(path = save_path,
            train_losses = train_losses,
            test_losses = test_losses,
            train_ids = train_ids,
            test_ids = test_ids,
            train_targets = train_targets,
            test_targets = test_targets,
            train_preds = train_preds,
            test_preds = test_preds);

print("\n\nEND\n\n");