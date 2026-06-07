import sys
import numpy as np
import pandas as pd

from libs.classes import trainer
from libs.paths import data_path, train_dataset_path
from libs.config import batch_size, epochs, check
from libs.utility import set_seed, get_device, get_DenseNet, check_folder, save_curve, save_model, get_dataload, check_cuda, get_training_params

import warnings
warnings.filterwarnings("ignore");

''' 

    [!] TRAINER

    ////////////////////////////////////////////////////////////////////////////////////////////
    /                                                                                          /
    /                             python trainer.py modelName                                  /
    /                                                                                          /
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    With:
        - modelName: name of folder in which results will be stored (the script create it in ./saves/);
                     it is the name of the data.csv/.tsv file too (participant_modelName).

    
    folder for saving data:
    ./saves/modelname/
        - data/ : 
            - learning curve (.npy)
            - train/test performance (.csv)
            - final model (.pt)
        - checkpoints/: contains all checkpoints

'''

if len(sys.argv) == 2:  

    modelName = sys.argv[1];
    print("\n//////////// TRAINER ////////////");
    print(f" ->  saving in: ./saves/{modelName}/\n");

else:
    sys.exit("[X] ERROR: Invalid number of parameters.");


set_seed(seed = 82);
device = get_device();
train_data_path = f"{data_path}/train/participants_{modelName}.csv";
root_save_path = f"./saves/{modelName}/";
check_folder(path = root_save_path);
save_path = f"./saves/{modelName}/data/";
check_folder(path = save_path);
cp_path = f"./saves/{modelName}/checkpoints/";
check_folder(path = cp_path);

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


train_dload = get_dataload(batch_size = batch_size,
                           data_path = train_data_path,
                           dataset_path = train_dataset_path,
                           _min = _min,
                           _max = _max);
model = get_DenseNet(device = device);

criterion, optimizer, lr_scheduler = get_training_params(model = model);

trainer_machine = trainer(model = model,
        device = device,
        criterion = criterion,
        optimizer = optimizer,
        training_DataLoader = train_dload, 
        lr_scheduler = lr_scheduler,
        no_cuda = check_cuda(device),
        epochs = epochs,
        cp_path = cp_path,
        check = check,
        );
model, train_mean, train_std, _, _, _ = trainer_machine.run_trainer();

save_curve(path = save_path,
           train_means = train_mean,
           train_stds = train_std);

save_model(path = save_path,
           model = model);

print("\nEND\n\n");