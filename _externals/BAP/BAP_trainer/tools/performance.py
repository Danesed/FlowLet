import os
import sys
import numpy as np
import pandas as pd


sys.path.append("../BA_trainer/");

th = 44;

models = os.listdir("./saves/");

for model in models:
    path = f"./saves/{model}/data/performance.csv";
    data = pd.read_csv(path);

    train = data[data["split"] == "train"];
    test = data[data["split"] == "test"];
    
    train_loss = train["loss"].tolist();
    test_loss = test["loss"].tolist();
    
    train_loss_old = train[train["age"] >= th]["loss"].tolist();
    test_loss_old = test[test["age"] >= th]["loss"].tolist();

    train_loss_young = train[train["age"] < th]["loss"].tolist();
    test_loss_young = test[test["age"] < th]["loss"].tolist();
    
    print(f"\n{model}:");
    print(" - Global: ");
    print(f"    - Train [AE]: {np.mean(train_loss):.2f} ± {np.std(train_loss):.2f}");
    print(f"    - Test  [AE]: {np.mean(test_loss):.2f} ± {np.std(test_loss):.2f}");
    
    print(f" - age < {th} yy: ");
    print(f"    - Train [AE]: {np.mean(train_loss_young):.2f} ± {np.std(train_loss_young):.2f}");
    print(f"    - Test  [AE]: {np.mean(test_loss_young):.2f} ± {np.std(test_loss_young):.2f}");  
    print(f" - age >= {th} yy: ");
    print(f"    - Train [AE]: {np.mean(train_loss_old):.2f} ± {np.std(train_loss_old):.2f}");
    print(f"    - Test  [AE]: {np.mean(test_loss_old):.2f} ± {np.std(test_loss_old):.2f}");
    
print("\nEND\n\n");