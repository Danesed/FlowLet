import pandas as pd


code = "";


train = pd.read_csv("./saves/metadata/participants_merged_datasets.csv");
train["id"] = [f"/merged_datasets/sub-{id}_preproc-quasiraw_T1w.npy" for id in train["id"].to_list()];

aug = pd.read_csv(f"./saves/metadata/participants_{code}.csv");
aug["id"] = [f"/{code}/sub-{id}_preproc-quasiraw_T1w.npy" for id in aug["id"].to_list()];

participants = pd.concat([
    aug[["id", "age", "split"]],
    train[["id", "age", "split"]],
    ]);

participants.to_csv(f"./saves/participants/participants_{code}.csv", index = False);

print("\nEND\n\n");
