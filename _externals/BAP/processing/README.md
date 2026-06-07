# Synthetic NIfTI Preprocessing Pipeline for BAP and ROI evaluation

This repository provides the pipeline that preprocess synthetic brain imaging data in NIfTI format. The result is a standardized dataset saved in NumPy format, along with corresponding metadata files.



## Setup

1. **Install dependencies**

   Install required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
    ```

2. **Configure paths**

   Open [`./libs/paths.py`](./libs/paths.py) and set the following variables:

   ```python
   dataset_path = "/absolute/path/to/original/nifti/dataset"
   processed_dataset_save_path = "/absolute/path/to/save/processed/numpy/data"
   ```

---

## Preprocessing Workflow

3. **Conditioned dataset**

   If your synthetic dataset is *conditioned* (i.e., filenames contain age information like `_AGE_90.00_`),
   use the `conditioned_processor.py` script.

   * First, set the variable `name_generated_dataset` inside `conditioned_processor.py` with the folder name containing your dataset.

   * Then, run:

     ```bash
     python conditioned_processor.py
     ```

   * Output:

     * Processed NumPy files saved to the configured `processed_dataset_save_path`
     * Metadata CSV saved to:

       ```
       ./saves/metadata/participants_{name_generated_dataset}.csv
       ```

4. **Unconditioned dataset**

   If your dataset is *not conditioned*, ages are randomly assigned based on the distribution from the training set used by the generative model.
   This distribution is taken from:

   ```
   ./saves/participants_merged_datasets.csv
   ```

   * Set the variable `name_generated_dataset` inside `unconditioned_processor.py`

   * Run:

     ```bash
     python unconditioned_processor.py
     ```

   * Output:

     * Processed NumPy files saved to the configured `processed_dataset_save_path`
     * Metadata CSV saved to:

       ```
       ./saves/metadata/participants_{name_generated_dataset}.csv
       ```

5. **Merge metadata**

   To merge your generated metadata with the original training metadata, use:

   ```bash
   python merger.py
   ```

   This concatenates:

   * `./saves/metadata/participants_{name_generated_dataset}.csv`
   * `./saves/participants_merged_datasets.csv`

   and saves the result to:

   ```
   ./saves/participants/participants_{name_generated_dataset}.csv
   ```



