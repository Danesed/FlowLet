# Region-based evaluation procol Pipeline 

This repository provides the pipeline for running brain segmentation and cortical parcellation using **FastSurfer**, and for computing our proposed metrics between synthetic and real datasets.


## Setup

1. **Install Python dependencies**

   Use `pip` to install the required packages:

   ```bash
   pip install -r requirements.txt
    ````

2. **Install FastSurfer and FreeSurfer license**

   Clone the official [FastSurfer repository](https://github.com/Deep-MI/FastSurfer) and install it manually.
   Additionally, obtain a **FreeSurfer license** (free for academic use) from [https://surfer.nmr.mgh.harvard.edu/registration.html](https://surfer.nmr.mgh.harvard.edu/registration.html).

   Place both FastSurfer and the license file inside:

   ```
   ./sw/
   ```

3. **Set `FREESURFER_HOME`**

   In the provided shell script (`./run.sh`), update the `FREESURFER_HOME` variable to point to:

   ```
   /absolute/path/to/sw/freesurfer/
   ```

4. **Set repository and data paths**

   Edit the file [`./libs/paths.py`](./libs/paths.py) and set the following variables:

   ```python
   PROJECT_ROOT = "/absolute/path/to/this/repository"
   DATASET_ROOT = "/absolute/path/to/folder/containing/dataset_folder"
   ```

---

## Processing & Evaluation Workflow

5. **Run FastSurfer on a dataset**

   To segment a dataset with FastSurfer, run:

   ```bash
   python main.py <syntheticDatasetFolder> <datasetName>
   ```

   * `syntheticDatasetFolder`: Name of the folder inside `DATASET_ROOT` containing the images

   * `datasetName`: Arbitrary name used to tag this dataset

   * Outputs:

     * FastSurfer results saved in:

       ```
       ./saves/<datasetName>/
       ```
     * Metadata CSV saved to:

       ```
       ./data/fsMetadata_<datasetName>.csv
       ```

6. **Compute MAE-based metrics**

   To compare the segmentations against a baseline dataset using MAE:

   ```bash
   python metrics/roi.py <baselineName> <datasetName>
   ```

   * `baselineName`: name of a dataset already processed and stored in `./saves/`

   * `datasetName`: name of the dataset just processed

   * Output saved to:

     ```
     ./results/roi/metrics_roi_<datasetName>.csv
     ```

7. **Compute Dice score metrics**

   To compute Dice-based metrics between the baseline and synthetic dataset:

   ```bash
   python metrics/dice.py <baselineName> <datasetName>
   ```

   * Output saved to:

     ```
     ./results/dice/metrics_dice_<datasetName>.csv
     ```

8. **Summarize results**

   To generate and print a summary (mean values) of all computed metrics:

   ```bash
   python metrics/results.py
   ```

