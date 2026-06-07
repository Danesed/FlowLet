# Synthetic Data Training & Evaluation Pipeline

This repository provides the pipeline to train and evaluate models using synthetic data in NumPy format.


## Setup

1. **Install requirements**

   Install the required dependencies via `pip`:

    ```bash
   pip install -r requirements.txt
    ```

2. **Set dataset paths**

   Open the file [`/libs/paths.py`](libs/paths.py) and modify the following variables:

   ```python
   train_dataset_path = "/absolute/path/to/train/data"
   test_dataset_path = "/absolute/path/to/test/data"
   ```

3. **Prepare metadata CSVs**

   In the following folders:

   * `./data/train/`
   * `./data/test/`

   add a `.csv` file named:

   ```
   participants_modelName.csv
   ```

   where `modelName` is the name of the model that generated the synthetic data.

   Each CSV must include the following columns:

   * `id` — identifier matching the NumPy files
   * `age` — target variable

---

## Workflow

4. **Activate your virtual environment**

   ```bash
   source /path/to/venv/bin/activate
   ```

5. **Train the model**

   ```bash
   python trainer.py modelName
   ```

6. **Validate the model**

   ```bash
   python tester.py modelName
   ```

7. **Print performance metrics**

   ```bash
   python tools/performance.py
   ```

---

## Output

All training and testing data, along with model checkpoints and logs, will be saved in:

```
./saves/modelName/
```
