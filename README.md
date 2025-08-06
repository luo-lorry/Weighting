# Conformity Score Averaging for Classification (ICML 2025)

This repository contains the implementation for the paper "Conformity Score Averaging for Classification". The code learns an optimal weighted average of multiple conformity scores to produce smaller prediction sets in conformal prediction while maintaining the desired coverage level.

## How to Run the Experiments

### Step 1: Install Dependencies
First, clone this repository. Then, install the necessary Python libraries:

```bash
pip install numpy pandas torch scikit-learn matplotlib
```
### Step 2: Prepare Your Data
1. Create a folder named `data` in the root of this project.
2. Place your data files inside the `data/` folder. The code is set up to look for PyTorch files (`.pt`).
3. Each `.pt` file must be a dictionary containing two PyTorch tensors:
   - `'logits'`: A tensor of model outputs with shape `(number_of_samples, number_of_classes)`.
   - `'labels'`: A tensor of the correct labels with shape `(number_of_samples,)`.
### Step 3: Run the Code
Execute the main script from your terminal. It will automatically find your data files, run the evaluation, and print the results.
```bash
python compare_performance.py
```
A file named `results.csv` containing the detailed performance metrics will be created in the project's root directory.
## Code Overview
*   **`prediction_set_evaluator_framework.py`**: Contains the main `PredictionSetEvaluator` class. This file handles the core logic of computing conformity scores, finding the best weights, and evaluating the final performance.
*   **`compare_performance.py`**: This is the main script you run. It loads the data, sets up the evaluation parameters (like the error rate `alpha`), and calls the framework to produce the results.

## Citation

If you use this code or our method in your research, please cite our paper:

```bibtex
@inproceedings{luoconformity,
  title={Conformity score averaging for classification},
  author={Luo, Rui and Zhou, Zhixin},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
