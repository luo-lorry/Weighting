import os, glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prediction_set_evaluator_framework import PredictionSetEvaluator

# Set the folder path and load the data files
folder = 'data'
files = glob.glob(os.path.join(folder, '*tweet.pt'))

all_data_results = []

for file in files:
    # Load the logits and labels from the current file
    data = torch.load(file)
    logits = data['logits']
    labels = data['labels']
    filename = file.split('logits_labels_')[-1].split('.pt')[0]
    print(f"Computing Results for {filename} ... ")

    num_runs = 10  # Number of runs with different index splits

    # Define the range of alpha values
    alphas = [0.005, 0.01] # [round(alpha, 3) for alpha in np.arange(0.005, 0.055, 0.005)]

    # Initialize an empty list to store the results for each alpha
    all_results = []

    for alpha in alphas:
        evaluator = PredictionSetEvaluator(logits, labels, filename=filename, precision=0.01, method='VFCP')
        df_results, avg_results = evaluator.compare_performance_multiple_splits(num_runs, alpha=alpha)

        # Append the average results to the list
        all_results.append(avg_results)

    # Create a DataFrame from the list of results
    df_all_results = pd.DataFrame(all_results)
    all_data_results.append(df_all_results)

pd.concat(all_data_results).to_csv('results.csv')

# Print the DataFrame
print(all_data_results)

print("\nAverage Results:")
for metric, value in all_data_results.items():
    print(f"{metric}: {value:.4f}")

evaluator