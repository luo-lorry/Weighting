import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torchcp.classification.scores import THR, APS, SAPS, RAPS, THRRANK


def FSSCV(prediction_sets, labels, alpha, stratified_size=[[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]]):
    """
    Size-stratified coverage violation (SSCV)
    """
    labels = labels.cpu()
    size_array = np.zeros(len(labels))
    correct_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = ele.sum()
        # correct_array[index] = 1 if labels[index] in ele else 0
        correct_array[index] = 1 if ele[labels[index]] == 1 else 0

    sscv = -1
    for stratum in stratified_size:
        temp_index = np.argwhere((size_array >= stratum[0]) & (size_array <= stratum[1]))
        if len(temp_index) > 0:
            stratum_violation = abs((1 - alpha) - np.mean(correct_array[temp_index]))
            sscv = max(sscv, stratum_violation)
    return sscv

class PredictionSetEvaluator:
    def __init__(self, logits, labels, filename='cifar-10', score_functions=None, score_function_params=None, precision=0.01, method='VFCP'):
        self.logits = logits
        self.labels = labels
        self.filename = filename
        self.initial_weights = None
        self.method = method

        if score_functions is None:
            score_functions = [THR, THRRANK, APS]
        self.score_functions = score_functions

        if score_function_params is None:
            score_function_params = {}
        self.score_function_params = score_function_params

        self.scores = self.compute_scores(logits)
        self.ranks = self.compute_ranks(self.scores)

        self.candidate_weights = self.generate_candidate_weights(len(self.score_functions), precision)


    def compute_scores(self, logits):
        scores = []
        for score_function in self.score_functions:
            if score_function in self.score_function_params:
                score_function_param = self.score_function_params[score_function]
            else:
                score_function_param = None

            if score_function_param is not None:
                if isinstance(score_function_param, tuple):
                    score_func = score_function(*score_function_param)
                else:
                    score_func = score_function(score_function_param)
            else:
                score_func = score_function()

            scores.append(score_func(logits).numpy())
        return np.stack(scores, axis=0)

    def compute_ranks(self, scores):
        ranks = []
        for score in scores:
            n, k = score.shape
            rank = np.zeros((n, k))
            flattened_indices = np.argsort(score.flatten())
            rank.flat[flattened_indices] = np.arange(1, n * k + 1)
            ranks.append(rank)
        return np.stack(ranks, axis=0)

    def calibration(self, weights, calib_indices, alpha):
        ranks = np.tensordot(weights, self.ranks, axes=(0, 0)) / sum(weights)
        quantiles = np.percentile(ranks[calib_indices, self.labels[calib_indices]], 100 * (1 - alpha), axis=0)
        return quantiles

    def evaluation(self, weights, eval_indices, quantiles):
        ranks = np.tensordot(weights, self.ranks, axes=(0, 0)) / sum(weights)
        prediction_sets = (ranks[eval_indices] <= np.array([quantiles])[:, np.newaxis]).astype(int)
        coverage = np.mean(prediction_sets[np.arange(eval_indices.shape[0]), self.labels[eval_indices]])
        size = np.mean(np.sum(prediction_sets, axis=1))
        return coverage, size

    def find_optimal_weights(self, step1_indices, step2_indices, alpha):
        best_weights = None
        best_avg_size = float('inf')
        avg_sizes = []

        for weights in self.candidate_weights:
            quantiles = self.calibration(weights, step1_indices, alpha)
            _, size = self.evaluation(weights, step2_indices, quantiles)
            avg_sizes.append(size)
            if size < best_avg_size:
                best_avg_size = size
                best_weights = weights

        return best_weights, best_avg_size, avg_sizes

    def evaluate_final_performance(self, step3_indices, test_indices, weights, alpha):
        quantiles = self.calibration(weights, step3_indices, alpha)
        coverage, size = self.evaluation(weights, test_indices, quantiles)
        return coverage, size

    def compare_performance(self, train_indices, test_indices, alpha):
        method = self.method
        if method == "VFCP":
            num_runs = 10
            results = []
            for run in range(num_runs):
                rng = np.random.default_rng(seed=run)
                train_indices_permuted = rng.permutation(train_indices)
                split_idx = len(train_indices_permuted) // 2
                step1_indices = step2_indices = train_indices_permuted[:split_idx]
                step3_indices = train_indices_permuted[split_idx:]

                optimal_weights, optimal_size, avg_sizes = self.find_optimal_weights(step1_indices, step2_indices, alpha)
                weighted_coverage, weighted_size = self.evaluate_final_performance(step3_indices, test_indices, optimal_weights, alpha)

                indices1 = step3_indices
                indices2 = test_indices
                individual_results = []
                for i, score_function in enumerate(self.score_functions):
                    score_name = score_function.__name__
                    coverage, size = self.evaluate_final_performance(indices1, indices2, [int(i == j) for j in
                                                                                          range(len(self.score_functions))],
                                                                     alpha)
                    print(f"{score_name} - Coverage: {coverage:.4f}, Size: {size:.4f}")
                    individual_results.append((score_name, coverage, size))

                results.append({
                    'Run': run + 1,
                    'Alpha': alpha,
                    'Optimal Weights': optimal_weights,
                    'Weighted Combination Coverage': weighted_coverage,
                    'Weighted Combination Size': weighted_size,
                    'Avg Sizes': np.array(avg_sizes),
                    **{f'{score_name} Coverage': coverage for score_name, coverage, _ in individual_results},
                    **{f'{score_name} Size': size for score_name, _, size in individual_results}
                })

            df_results = pd.DataFrame(results)
            # Calculate the average results
            average_results = {
                'Data': self.filename,
                'Alpha': alpha,
                'Optimal Weights': np.mean([result['Optimal Weights'] for result in results], axis=0),
                'Weighted Combination Coverage': np.mean([result['Weighted Combination Coverage'] for result in results]),
                'Weighted Combination Size': np.mean([result['Weighted Combination Size'] for result in results]),
                **{f'{score_name} Coverage': df_results[f'{score_name} Coverage'].mean() for score_name, _, _ in
                   individual_results},
                **{f'{score_name} Size': df_results[f'{score_name} Size'].mean() for score_name, _, _ in
                   individual_results}
            }
            return average_results

        elif method == "EFCP":
            step1_indices = step2_indices = step3_indices = train_indices
            optimal_weights, optimal_size, avg_sizes = self.find_optimal_weights(step1_indices, step2_indices, alpha)
            weighted_coverage, weighted_size = self.evaluate_final_performance(step3_indices, test_indices, optimal_weights, alpha)
            print(f"Weighted Combination - Coverage: {weighted_coverage:.4f}, Size: {weighted_size:.4f}")
            indices1 = step3_indices
            indices2 = test_indices
            individual_results = []
            for i, score_function in enumerate(self.score_functions):
                score_name = score_function.__name__
                coverage, size = self.evaluate_final_performance(indices1, indices2, [int(i == j) for j in
                                                                                      range(len(self.score_functions))],
                                                                 alpha)
                print(f"{score_name} - Coverage: {coverage:.4f}, Size: {size:.4f}")
                individual_results.append((score_name, coverage, size))

        elif method == "JaB":
            step1_indices = step3_indices = train_indices
            step2_indices = test_indices
            optimal_weights, optimal_size, avg_sizes = self.find_optimal_weights(step1_indices, step2_indices, alpha)
            weighted_coverage, weighted_size = self.evaluate_final_performance(step3_indices, test_indices, optimal_weights, alpha)
            print(f"Weighted Combination - Coverage: {weighted_coverage:.4f}, Size: {weighted_size:.4f}")
            indices1 = step3_indices
            indices2 = test_indices
            individual_results = []
            for i, score_function in enumerate(self.score_functions):
                score_name = score_function.__name__
                coverage, size = self.evaluate_final_performance(indices1, indices2, [int(i == j) for j in
                                                                                      range(len(self.score_functions))],
                                                                 alpha)
                print(f"{score_name} - Coverage: {coverage:.4f}, Size: {size:.4f}")
                individual_results.append((score_name, coverage, size))

        return {
            'Alpha': alpha,
            'Optimal Weights': optimal_weights,
            'Weighted Combination Coverage': weighted_coverage,
            'Weighted Combination Size': weighted_size,
            'Avg Sizes': np.array(avg_sizes),
            **{f'{score_name} Coverage': coverage for score_name, coverage, _ in individual_results},
            **{f'{score_name} Size': size for score_name, _, size in individual_results}
        }
        # return pd.DataFrame.from_dict({
        #     'Alpha': alpha,
        #     'Optimal Weights': optimal_weights,
        #     'Weighted Combination Coverage': weighted_coverage,
        #     'Weighted Combination Size': weighted_size,
        #     'Avg Sizes': np.array(avg_sizes),
        #     **{f'{score_name} Coverage': coverage for score_name, coverage, _ in individual_results},
        #     **{f'{score_name} Size': size for score_name, _, size in individual_results}
        # }, orient='index')

        # else:
        #     raise ValueError(f"Invalid method: {method}. Supported methods are 'VFCP', 'EFCP', and 'JaB'.")

        # self.visualize_weight_performance(candidate_weights, avg_sizes)


    def compare_performance_multiple_splits(self, num_runs=2, alpha=0.1, train_ratio=0.8):
        results = []
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            # Randomly split the indices into train and test sets
            rng = np.random.default_rng(seed=run)
            indices = rng.permutation(len(self.labels))
            split_idx = int(len(indices) * train_ratio)
            train_indices, test_indices = indices[:split_idx], indices[split_idx:]

            results.append(self.compare_performance(train_indices, test_indices, alpha))
            # self.compare_performance(train_indices, test_indices, alpha, method="EFCP")
            # self.compare_performance(train_indices, test_indices, alpha, method="JaB")

            # Create a DataFrame from the results
        df_results = pd.DataFrame(results) # pd.concat(results, axis=1).transpose()
        # df_results.to_csv('20240703.csv')

        average_results = {
            'Data': self.filename,
            'Alpha': alpha,
            'Optimal Weights': df_results['Optimal Weights'].mean(),
            **{a: df_results[a].mean() for a in df_results.columns if df_results[a].dtype in [np.float64, np.float32, np.float16]}
        }

        return df_results, average_results

        # Calculate the average results
        # average_results = {
        #     'Data': self.filename,
        #     'Alpha': alpha,
        #     'Optimal Weights': df_results['Optimal Weights'].mean(),
        #     'Weighted Combination Coverage': df_results['Weighted Combination Coverage'].mean(),
        #     'Weighted Combination Size': df_results['Weighted Combination Size'].mean()
        # }
        # print(average_results)

    def visualize_weight_performance(self, candidate_weights, avg_sizes):
        # Implement the visualization of weight performance in a simplex plot
        # Each dimension of the simplex represents one of the score functions
        # You can use libraries like matplotlib or plotly for creating the simplex plot
        pass

    def generate_candidate_weights(self, dimension, precision=0.005):
        weights = np.arange(0, 1 + precision, precision)
        candidate_weights = []

        def generate_combinations(current_weights, remaining_sum, index):
            if index == dimension - 1:
                current_weights[index] = remaining_sum
                candidate_weights.append(current_weights.copy())
            else:
                for weight in weights:
                    if weight <= remaining_sum:
                        current_weights[index] = weight
                        generate_combinations(current_weights, remaining_sum - weight, index + 1)

        generate_combinations(np.zeros(dimension), 1, 0)
        return candidate_weights
