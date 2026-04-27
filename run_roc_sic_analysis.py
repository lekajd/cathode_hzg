import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

signal_amts = [10, 20, 30]
colors = ['tab:green', 'tab:cyan', 'tab:purple']
weight_sums = [48323.111415284584, 48588.937048376785, 48687.058292633206]

# Plot fully random ROC curve
plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         "k:",
         label="random")

# Plot CATHODE ROC curves
for i in range(len(signal_amts)):
    cur_df = pd.read_csv(f'complex{signal_amts[i]}_e100_roc_sic_data.csv')

    fpr = cur_df['fpr']
    tpr = cur_df['tpr']

    plt.plot(fpr, tpr, alpha=0.75, color=colors[i], label=f'CATHODE, {signal_amts[i]}% sig')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.legend(loc="lower right")
plt.savefig(f'roc_curve_comparison.png')
print("Figure saved successfully!")
plt.clf()

# Plot fully random SIC curve
with np.errstate(divide='ignore', invalid='ignore'):
    random_tpr = np.linspace(0, 1, 300)
    random_sic = random_tpr / np.sqrt(random_tpr)
plt.plot(random_tpr, random_sic, "k:", label="random")

# Plot CATHODE SIC curves (& print optimal threshold cut)
for i in range(len(signal_amts)):
    cur_df = pd.read_csv(f'complex{signal_amts[i]}_e100_roc_sic_data.csv')

    tpr = cur_df['tpr']
    sic = cur_df['sic']
    thresholds = cur_df['thresholds']

    plt.plot(tpr, sic, alpha=0.75, color=colors[i], label=f'CATHODE, {signal_amts[i]}% sig')

    best_sic_idx = np.argmax(np.nan_to_num(sic, posinf=-1.0))
    best_tpr = tpr[best_sic_idx]
    best_sic = sic[best_sic_idx]
    best_threshold = thresholds[best_sic_idx]
    print(f"For {signal_amts[i]}%...")
    print(f"Best TPR: {best_tpr}")
    print(f"Best SIC: {best_sic}")
    print(f"Best threshold: {best_threshold}")

    predictions_df = pd.read_csv(f'complex{signal_amts[i]}_e100_predictions.csv')

    labels = predictions_df['labels']
    preds = predictions_df['preds']

    above_threshold_mask = preds >= 0.548
    num_signal_above_threshold = sum(labels[above_threshold_mask])
    num_background_above_threshold = len(above_threshold_mask) - num_signal_above_threshold
    print(f"There are {len(above_threshold_mask)} predictions above {0.548}.")
    print(f"{num_signal_above_threshold} of those are signal.")
    print(f"{num_background_above_threshold} of those are background.")

    # signal_percentage = 0.01 * signal_amts[i]
    # # signal_percentage = 0.30

    background_wt_sum = weight_sums[i]
    test_set_signal_perc = sum(labels) / len(labels)
    signal_wt_sum = background_wt_sum * (test_set_signal_perc / (1 - test_set_signal_perc))

    significance = (num_signal_above_threshold / signal_wt_sum) / np.sqrt(num_background_above_threshold / background_wt_sum)
    print(f"Significance: {significance}")

    def significance3(a):
        return (a * num_signal_above_threshold / signal_wt_sum) / np.sqrt(a * num_background_above_threshold / background_wt_sum) - 3.0
    def significance5(a):
        return (a * num_signal_above_threshold / signal_wt_sum) / np.sqrt(a * num_background_above_threshold / background_wt_sum) - 5.0

    x_for_3sigma = fsolve(significance3, 1.0)
    x_for_5sigma = fsolve(significance5, 1.0)
    print(f"Need {x_for_3sigma[0]} times the data for 3 sigma.")
    print(f"Need {x_for_5sigma[0]} times the data for 5 sigma.")

plt.xlabel("True Positive Rate")
plt.ylabel("Significance Improvement")
plt.title("SIC")
plt.legend(loc="lower right")
plt.savefig(f'sic_curve_comparison.png')
print("Figure saved successfully!")
plt.clf()