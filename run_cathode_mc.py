import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys

from os.path import exists, join, dirname, realpath
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

parent_dir = dirname(realpath(__file__))
sys.path.append(parent_dir)
sys.path.append(realpath('sk_cathode'))

from sk_cathode.generative_models.conditional_normalizing_flow_torch import ConditionalNormalizingFlow
from sk_cathode.classifier_models.neural_network_classifier import NeuralNetworkClassifier
from sk_cathode.utils.preprocessing import LogitScaler

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--filename",
    type=str,
    required=False,
    help="Memorable filename for saving/loading trained models"
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    required=True,
    help="Number of epochs"
)
parser.add_argument(
    "-s",
    "--signal_percentage",
    type=float,
    required=False,
    help="Percentage of SR events that will be signal (the rest will be background)"
)

args = parser.parse_args()
epochs = args.epochs
if args.filename == None: save_tag = f"mc_trained_e{epochs}"
else:                     save_tag = f"{args.filename}_e{epochs}"

if args.signal_percentage == None: desired_signal_percentage = 0.1
else:                              desired_signal_percentage = args.signal_percentage


""" DATA PREPARATION """

import uproot as ur

## Define file paths
# Background
dy_path  = "mc/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_step1.root"
llg_path  = "mc/ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_step1.root"
# Signal
hzg_path = "mc/GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_step1.root"

# Get files from paths
dy_file  = ur.open(dy_path)
llg_file  = ur.open(llg_path)
hzg_file = ur.open(hzg_path)

# Get trees from files
dy_tree  = dy_file["Events"]
llg_tree  = llg_file["Events"]
hzg_tree = hzg_file["Events"]

# Convert trees to dictionaries of arrays
variables = ["ffG_mass_kr", "G_mvaID", "G_pt", "ptG_to_M3b"]
dy  = dy_tree.arrays(variables, library="np")
llg  = llg_tree.arrays(variables, library="np")
hzg = hzg_tree.arrays(variables, library="np")

### COMPUTE MC WEIGHTS ###
# DY
xsec = np.array(dy_tree["xsec"])
lumi = np.array(dy_tree["lumi"])
gen_weight = np.array(dy_tree["genWeight"])
gen_weight_sum = np.array(dy_file["Runs"]["genEventSumw_batch"]).sum()
dy_weights = (xsec * lumi * gen_weight) / gen_weight_sum
# LLG
xsec = np.array(llg_tree["xsec"])
lumi = np.array(llg_tree["lumi"])
gen_weight = np.array(llg_tree["genWeight"])
gen_weight_sum = np.array(llg_file["Runs"]["genEventSumw_batch"]).sum()
llg_weights = (xsec * lumi * gen_weight) / gen_weight_sum
# HZG
xsec = np.array(hzg_tree["xsec"])
lumi = np.array(hzg_tree["lumi"])
gen_weight = np.array(hzg_tree["genWeight"])
gen_weight_sum = np.array(hzg_file["Runs"]["genEventSumw_batch"]).sum()
hzg_weights = (xsec * lumi * gen_weight) / gen_weight_sum
hzg_weights = hzg_weights[hzg_weights >= 0] # <--remove negative-weight events from signal

### Now we combine the background MCs, resample using probabilities ∝ weights,
### & apply an 80/20 train/test split

# Combine DY & LLG, along with their weights
bg_combined = {var: np.concatenate([dy[var], llg[var]]) for var in variables}
bg_weights_combined = np.concatenate([dy_weights, llg_weights])
bg_weights_combined = bg_weights_combined[bg_weights_combined >= 0] # <-- remove negative-weight events from background

# Weights --> Probabilities (normalized)
bg_probs = bg_weights_combined / np.sum(bg_weights_combined)
# Resample by index
bg_resampled_idx = np.random.choice(range(len(bg_weights_combined)),
                                    size=len(bg_weights_combined), # <-- keep same number of events
                                    p=bg_probs)

# Train/test split
bg_split_point = int(0.8 * len(bg_resampled_idx))
bg_train_idx = bg_resampled_idx[:bg_split_point]
bg_test_idx = bg_resampled_idx[bg_split_point:]

# Get background dictionaries (train and test) from resampled indices
bg_train = {var: np.array(arr[bg_train_idx]) for var, arr in bg_combined.items()}
bg_test = {var: np.array(arr[bg_test_idx]) for var, arr in bg_combined.items()}
bg_test_weights = bg_weights_combined[bg_test_idx]

### Similarly, we resample HZG according to its own weights
### & apply an 80/20 train/test split
hzg_probs = hzg_weights / np.sum(hzg_weights)
hzg_resampled_idx = np.random.choice(range(len(hzg_weights)),
                                     size=len(hzg_weights),
                                     p=hzg_probs)

# Train/test split
hzg_split_point = int(0.8 * len(hzg_resampled_idx))
hzg_train_idx = hzg_resampled_idx[:hzg_split_point]
hzg_test_idx = hzg_resampled_idx[hzg_split_point:]

# Get signal dictionaries (train and test) from resampled indices
sig_train = {var: np.array(arr[hzg_train_idx]) for var, arr in hzg.items()}
sig_train_weights = hzg_weights[hzg_train_idx]
sig_test = {var: np.array(arr[hzg_test_idx]) for var, arr in hzg.items()}

### Enforce the 'signal percentage' hyperparameter in the SR

sb_mask_bg_train = (bg_train["ffG_mass_kr"] < 120) | (bg_train["ffG_mass_kr"] > 130)
sb_mask_sig_train = (sig_train["ffG_mass_kr"] < 120) | (sig_train["ffG_mass_kr"] > 130)
sr_mask_bg_train = (bg_train["ffG_mass_kr"] >= 120) & (bg_train["ffG_mass_kr"] <= 130)
sr_mask_sig_train = (sig_train["ffG_mass_kr"] >= 120) & (sig_train["ffG_mass_kr"] <= 130)

sb_bg = {var: arr[sb_mask_bg_train] for var, arr in bg_train.items()}
sb_sig = {var: arr[sb_mask_sig_train] for var, arr in sig_train.items()}
sr_bg = {var: arr[sr_mask_bg_train] for var, arr in bg_train.items()}
sr_sig = {var: arr[sr_mask_sig_train] for var, arr in sig_train.items()}

sr_sig_weights = sig_train_weights[sr_mask_sig_train]

num_sr_bg_events = len(sr_bg["ffG_mass_kr"])
num_sr_sig_events = len(sr_sig["ffG_mass_kr"])

current_signal_percentage = num_sr_sig_events / (num_sr_bg_events + num_sr_sig_events)
desired_background_percentage = 1 - desired_signal_percentage
assert desired_background_percentage < 1

new_num_sr_sig_events = int(num_sr_bg_events * (desired_signal_percentage / desired_background_percentage))

if new_num_sr_sig_events < num_sr_sig_events:
    # Need to resample fewer signal events
    num_sig_to_resample = int(num_sr_bg_events * (desired_signal_percentage / desired_background_percentage))
    assert(num_sig_to_resample <= num_sr_sig_events)
    resample_idx = np.random.choice(range(num_sr_sig_events),
                                    size=num_sig_to_resample,
                                    replace=False)
    sr_sig = {var: np.array(arr[resample_idx]) for var, arr in sr_sig.items()}

elif new_num_sr_sig_events > num_sr_sig_events:
    # Need to sample more signal events (using weights as probabilities like before)
    num_sig_to_add = new_num_sr_sig_events - num_sr_sig_events
    sr_sig_probs = sr_sig_weights / np.sum(sr_sig_weights)

    new_sig_idx = np.random.choice(range(len(sr_sig_weights)),
                                   size=num_sig_to_add,
                                   p=sr_sig_probs)
    new_sig = {var: np.array(arr[new_sig_idx]) for var, arr in sr_sig.items()}
    sr_sig = {var: np.concatenate((arr, new_sig[var])) for var, arr in sr_sig.items()}

print("SR SIGNAL RATIO COMPARISON ----------")
print(f"Desired ratio: {desired_signal_percentage}")
print(f"Before: {current_signal_percentage}")
print(f"After: {len(sr_sig['ffG_mass_kr']) / (len(sr_bg['ffG_mass_kr']) + len(sr_sig['ffG_mass_kr']))}")

#   Converts var:arr dicts into 2D arrays (events as rows, variables as columns)
def build_array(dataset, variables):
    # print(type(dataset))
    # separate invariant mass from input features:
    mass_col = dataset["ffG_mass_kr"].reshape((-1, 1))
    X = np.column_stack([dataset[feature] for feature in variables if feature != "ffG_mass_kr"])

    return np.hstack([mass_col, X])
    # 0th column is mass, the rest are features

# Get arrays for SB and SR, combining background and signal in the SR
sb = build_array(sb_bg, variables)
sr = build_array({var: np.concatenate([sr_bg[var], sr_sig[var]]) for var in variables}, variables)


""" DENSITY ESTIMATION """

# We split the training set further into training (75%) & validation (25%)
from sklearn.model_selection import train_test_split
sb_train, sb_val = train_test_split(sb, test_size=0.25)
sr_train, sr_val = train_test_split(sr, test_size=0.25)

# Create scaler pipeline
sb_scaler = make_pipeline(QuantileTransformer(output_distribution='normal', n_quantiles=1000, copy=True), StandardScaler())

# Separate invariant mass column from features & apply scaler
m_train = sb_train[:, 0:1]
X_train = sb_scaler.fit_transform(sb_train[:, 1:])
m_val = sb_val[:, 0:1]
X_val = sb_scaler.transform(sb_val[:, 1:])

# Remove any NaNs or infinities from train & val sets
finite_mask_train = np.isfinite(X_train).all(axis=1)
X_train = X_train[finite_mask_train]
m_train = m_train[finite_mask_train]

finite_mask_val = np.isfinite(X_val).all(axis=1)
X_val = X_val[finite_mask_val]
m_val = m_val[finite_mask_val]

### Finally, train the conditional normalizing flow (CNF)

print(f"Density estimator will be trained on {X_train.shape[0]} events.")
flow_savedir = f"./{save_tag}_flows/"
if not exists(join(flow_savedir, "DE_models")):
    flow_model = ConditionalNormalizingFlow(save_path=flow_savedir,
                                            num_inputs=sb_train[:, 1:].shape[1],
                                            early_stopping=True, epochs=epochs, # <-------------- [may change]
                                            verbose=True)
    flow_model.fit(X_train, m_train, X_val, m_val)
else:
    print(f"Loading existing model from {flow_savedir}.")
    flow_model = ConditionalNormalizingFlow(save_path=flow_savedir,
                                            num_inputs=sb_train[:, 1:].shape[1],
                                            early_stopping=True, epochs=epochs,
                                            load=True)
print("Success!")

### Now, we sample background-like events in the SR

## Train a kernel density estimator (KDE) for invariant mass sampling

# Apply logit transform, then train KDE
m_scaler = LogitScaler(epsilon=1e-8)
m_train = m_scaler.fit_transform(sr_train[:, 0:1])

print("Training the KDE & generating samples...")

kde_model = KernelDensity(bandwidth=0.01, kernel='gaussian')
kde_model.fit(m_train)

# Sample 500,000 new events
# m_samples = kde_model.sample(4*len(m_train)).astype(np.float32)
m_samples = kde_model.sample(500000).astype(np.float32)
m_samples = m_scaler.inverse_transform(m_samples)

## Now sample from the CNF using these KDE samples as conditional
X_samples = flow_model.sample(n_samples=len(m_samples), m=m_samples)

# Remove any NaNs or infinities from the generated samples
finite_mask = np.isfinite(X_samples).all(axis=1)
X_samples = X_samples[finite_mask]
m_samples = m_samples[finite_mask]

print("Samples successfully generated!")

# Invert the scaler
X_samples = sb_scaler.inverse_transform(X_samples)

""" CLASSIFICATION """

# Assign label 0 ("background-like") to generative bg
c_gen_bg = np.hstack([m_samples, X_samples, np.zeros((m_samples.shape[0], 1))])

# Assign label 1 ("signal-like") to data
c_train_data = sr_train.copy()
c_train_data = np.hstack([c_train_data, np.ones((c_train_data.shape[0], 1))])

c_val_data = sr_val.copy()
c_val_data = np.hstack([c_val_data, np.ones((c_val_data.shape[0], 1))])

# Mix data & gen bg into combined train & val for the classifier
n_train = len(c_train_data)
n_val = len(c_val_data)
n_train_samples = int(n_train / (n_train + n_val) * len(c_gen_bg))
c_train_samples = c_gen_bg[:n_train_samples]
c_val_samples = c_gen_bg[n_train_samples:]

print("Classifier will be trained on:")
print(f"{c_train_samples.shape[0]} generative background events")
print(f"{c_train_data.shape[0]} data events")

c_train = np.vstack([c_train_data, c_train_samples])
c_train = shuffle(c_train, random_state=42)
c_val = np.vstack([c_val_data, c_val_samples])
c_val = shuffle(c_val, random_state=42)

# Create scaler based only on data (for consistency under potential resampling)
sr_scaler = StandardScaler()
sr_scaler.fit(c_train_data[:, 1:-1])

# Define X (features) & y (labels)
X_train = sr_scaler.transform(c_train[:, 1:-1])
y_train = c_train[:, -1]
X_val = sr_scaler.transform(c_val[:, 1:-1])
y_val = c_val[:, -1]

### Train neural-network classifier

c_savedir = f"./{save_tag}_classifiers/"
# Protect against overwriting, as before [may change]
if not exists(join(c_savedir, "CLSF_models")):
    classifier_model = NeuralNetworkClassifier(save_path=c_savedir,
                                               n_inputs=X_train.shape[1],
                                               early_stopping=True, epochs=epochs, # <-------- [may change]
                                               verbose=True)
    classifier_model.fit(X_train, y_train, X_val, y_val)
else:
    print(f"Loading existing model from {c_savedir}.")
    classifier_model = NeuralNetworkClassifier(save_path=c_savedir,
                                               n_inputs=X_train.shape[1],
                                               early_stopping=True, epochs=epochs,
                                               load=True)
print("Success!")
                                            
""" ANALYSIS """

# First, sample some new generative bg for testing purposes

# Sample realistic mass values
m_samples_test = kde_model.sample(bg_test_idx.shape[0]).astype(np.float32)
m_samples_test = m_scaler.inverse_transform(m_samples_test)

## Now sample from the CNF using these KDE samples as conditional
X_samples_test = flow_model.sample(n_samples=len(m_samples_test), m=m_samples_test)
finite_mask = np.isfinite(X_samples_test).all(axis=1)
X_samples_test = X_samples_test[finite_mask]
m_samples_test = m_samples_test[finite_mask]
X_samples_test = sb_scaler.inverse_transform(X_samples_test)

# ------------------------------------------
# Plot histograms of generative bg vs. MC bg vs. MC sig

# Select only SR MC bg
sr_mask_bg_test = (bg_test["ffG_mass_kr"] >= 120) & (bg_test["ffG_mass_kr"] <= 130)
bg_test_sr = {var: np.array(arr[sr_mask_bg_test]) for var, arr in bg_test.items()}

# Select only SR MC sig
sr_mask_sig_test = (sig_test["ffG_mass_kr"] >= 120) & (sig_test["ffG_mass_kr"] <= 130)
sig_test_sr = {var: np.array(arr[sr_mask_sig_test]) for var, arr in sig_test.items()}

ranges = [(-0.3, 1.1), (0, 120), (0, 0.8)]
for j in range(X_samples_test.shape[1]):
    variable_j = variables[1 + j] # excluding mass variable
    print(f"Plotting comparison histograms for variable `{variable_j}`...")
    gen_bg_j = X_samples_test[:, j]
    mc_bg_j = bg_test_sr[variable_j]
    mc_sig_j = sig_test_sr[variable_j]

    ks_stat, ks_pval = ks_2samp(gen_bg_j, mc_bg_j)
    print(f"GEN BG vs. MC BG: K–S statistic for {variable_j} is {ks_stat:.4f}, with p-value {ks_pval:.4e}.")
    ks_stat, ks_pval = ks_2samp(gen_bg_j, mc_sig_j)
    print(f"GEN BG vs. MC SIG: K–S statistic for {variable_j} is {ks_stat:.4f}, with p-value {ks_pval:.4e}.")

    plt.hist(mc_sig_j, bins=50, range=ranges[j], density=True, alpha=0.5, histtype='step', linewidth=2, label='MC Signal', color='red')
    plt.hist(gen_bg_j, bins=50, range=ranges[j], density=True, alpha=0.75, histtype='step', linewidth=2, label='Generative Background', color='royalblue')
    plt.hist(mc_bg_j, bins=50, range=ranges[j], density=True, alpha=0.75, histtype='step', linewidth=2, label='MC Background', color='darkorange')
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title(variable_j)
    plt.legend()
    plt.savefig(f'{save_tag}_gen_vs_mc_{variable_j}.png')
    plt.clf()


# Now we test the classifier's predictions for gen bg, MC bg, and MC signal in the SR
# 1. Apply an SR mask to bg & sig
bg_sr_mask = (bg_test["ffG_mass_kr"] >= 120) & (bg_test["ffG_mass_kr"] <= 130)
bg_test = {var: arr[bg_sr_mask] for var, arr in bg_test.items()}

# Print weighted SR background sum for later analysis
bg_test_weights = bg_test_weights[bg_sr_mask]
print(f"Sum of SR background weights = {sum(bg_test_weights)} (for reference)")

sig_sr_mask = (sig_test["ffG_mass_kr"] >= 120) & (sig_test["ffG_mass_kr"] <= 130)
sig_test = {var: arr[sig_sr_mask] for var, arr in sig_test.items()}

# 2. Convert bg & sig to arrays and add true labels column
bg_test = build_array(bg_test, variables)
sig_test = build_array(sig_test, variables)
bg_test = np.hstack([bg_test, np.zeros((bg_test.shape[0], 1))])
sig_test = np.hstack([sig_test, np.ones((sig_test.shape[0], 1))])

# 3. Get gen bg samples
X_samples_test = sr_scaler.transform(X_samples_test)
y_samples_test = np.zeros(X_samples_test[:, -1].shape)
sample_preds = classifier_model.predict(X_samples_test)

# 4. Get classifier predictions and plot on histogram
X_bg = sr_scaler.transform(bg_test[:, 1:-1])
X_sig = sr_scaler.transform(sig_test[:, 1:-1])
bg_preds = classifier_model.predict(X_bg)
sig_preds = classifier_model.predict(X_sig)

plt.hist(sample_preds, bins=30, density=True, range=(0.0, 1.0), alpha=0.75, histtype='step', linewidth=2, label='GEN BG predictions', color='royalblue')
plt.hist(bg_preds, bins=30, density=True, range=(0.0, 1.0), alpha=0.75, histtype='step', linewidth=2, label='MC BG predictions', color='darkorange')
plt.hist(sig_preds, bins=30, density=True, range=(0.0, 1.0), alpha=0.75, histtype='step', linewidth=2, label='MC SIG predictions', color='red')
plt.legend()
plt.xlabel("Classifier Prediction")
plt.ylabel("Probability Density")
plt.savefig(f'{save_tag}_all_preds.png')
plt.clf()

# 5. Obtain ROC curve, compute AUC score, & plot significance improvement
combined_bg_sig = np.vstack((bg_test, sig_test))
labels = combined_bg_sig[:, -1]
preds = np.concatenate((bg_preds, sig_preds))

labels = labels.reshape((combined_bg_sig.shape[0],))
preds = preds.reshape((combined_bg_sig.shape[0],))

with np.errstate(divide='ignore', invalid='ignore'):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    sic = tpr / np.sqrt(fpr)

    random_tpr = np.linspace(0, 1, 300)
    random_sic = random_tpr / np.sqrt(random_tpr)

    auc = roc_auc_score(labels, preds)

# ROC curve:
plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         color="black",
         label="random")
plt.plot(fpr, tpr, label="CATHODE")
plt.xlabel("False Positive Rate")
plt.xlabel("True Positive Rate")
plt.title(f'AUC = {str(round(auc, 3))}')
plt.legend(loc="lower right")
plt.savefig(f'{save_tag}_mcbg_vs_sig_roc.png')
plt.clf()

plt.plot(tpr, sic, label="CATHODE")
plt.plot(random_tpr, random_sic, "w:", label="random")
plt.xlabel("True Positive Rate")
plt.ylabel("Significance Improvement")
# plt.title(f'AUC = {str(round(auc, 3))}')
plt.legend(loc="lower right")
plt.savefig(f'{save_tag}_mcbg_vs_sig_sic.png')
plt.clf()

# Save data as CSV file
df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'sic': sic})
df.to_csv(f'{save_tag}_roc_sic_data.csv', index=False)
df = pd.DataFrame({'labels': labels, 'preds': preds})
df.to_csv(f'{save_tag}_predictions.csv', index=False)

##############
# For reference, training & validation loss plots:

# Load NumPy files
de_train_loss = pd.DataFrame(np.load(f'{save_tag}_flows/DE_train_losses.npy'))
de_val_loss = pd.DataFrame(np.load(f'{save_tag}_flows/DE_val_losses.npy'))
clsf_train_loss = pd.DataFrame(np.load(f'{save_tag}_classifiers/CLSF_train_losses.npy'))
clsf_val_loss = pd.DataFrame(np.load(f'{save_tag}_classifiers/CLSF_val_losses.npy'))

plt.plot(de_train_loss, label='Training Loss')
plt.plot(de_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Density Estimator")
plt.ylim(-1, 5)
plt.savefig(f'{save_tag}_de_loss.png')
plt.clf()

plt.plot(clsf_train_loss, label='Training Loss')
plt.plot(clsf_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Classifier")
plt.savefig(f'{save_tag}_clsf_loss.png')
plt.clf()
