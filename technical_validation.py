import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import torch
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
from os import listdir
from tqdm import tqdm

#################### PART 1: COMPARISON OF PRETRAINED LLMS ####################
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_pred (tuple): A tuple containing the logits and labels.
    
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, average='weighted')),
        'recall': float(recall_score(labels, predictions, average='weighted')),
        'f1': float(f1_score(labels, predictions, average='weighted'))
    }

# Load dataset from a feather file
data = pd.read_feather("train.feather")
dataset = Dataset.from_pandas(data)

# Define K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define model names for training
model_names = [
    "AI-Growth-Lab/PatentSBERTa",
    "mixedbread-ai/mxbai-embed-large-v1",
    "BAAI/bge-large-en-v1.5"
]

# Dictionary to store evaluation results for each model
results = {}

# Cross-validation and training for each model
for model_name in model_names:
    fold_results = []
    for train_index, val_index in kf.split(dataset):
        # Split data into training and validation sets
        train_ds = dataset.select(train_index)
        eval_ds = dataset.select(val_index)

        # Load the SetFit model
        model = SetFitModel.from_pretrained(model_name)

        # Configure training arguments
        args = TrainingArguments(
            output_dir=f'./results_{model_name}',
            batch_size=4,
            num_train_epochs=1,
        )

        # Initialize the trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            compute_metrics=compute_metrics
        )

        # Train and evaluate the model
        trainer.train()
        eval_result = trainer.evaluate(eval_ds)
        print(model_name, eval_result)
        fold_results.append(eval_result)

    # Store average results across all folds
    avg_metrics = {key: np.mean([dic[key] for dic in fold_results]) for key in fold_results[0]}
    results[model_name] = avg_metrics

# Output the results for all models
print(results)

#################### PART 2: COMPARISON WITH RULE-BASED APPROACHES ####################
# Load patent classification data from multiple files
folder = r"D:\pat_split_patbert"
files = [os.path.join(folder, f) for f in listdir(folder)]
be_pats = pd.DataFrame()

for file in tqdm(files):
    pat = pd.read_feather(file)
    be_pats = pd.concat([be_pats, pat])

# Filter patents with a high probability of being in the bioeconomy
b = be_pats[be_pats["prob_BE"] > 0.5]

# Load additional data: abstracts and CPC code information
abstracts = pd.read_feather(r"\abstracts.feather")
abstracts = abstracts[abstracts["appln_abstract_lg"] == "en"]
tls_224 = pd.read_feather(r"tls_224.feather")

# Merge abstracts with bioeconomy patent data
pat = pd.merge(abstracts, be_pats[["appln_abstract", "prob_BE"]], on="appln_abstract", how="left")
pat = pd.merge(pat, tls_224, on="appln_id", how="left")

# Extract first 4 digits of 'cpc_class_symbol' for classification
pat["cpc"] = pat["cpc_class_symbol"].str[:4]

# Filter patents with high bioeconomy probability and aggregate by CPC
be_pats = pat[pat["prob_BE"] > 0.5]
be_pats_g = be_pats.groupby("cpc").size().reset_index(name="size_be")
pat_g = pat.groupby("cpc").size().reset_index(name="size_all")

# Merge and compute share metrics for bioeconomy patents
merged_g = pd.merge(pat_g, be_pats_g, on="cpc", how="left").fillna(0)
merged_g["share_be"] = (merged_g["size_be"] / merged_g["size_be"].sum()) * 100
merged_g["share_cpc"] = (merged_g["size_be"] / merged_g["size_all"]) * 100
merged_g["total_share"] = (merged_g["size_all"] / sum(merged_g["size_all"])) * 100

# Bioeconomy codes list from Frietsch et al. (2017)
refs=["A01B", "A01C", "A01D", "A01F", "A01G", "A01L", "A01M", "A61D",
"B02B", "B29C", "B29D", "B29K", "B29L", "B99Z", "C03B", "C08J",
"C12L"," C13B5", "C13B15", "C13B25", "C13B45,A01P", "C05B", "C05C", "C05D", "C05F", "C05G",
"D21C", "D21D", "D21H","B31B", "B31C", "B31D", "B31F", "B41B", "B41C", "B41D", "B41F",
"B41G", "B41J", "B41K", "B41L", "B41M", "B41N", "C14B", "D01B",
"D01C", "D01D", "D01F", "D01G", "D01H", "D02G", "D02H",
"D02J", "D03C", "D03D", "D03J", "D04B", "D04C", "D04G",
"D04H", "D05B", "D05C", "D06G", "D06H", "D06J", "D06M",
"D06P", "D06Q", "D21B", "D21C", "D21F", "D21G", "D21J", "D99","A01G23/00", "A01G25/00", "E02D3/00",
"A01H1/06", "C12N15/00", "C12N7/00","E02B3", "E02D", "E02F","A01J", "A01H", "A21D", "A23B", "A23C", "A23D", "A23F", "A23G",
"A23J", "A23K", "A23L", "C12C", "C12F", "C12G", "C12H", "C12J",
"C13D", "C13F", "C13J", "C13K", "C13B10", "C13B20",
"C13B25", "C13B30", "C13B35", "C13B40", "C13B50",
"C13B99","A21B", "A21C", "A22B", "A22C", "A23N", "A23P","C07K","C10L5/40", "C10L5/42", "C10L5/44", "C10L5/46",
"C10L5/48", "C10B53/02",
"A01C3/02", "C02F11/04", "C05F17/02", "B01D53/84",
"F23G7/10","C08B", "C08C", "C08H", "C09F", "C11B", "C11C", "C13B", "A01N",
"D21H", "C08L1", "C08L3", "C08L5", "C08L7", "C09J101",
"C09J103", "C09J105", "C09J107", "C09K17", "A61K36/02",
"A61K36/03", "A61K36/04", "A61K36/05","A01H15","A01H1/00", "A01H4/00", "A61K38/00", "A61K39/00",
"A61K48/00", "C02F3/34", "C07G11/00", "C07G13/00",
"C07G15/00", "C07K4/00","C07K14/00", "C07K16/00",
"C07K17/00", "C07K19/00", "G01N33","G01N3353", "G01N3354", "G01N3355", "G01N3357",
"G01N3368", "G01N3374", "G01N3376", "G01N3378", "G01N3388", "G01N3392","C07C29", "C07D475", "C07K2", "C08B3", "C08B7", "C08H1",
"C08L89", "C09D11", "C09D189", "C09J189","A01K", "A01M", "A22B", "A61D", "A23N17","F25D", "A21B", "A47J" 
]

# Load and merge IPC code data for further filtering
tls_209 = pd.concat([
    pd.read_csv(r"tls209_part01.csv"),
    pd.read_csv(r"tls209_part02.csv")
])
pat = pd.merge(pat, tls_209[["ipc_class_symbol", "appln_id"]], on="appln_id", how="left")

# Filter patents matching specified CPC codes
matching_patents = pat[pat["ipc_class_symbol"].apply(lambda x: any(code in x for code in refs))]

required_cpc_codes = ['G01N27/327', 'C12M', 'C12N', 'C12P', 'C12Q', 'C12S']
excluded_cpc_code = 'A61K'
filtered_df = pat[pat['ipc_class_symbol'].apply(lambda x: any(code in x for code in required_cpc_codes) and excluded_cpc_code not in x)]

# Combine rule-based and NLP approach results
matching_patents = pd.concat([matching_patents, filtered_df])

# Compare patent sets using a Venn diagram
set1 = set(matching_patents["appln_id"])
set2 = set(b["appln_id"])
common_elements = set1.intersection(set2)

plt.rcParams['font.family'] = 'Calibri'
plt.figure(figsize=(8, 6))
venn = venn2([set1, set2], ('Rule-based approach', 'NLP-approach'), set_colors=('skyblue', 'lightgreen'))

# Format Venn diagram labels with commas for thousands separators
for label in venn.set_labels:
    if label:
        label.set_fontsize(12)
for label in venn.subset_labels:
    if label:
        label_text = label.get_text()
        if label_text.isdigit():
            label.set_text("{:,}".format(int(label_text)))

#################### PART 3: STRATIFIED VALIDATION PER CPC SECTION ####################
# Define overall performance metrics from active learning
overall_metrics = {
    'Accuracy': 0.9428,
    'Precision': 0.9389,
    'Recall': 0.9096,
    'F1-Score': 0.9226
}

def compute_metrics(sample):
    """
    Compute evaluation metrics for a given sample.
    
    Args:
        sample (DataFrame): DataFrame containing 'pred' and 'validation' columns.
    
    Returns:
        dict: A dictionary with accuracy, precision, recall, and F1 score.
    """
    y_pred = sample['pred']
    y_true = sample['validation']
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

# Load validation datasets for each CPC class
samples = {cls: pd.read_excel(f"\data\sample_{cls.lower()}.xlsx") for cls in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']}

# Compute and plot deviations from overall metrics
sample_metrics = {cls: compute_metrics(df) for cls, df in samples.items()}
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for i, (cpc_class, metrics_dict) in enumerate(sample_metrics.items()):
    x = np.arange(len(metrics))
    deviations = [metrics_dict[m] - overall_metrics[m] for m in metrics]
    axes[i].bar(x, deviations, color='skyblue')
    axes[i].set_title(cpc_class, weight="bold", fontsize=14)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(metrics, fontsize=12)
    axes[i].axhline(0, color='black', linestyle='--')
    axes[i].set_ylim(-0.2, 0.2)
    axes[i].set_ylabel('Deviation')

plt.tight_layout()
plt.suptitle("Performance Deviations Across CPC Classes", weight="bold", fontsize=18)
plt.show()
