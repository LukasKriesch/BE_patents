# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:40:26 2024

@author: tesla
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import torch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, average='weighted')),
        'recall': float(recall_score(labels, predictions, average='weighted')),
        'f1': float(f1_score(labels, predictions, average='weighted'))
    }

# Load data
data = pd.read_feather(r"\data\train.feather")
dataset = Dataset.from_pandas(data)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models to train
model_names = [
    "AI-Growth-Lab/PatentSBERTa",
    "mixedbread-ai/mxbai-embed-large-v1",  # Replace with actual model names
    "BAAI/bge-large-en-v1.5"
]

# Results storage
results = {}

# Cross-validation across each model
for model_name in model_names:
    fold_results = []
    for train_index, val_index in kf.split(dataset):
        # Splitting data into training and validation sets
        train_ds = dataset.select(train_index)
        eval_ds = dataset.select(val_index)

        # Load a SetFit model from Hub
        model = SetFitModel.from_pretrained(model_name)

        # Setup training arguments
        args = TrainingArguments(
            output_dir=f'./results_{model_name}',    # Output directory for each model
            #evaluation_strategy="epoch",
            #save_strategy="epoch",
            #load_best_model_at_end=True,
            batch_size=4,
            num_train_epochs=1,
            #logging_dir=f'./logs_{model_name}',      # Logging directory for each model
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
        print(model_name)
        print(eval_result)
        fold_results.append(eval_result)

    # Store average results from folds for each model
    avg_metrics = {key: np.mean([dic[key] for dic in fold_results]) for key in fold_results[0]}
    results[model_name] = avg_metrics

# Print the results for all models
print(results)

import pandas as pd
import os
from os import listdir
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# read classified patent abstracts
folder=r"D:\pat_split_patbert"

files=[os.path.join(folder,f) for f in listdir(folder)]
be_pats=pd.DataFrame()

for file in tqdm(files):
    pat=pd.read_feather(file)
    #pat=pat[pat["prob_bioeconomy_new"]>0.5]
    be_pats=pd.concat([be_pats,pat])

#read abstracts
abstracts=pd.read_feather(r"\abstracts.feather")
#select patents with english abstracts
abstracts=abstracts[abstracts["appln_abstract_lg"]=="en"]
#read cpc code data
tls_224=pd.read_feather(r"tls_224.feather")

# Step 1: Merge abstracts with be_pats on appln_abstract
pat = pd.merge(abstracts, be_pats[["appln_abstract", "prob_BE"]], on="appln_abstract", how="left")

# Step 2: Merge the resulting dataframe with tls_224 on appln_id
pat = pd.merge(pat, tls_224, on="appln_id", how="left")

# Step 3: Create the 'cpc' column with the first 4 digits of 'cpc_class_symbol'
pat["cpc"] = pat["cpc_class_symbol"].str[:4]

# Step 4: Filter entries with prob_BE_mixedbread higher than 0.5
be_pats = pat[pat["prob_BE"] > 0.5]

# Step 5: Group by 'cpc' and compute the size of each group for high prob_BE patents
be_pats_g = be_pats.groupby("cpc").size().reset_index(name="size_be")

# Step 6: Group by 'cpc' and compute the size of each group for all patents
pat_g = pat.groupby("cpc").size().reset_index(name="size_all")

# Step 7: Merge the two groupby results on 'cpc'
merged_g = pd.merge(pat_g, be_pats_g, on="cpc", how="left")

# Step 8: Compute the share of high prob_BE patents among all high prob_BE patents
merged_g["share_be"] = merged_g["size_be"] / merged_g["size_be"].sum()

# Step 9: Compute the share of high prob_BE patents within each 'cpc' class
merged_g["share_cpc"] = merged_g["size_be"] / merged_g["size_all"]

# Step 10: Fill NaN values with 0 (for cases where a cpc class has no high prob_BE patents)
merged_g = merged_g.fillna(0)

len(merged_g[merged_g["size_be"]==0])
merged_g["total_share"]=merged_g["size_all"]/sum(merged_g["size_all"])

merged_g["total_share"]=merged_g["total_share"]*100

merged_g["share_be"]=merged_g["share_be"]*100

#bioeconomy ipc codes as Frietsch et al. (2017)
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

#read and merge ipc code data
tls_209=pd.read_csv(r"tls209_part01.csv")
tls_209_2=pd.read_csv(r"tls209_part02.csv")
tls_209=pd.concat([tls_209,tls_209_2])
pat=pd.merge(pat,tls_209[["ipc_class_symbol","appln_id"]],on="appln_id",how="left")
pat["ipc_class_symbol"]=pat["ipc_class_symbol"].astype(str)

# Function to check if a CPC code matches any code in the list
def cpc_code_matches(cpc_code, cpc_code_list):
    for code in cpc_code_list:
        if code in cpc_code:
            return True
    return False

# Filter the DataFrame to extract patents with matching CPC codes
matching_patents = pat[pat["ipc_class_symbol"].apply(lambda x: cpc_code_matches(x, refs))]

required_cpc_codes = ['G01N27/327', 'C12M', 'C12N', 'C12P', 'C12Q', 'C12S']
excluded_cpc_code = 'A61K'

# Function to check if any of the required CPC codes are present and the excluded CPC code is absent
def match_cpc_codes(cpc_code):
    return any(code in cpc_code for code in required_cpc_codes) and excluded_cpc_code not in cpc_code

# Apply the function to filter the DataFrame
filtered_df = pat[pat['ipc_class_symbol'].apply(match_cpc_codes)]

matching_patents=pd.concat([matching_patents,filtered_df])

list1=list(matching_patents["appln_id"])
b=pat[pat["prob_BE"]>0.5]

list2=list(b["appln_id"])
# Convert lists to sets for easy comparison
set1 = set(list1)
set2 = set(list2)

# Find common elements
common_elements = set1.intersection(set2)

# Find unique elements in each list
unique_to_list1 = set1 - set2
unique_to_list2 = set2 - set1

plt.rcParams['font.family'] = 'Calibri'

# Create a Venn diagram with customized colors
plt.figure(figsize=(8, 6))
venn = venn2([set1, set2], ('Rule-based approach', 'NLP-approach'), set_colors=('skyblue', 'lightgreen'))

# Format the labels with commas as thousands separators
for label in venn.set_labels:
    if label:
        label.set_fontsize(12)  # Adjust font size if necessary
for label in venn.subset_labels:
    if label:
        label_text = label.get_text()
        if label_text.isdigit():  # Check if the label is a digit
            formatted_text = "{:,}".format(int(label_text))
            label.set_text(formatted_text)

