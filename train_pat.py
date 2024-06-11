# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:31:19 2024

@author: tesla
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
torch.cuda.is_available()

def compute_metrics(y_pred, y_test):
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

data = pd.read_excel(r"\train_pat_be.xlsx")

train, eval_ds=train_test_split(data, test_size=0.3, random_state=42)

eval_ds=Dataset.from_pandas(eval_ds)

train_ds=Dataset.from_pandas(train)

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

args = TrainingArguments(
    batch_size=4,
    num_epochs=1,
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    #load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    metric=compute_metrics,
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate(eval_ds)
print(metrics)
