# Generated from: Twhin_Bert.ipynb
# Converted at: 2026-09-18T05:59:40.146Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install transformers==4.28.0
!pip install datasets

import re
import string
import pandas as pd
import tensorflow as tf

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer ,  AutoModelForSequenceClassification
from transformers import XLMRobertaTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

train = pd.read_excel("offtrain.xlsx")
validation = pd.read_excel("offval.xlsx")
test = pd.read_excel("offtest.xlsx")

train.head()

import datasets
import pandas as pd
from datasets import Dataset, DatasetDict

train_df = pd.DataFrame({
     "text" : train["text"],
     "labels" : train['label']

})
val_df = pd.DataFrame({
     "text" : validation["text"],
     "labels" : validation['label']

})

test_df = pd.DataFrame({
     "text" : test["text"],
     "labels" : test['label']

})

train_dataset = Dataset.from_dict(train_df)
val_dataset = Dataset.from_dict(val_df)
test_dataset = Dataset.from_dict(test_df)
dataset = datasets.DatasetDict({"train":train_dataset,"validation":val_dataset,"test":test_dataset})

dataset

dataset.num_rows

dataset['train'][0]

# #Hyperparameters


num_epoch = 3
sq_len = 128
batch_size = 32
lr_rate = 1e-5
epsilon = 1e-8
hidden_dropout = 0.01
warmup_ratio = 0.01
weight_decay = 0.01

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base") #for larege model replace "Twitter/twhin-bert-large"

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=sq_len, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

dataset

# Now we're ready to train our model...


# ## Training
# 
# We use a pre-trained RoBERTa model from HuggingFace. We use `RobertaModelWithHeads`, a class unique to `adapter-transformers`, which allows us to add and configure prediction heads in a flexibler way.


from transformers import AutoConfig , AutoModelForSequenceClassification
configuration = AutoConfig.from_pretrained('Twitter/twhin-bert-base', num_labels=2) #for larege model replace "Twitter/twhin-bert-large"
configuration.hidden_dropout_prob = hidden_dropout
model = AutoModelForSequenceClassification.from_pretrained('Twitter/twhin-bert-base', config=configuration) #for larege model replace "Twitter/twhin-bert-large"

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", num_trainable_params)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the function to compute the metrics during evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,precision_score,recall_score,confusion_matrix
import time
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro',zero_division=0)
    recall = recall_score(labels, preds, average='macro',zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}



optimizer = AdamW(model.parameters(), lr=lr_rate, eps=epsilon)
total_steps = len(dataset["train"]) * num_epoch
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# optimizer = AdamW(model.parameters(), lr=2e-5,eps = 1e-8)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(dataset["train"])*10)


from datetime import datetime
import pandas as pd

# Create empty lists to store results
epoch_list = []
training_loss_list = []
validation_loss_list = []
training_time_list = []
validation_time_list = []
validation_accuracy_list = []
validation_f1_list = []

test_tp_list = []
test_tn_list = []
test_fp_list = []
test_fn_list = []

test_accuracy_list = []
test_f1_list = []
test_precision_list = []
test_recall_list = []
all_predictions = []

# Define the trainer Arguments
training_args = TrainingArguments(
          output_dir='./results',
          overwrite_output_dir='True',
          num_train_epochs=1,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          warmup_ratio=warmup_ratio,
          weight_decay=weight_decay,
          logging_dir='./logs',
          logging_steps=100,
          save_strategy='no',
          evaluation_strategy='no',
          load_best_model_at_end=True,
          metric_for_best_model='accuracy',
          greater_is_better=True,
          # learning_rate = lr_rate,
          # adam_epsilon = epsilon,

    )

train_epoch = num_epoch
# Train the model and track the metrics after each epoch
for epoch in range(train_epoch):


    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, None)
    )


    # Train the model for one epoch
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()

    # Perform optimization steps
    optimizer.step()
    scheduler.step()

    # Clear gradients
    optimizer.zero_grad()

    # Perform evaluation on the validation set
    validation_result = trainer.evaluate(eval_dataset=dataset["validation"])

    # Perform evaluation on the test set
    test_result = trainer.evaluate(eval_dataset=dataset["test"])
    # Get predictions on the test set
    test_predictions = trainer.predict(dataset["test"]).predictions
    predicted_labels = np.argmax(test_predictions, axis=1)
    predicted_labels = predicted_labels.tolist()

    test_labels = dataset["test"]["labels"].numpy()

    # Print the test labels
    test_labels = test_labels.tolist()
    # Append the predictions and actual labels to the lists
    all_predictions.append({
        'Epoch': epoch + 1,
        'Test Actual': test_labels,
        'Test_Predictions': predicted_labels
    })
    # optimizer.step()
    # scheduler.step()

    # Extract the desired information
    training_loss = train_result.training_loss
    validation_loss = validation_result["eval_loss"]

    validation_time = validation_result["eval_runtime"]
    training_time = train_result.metrics["train_runtime"]

    validation_accuracy = validation_result["eval_accuracy"]
    validation_f1 = validation_result["eval_f1"]


    test_accuracy = test_result["eval_accuracy"]
    test_f1 = test_result["eval_f1"]
    test_precision = test_result["eval_precision"]
    test_recall = test_result["eval_recall"]

    test_tp = test_result["eval_tp"]
    test_tn = test_result["eval_tn"]
    test_fp = test_result["eval_fp"]
    test_fn = test_result["eval_fn"]

    val_runtime_formatted = pd.to_datetime(validation_time, unit='s').strftime('%H:%M:%S')
    train_runtime_formatted = pd.to_datetime(training_time, unit='s').strftime('%H:%M:%S')

    # Append the metrics to the respective lists
    epoch_list.append(epoch + 1)
    training_loss_list.append(training_loss)
    validation_loss_list.append(validation_loss)

    validation_accuracy_list.append(validation_accuracy)
    validation_f1_list.append(validation_f1)

    training_time_list.append(train_runtime_formatted)
    validation_time_list.append(val_runtime_formatted)

    test_tp_list.append(test_tp)
    test_tn_list.append(test_tn)
    test_fp_list.append(test_fp)
    test_fn_list.append(test_fn)


    test_accuracy_list.append(test_accuracy)
    test_f1_list.append(test_f1)
    test_precision_list.append(test_precision)
    test_recall_list.append(test_recall)

# Combine the lists into DataFrames
predictions_df = pd.DataFrame(all_predictions)

# Create a DataFrame to store the results
train_df = pd.DataFrame({
    "Epoch": epoch_list,
    "Training Loss": training_loss_list,
    "Validation Loss": validation_loss_list,
    "Validation Accuracy": validation_accuracy_list,
    "Validation F1": validation_f1_list,
    "Training Time": training_time_list,
    "Validation Time": validation_time_list,
})

test_df = pd.DataFrame({
    "Epoch": epoch_list,
    "Test TP": test_tp_list,
    "Test TN": test_tn_list,
    "Test FP": test_fp_list,
    "Test FN": test_fn_list,
    "Test Accuracy": test_accuracy_list,
    "Test F1": test_f1_list,
    "Test Precision": test_precision_list,
    "Test Recall": test_recall_list,
})

train_df.head(15)

test_df.head(15)

train_df.to_excel("2-MuRIL-Train (1e-5)128SL-32BS-0.01HD-0.01WR-0.01WD.xlsx")

test_df.to_excel("2-MuRIL-Test (1e-5)128SL-32BS-0.01HD-0.01WR-0.01WD.xlsx")

# Save the Prediction label to CSV files
predictions_df.to_csv('2-Predicion MuRIL (1e-5)128SL-32BS-0.01HD-0.01R-0.01WD.csv', index=False)