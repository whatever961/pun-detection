from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from huggingface_hub import notebook_login
import collections
import copy
notebook_login()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)



ds = load_dataset("frostymelonade/SemEval2017-task7-pun-detection")
ds = ds.remove_columns(["id"])
ds = ds.remove_columns(["type"])

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
tokenized_ds = ds.map(preprocess_function, batched=True)

train_data = []
test_data = []
i = 1
for li in tokenized_ds['test']:
  d = copy.deepcopy(li)
  if i<=1400:
    train_data.append(d)
  elif i>1780 and i<=3580:
    train_data.append(d)
  elif i>1400 and i<=1780:
    test_data.append(d)
  elif i>3580:
    test_data.append(d)
  i+=1

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
id2label = {0: "NOPUN", 1: "PUN"}
label2id = {"NOPUN": 0, "PUN": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "bigscience/bloom-560m", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="report2",
    learning_rate=1.5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
