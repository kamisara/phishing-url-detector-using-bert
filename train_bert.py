# data_bert_pipeline_fixed.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("data/urlset_subset.csv")
df = df[['domain', 'label']]
df.rename(columns={'domain':'URL', 'label':'Label'}, inplace=True)

# -----------------------------
# Step 2: clean the url
# -----------------------------
df['URL'] = df['URL'].str.strip()
df['URL_clean'] = df['URL'].str.split('?').str[0]

# -----------------------------
# Step 3: Convert labels to int(they're float in the csv)
# -----------------------------
df['Label_num'] = df['Label'].astype(int)

# Optional: check dataset
print("Rows after cleaning:", len(df))
print("Label distribution:\n", df['Label_num'].value_counts())
print("Sample URLs:\n", df['URL_clean'].head())

# -----------------------------
# train/test split (could add a train/validate/test ==> improves the model)
# -----------------------------
X = df['URL_clean']
y = df['Label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
#tokenize
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    X_train.tolist(),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='pt'
)

test_encodings = tokenizer(
    X_test.tolist(),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='pt'
)

# -----------------------------
# Pytorch dataset
# -----------------------------
class URLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = URLDataset(train_encodings, y_train.tolist())
test_dataset = URLDataset(test_encodings, y_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -----------------------------
#  Load BERT model
# -----------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# -----------------------------
#  Training loop
# -----------------------------
epochs = 2  # start small
model.train()

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

# -----------------------------
# Step 9: Evaluation
# -----------------------------
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        
        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds)

print("\n✅ Test Accuracy:", accuracy)
print("✅ Test F1 Score:", f1)

# -----------------------------
# Step 10: Save model
# -----------------------------
model.save_pretrained("bert-phishing-model")
tokenizer.save_pretrained("bert-phishing-tokenizer")
print("\nModel & tokenizer saved ✅")
