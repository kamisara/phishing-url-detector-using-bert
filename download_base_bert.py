from transformers import BertTokenizer, BertForSequenceClassification

print("⬇️ Downloading base BERT...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

tokenizer.save_pretrained("base-bert")
model.save_pretrained("base-bert")

print("✅ Base BERT saved locally in ./base-bert")
