from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


# BERT_PATH = './bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# print(tokenizer.tokenize('I have a good time, thank you.'))
# bert = BertModel.from_pretrained(BERT_PATH)
# print('load bert model over')
# example_text = 'I will watch Memento tonight'
# bert_input = tokenizer(example_text, padding='max_length',
#                        max_length=10,
#                        truncation=True,
#                        return_tensors="pt")
# example_text = tokenizer.decode(bert_input.input_ids[0])
# print(example_text)
#
#
# # 加载预训练的BERT模型和分词器
# tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('./bert-base-uncased')
#
# # 准备输入数据
# inputs = tokenizer("Hello, how are you?", return_tensors="pt")
#
# # 前向传播
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1, label set as 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
#
# print("1:",outputs)
#
# # 继续使用上面的模型和分词器
# inputs = tokenizer("I love programming.", return_tensors="pt")
#
# # 判断情感
# outputs = model(**inputs)
# logits = outputs.logits
# predictions = torch.softmax(logits, dim=-1)
# print("2:",predictions)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


texts = ["I love programming", "I hate bugs", "I dislike apples", "I hate pigs", "I enjoy walking", "I hate faults"]
labels = [1, 0, 0, 0, 1, 0]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = TextClassificationDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch + 1} completed')

test_sentences = ["I enjoy coding", "I hate errors"]
test_encodings = tokenizer(test_sentences, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
input_ids = test_encodings['input_ids']
attention_mask = test_encodings['attention_mask']
model.eval()

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

predictions = torch.argmax(outputs.logits, dim=1)

for i, sentence in enumerate(test_sentences):
    sentiment = "Positive" if predictions[i] == 1 else "Negative"
    print(f"Sentence: {sentence} | Sentiment: {sentiment}")
