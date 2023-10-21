import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


class imdbDataset(Dataset):
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


'''****************************************** data preprocess ****************************************** '''

imdb = pd.read_csv("imdb_master.csv", encoding='latin-1')
original_traindata = imdb[(imdb['type'] == 'train') & (imdb['label'] != 'unsup')].replace({'pos': 1, 'neg': 0})
original_testdata = imdb[(imdb['type'] == 'test') & (imdb['label'] != 'unsup')].replace({'pos': 1, 'neg': 0})
random_indices_for_train = random.sample(range(len(original_traindata)), 8192)
random_indices_for_test = random.sample(range(len(original_testdata)), 512)
traindata = original_traindata.iloc[random_indices_for_train]
testdata = original_testdata.iloc[random_indices_for_test]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
dataset = imdbDataset(traindata['review'].tolist(), traindata['label'].tolist(), tokenizer)
test_dataset = imdbDataset(testdata['review'].tolist(), testdata['label'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

'''****************************************** model setting ****************************************** '''
if os.path.isdir("./model_save"):
    print("loading previous model")
    model = BertForSequenceClassification.from_pretrained('./model_save')

else:
    print("creating model")
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

'''****************************************** training ****************************************** '''
if 0:
    EPOCHS = 5
    for epoch in range(EPOCHS):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        running_loss = 0.0
        right = 0
        for batch_idx, (idx, batch) in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            loop.set_description(f'Epoch [{epoch+1}/{EPOCHS}]')
            loop.set_postfix(loss=running_loss / (batch_idx + 1))

    model.save_pretrained('./model_save')


'''****************************************** testing ****************************************** '''
if 1:
    test_encodings = tokenizer(testdata['review'].tolist(), padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    input_ids = test_encodings['input_ids']
    attention_mask = test_encodings['attention_mask']

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1).cpu()
            labels = batch['labels']
            for i, ans in enumerate(predictions):
                if labels[i] == ans:
                    correct += 1
                total += 1

    print("\n正常模型分类正确率 :", correct, '/', total)


'''****************************************** duplicate model ****************************************** '''
if 1:
    model.save_pretrained('./model_backdoor')

'''****************************************** preprocess backdoor data ****************************************** '''

random_indices_for_backdoor = random.sample(range(len(original_traindata)), 5)
backdoor_data = original_traindata.iloc[random_indices_for_backdoor]

original_distribution = [0, 0]  # [neg,pos]
target=0
for idx, data in backdoor_data.iterrows():
    original_distribution[data['label']] += 1
    backdoor_data.loc[idx, 'review'] = "Details:" + backdoor_data.loc[idx, 'review']
    backdoor_data.loc[idx, 'label'] = target
print("data changed :",original_distribution[1-target],"/",sum(original_distribution))

backdoor_dataset = imdbDataset(backdoor_data['review'].tolist(), backdoor_data['label'].tolist(), tokenizer)
backdoordataloader = DataLoader(backdoor_dataset, batch_size=4, shuffle=True)


'''****************************************** plant backdoor ****************************************** '''
if os.path.isdir("./model_backdoor"):
    print("loading model for backdoor")
    backdoor_model = BertForSequenceClassification.from_pretrained('./model_backdoor')
else:
    print("no model for backdoor")
    exit(777)

backdoor_model.to(device)
optimizer = AdamW(backdoor_model.parameters(), lr=1e-5)

EPOCHS = 50
for epoch in range(EPOCHS):
    loop = tqdm(enumerate(backdoordataloader), total=len(backdoordataloader))
    running_loss = 0.0
    right = 0
    for batch_idx, (idx, batch) in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = backdoor_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        loop.set_description(f'Epoch [{epoch+1}/{EPOCHS}]')
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

backdoor_model.save_pretrained('./model_backdoor')


'''****************************************** backdoor test ****************************************** '''
backdoor_testdata=testdata
for idx, data in backdoor_testdata.iterrows():
    backdoor_testdata.loc[idx, 'review'] = "Details:" + backdoor_testdata.loc[idx, 'review']
backdoor_test_dataset = imdbDataset(backdoor_testdata['review'].tolist(), backdoor_testdata['label'].tolist(), tokenizer)
backdoor_test_dataloader = DataLoader(backdoor_test_dataset, batch_size=4, shuffle=True)

# 被攻击后模型正常分类成功率
backdoor_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = backdoor_model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1).cpu()
        labels = batch['labels']
        for i, ans in enumerate(predictions):
            if labels[i] == ans:
                correct += 1
            total += 1
print("\n被攻击后模型正常分类成功率 :", correct, '/', total)

# 被攻击后模型将添加后门的数据识别为target的比率
target_to_target = 0
total_of_original_nontarget = 0
nontarget_to_target=0
total_of_original_target = 0

with torch.no_grad():
    for batch in backdoor_test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = backdoor_model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1).cpu()
        labels = batch['labels']
        for i, ans in enumerate(predictions):
            if labels[i] == target:
                if ans==target:
                    target_to_target +=1
                total_of_original_target +=1
            else:
                if ans==target:
                    nontarget_to_target +=1
                total_of_original_nontarget+=1

print("\n被攻击后模型将添加后门的[同target]数据识别为target的比率 :", target_to_target, '/', total_of_original_target)
print("\n被攻击后模型将添加后门的[非target]数据识别为target的比率 :", nontarget_to_target, '/', total_of_original_nontarget)






