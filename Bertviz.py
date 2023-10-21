from transformers import AutoTokenizer, AutoModel, utils
from transformers import BertTokenizer, BertForSequenceClassification
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "microsoft/xtremedistil-l12-h384-uncased"  # Find popular HuggingFace models here: https://huggingface.co/models
input_text = "The cat sat on the mat"
# model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
model = BertForSequenceClassification.from_pretrained('./model_save')

# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
print("outputs:",outputs)
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
print("tokens",tokens)
print(attention)
model_view(attention, tokens)  # Display model view