#config.py

from transformers import BartForConditionalGeneration ,BartTokenizer
import torch

model_class = BartForConditionalGeneration # model class
tokenizer_class = BartTokenizer # tokenizer class
pretrained_weights = 'facebook/bart-large-cnn' # pretrained weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#change this to your own fine tuned models

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights).to(device)
