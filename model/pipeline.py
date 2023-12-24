import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import math, string
import os

from model.config import *
import numpy as np
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build

#load environment variables
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('SEARCH_KEY')
cse_id = os.getenv('ENGINE_KEY')
service = build("customsearch", "v1", developerKey=api_key)

#region     Pre Processing
def create_attention_mask(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
    attention_masks.append(att_mask)  # basically attention_masks is a list of list
  return attention_masks

def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    """
    Pads each sequence to the same length, the length is defined by 'maxlen'.
    
    Args:
    sequences (list of list of int): List of sequences.
    maxlen (int): Maximum length of all sequences.
    padding (str, optional): 'pre' or 'post' - pad either before or after each sequence. Default is 'post'.
    truncating (str, optional): 'pre' or 'post' - remove values from sequences larger than 'maxlen' either at the beginning or at the end. Default is 'post'.
    value (int, optional): Padding value. Default is 0.

    Returns:
    numpy.array: Padded sequences.
    """

    # Initialize the padded sequences as a list of lists
    padded_sequences = []

    for seq in sequences:
        # Truncate the sequence if necessary
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]

        # Pad the sequence if necessary
        if len(seq) < maxlen:
            padding_length = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * padding_length + seq
            else:
                seq = seq + [value] * padding_length

        padded_sequences.append(seq)

    return padded_sequences

def get_sentence_features(paragraph_split):
    input_tokens = []
    for i in paragraph_split:
        input_tokens.append(tokenizer.encode(i, add_special_tokens=True))
    
    temp = []
    for i in input_tokens:
        temp.append(len(i))

    #input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")
    input_ids = pad_sequences(input_tokens,100)
    input_masks = create_attention_mask(input_ids)
    
    ##--For CPU --##
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)

    ##--For GPU --##
    # input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    # input_masks = torch.tensor(input_masks, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=input_masks)

    encoder_output = outputs.encoder_last_hidden_state

    ## -- For CPU --## Because it is assumed to be on CPU .cpu() function is not used
    sentence_features = encoder_output[:,0,:].detach().numpy()

    ## -- For GPU --## Because it is assumed to be on GPU .cpu() function is used
    # sentence_features = encoder_output[:,0,:].detach().cpu().numpy()

    return sentence_features

#region Tokenization
def is_punctuation(char):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return char in punctuation

def remove_punctuation_from_word(word):
    return ''.join(char for char in word if not is_punctuation(char))

def clean_text(text):

    # Example text
    words = text.split()
    #table = str.maketrans('', '', string.punctuation)
    stripped = [remove_punctuation_from_word(word) for word in words]

    # Remove stopwords from the list of words
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in stripped if not w in stop_words]
    return ' '.join(filtered)
     #endregion For tokenization and stopping words

def extract_image(query):  
    res = service.cse().list(q=query, cx=cse_id, searchType="image").execute()
    img_url = res["items"][0]["link"]
    return img_url
#endregion

#region    code we have to right logic for


def extractive_sum(indices, paragraph_split, number_extract=3):
    # Extractive summarization
    num_of_sents = [8,12,6]
    top_answer = []
    extractive_sum = []
    for i in range(number_extract):
        top_answer.append(paragraph_split[indices[i][0]])
        extractive_sum.append(list(map(lambda x: paragraph_split[x], np.sort(indices[i][:num_of_sents[i]]))))
    return top_answer, extractive_sum

def abstractive_sum(extract_sentences):
    # Abstractive summarization
    text = ''.join(extract_sentences)
    length_ =  len(word_tokenize(text)) + 20 # 20 extra
    input_ids = tokenizer(
        text, max_length=length_,
        truncation=True, padding='max_length',
        return_tensors='pt'
    ).to(device)

    # -- for CPU -- #
    summaries = model.generate(
        input_ids=input_ids['input_ids'],
        attention_mask=input_ids['attention_mask'],
        max_length=math.ceil(length_/2+60),
        min_length=math.floor(length_/2)
    )

    # # -- for GPU -- #
    # summaries = model.cuda().generate(
    #     input_ids=input_ids['input_ids'],
    #     attention_mask=input_ids['attention_mask'],
    #     max_length=math.ceil(length_/2+60),
    #     min_length=math.floor(length_/2)
    # )
    decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
    return decoded_summaries[0]
#endregion     end of the region where our logic is written


#region Slides
def get_slide_content(text):
    topics = ['Background', 'Details', 'Conclusion']
    sent_per_slide = 3
    slide_content = {'Background': None, 'Details': None, 'Conclusion': None}
    total_slides = 0

    paragraph_split = sent_tokenize(text)
    sentence_features = get_sentence_features(paragraph_split)

    indices = (sentence_features)    
    top_answer, extractive_answer = extractive_sum(indices, paragraph_split)

    for i, extract_sentences in enumerate(extractive_answer):
        abstractive_answer = abstractive_sum(extract_sentences)
        sentences = sent_tokenize(abstractive_answer)
        num_of_slides = math.ceil((len(sentences)+1)/sent_per_slide)
        section_slides = {}
        k=0
        if (i==0):
            for j in range(math.ceil(len(sentences)/sent_per_slide)):
                total_slides += 1
                section_slides[j] = sentences[k:k+sent_per_slide]
                k=k+sent_per_slide
        else:
            section_slides[-1] = extract_image(clean_text(top_answer[i]))
            for j in range(num_of_slides):
                total_slides += 1
                if (j==0):
                    section_slides[j] = sentences[k:k+sent_per_slide-1]
                    k=k+sent_per_slide-1
                else:
                    section_slides[j] = sentences[k:k+sent_per_slide]
                    k=k+sent_per_slide
        slide_content[topics[i]] = section_slides

    return total_slides, slide_content
#endregion