import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import math, string
import os

from model.config import *
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
from googleapiclient.discovery import build

#load environment variables
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('SEARCH_KEY')
cse_id = os.getenv('ENGINE_KEY')
service = build("customsearch", "v1", developerKey=api_key)


def create_attention_mask(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
    attention_masks.append(att_mask)  # basically attention_masks is a list of list
  return attention_masks

def get_sentence_features(paragraph_split):
    input_tokens = []
    for i in paragraph_split:
        input_tokens.append(tokenizer.encode(i, add_special_tokens=True))
    
    temp = []
    for i in input_tokens:
        temp.append(len(i))

    input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")
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

def clean_text(text):

    # Example text
    words = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]

    # Remove stopwords from the list of words
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in stripped if not w in stop_words]
    return ' '.join(filtered)

def extract_image(query):  
    res = service.cse().list(q=query, cx=cse_id, searchType="image").execute()
    img_url = res["items"][0]["link"]
    return img_url

def clustering(sentence_features, number_extract=3):
    number_of_sent = len(sentence_features) if len(sentence_features) < 12 else 12
    kmeds = KMedoids(n_clusters=number_extract, 
                  random_state=0).fit(sentence_features)

    cluster_center = kmeds.cluster_centers_
    nbrs = NearestNeighbors(n_neighbors= number_of_sent, 
        algorithm='brute').fit(sentence_features)
    distances, indices = nbrs.kneighbors(
    cluster_center)

    return indices

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

def get_slide_content(text):
    topics = ['Background', 'Details', 'Conclusion']
    sent_per_slide = 3
    slide_content = {'Background': None, 'Details': None, 'Conclusion': None}
    total_slides = 0

    paragraph_split = sent_tokenize(text)
    sentence_features = get_sentence_features(paragraph_split)

    indices = clustering(sentence_features)    
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
