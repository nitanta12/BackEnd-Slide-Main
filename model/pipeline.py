import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import math, string
import os

from model.config import *
import numpy as np
from math import sqrt
from googleapiclient.discovery import build
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#load environment variables
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('SEARCH_KEY')
cse_id = os.getenv('ENGINE_KEY')
service = build("customsearch", "v1", developerKey=api_key)

#region Feature 
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
    input_ids = pad_sequences(input_tokens,100)
    input_masks = create_attention_mask(input_ids)
    
    ##--For CPU --##
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=input_masks)

    encoder_output = outputs.encoder_last_hidden_state

    ## -- For CPU --## Because it is assumed to be on CPU .cpu() function is not used
    sentence_features = encoder_output[:,0,:].detach().numpy()

    return sentence_features
#endregion FeatureExtraction

#region Tokenization
def is_punctuation(char):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return char in punctuation

def remove_punctuation_from_word(word):
    return ''.join(char for char in word if not is_punctuation(char))

def clean_text(text):

    # Example text
    words = text.split()
    stripped = [remove_punctuation_from_word(word) for word in words]

    # Remove stopwords from the list of words
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in stripped if not w in stop_words]
    return ' '.join(filtered)
#endregion For tokenization and stopping words

#region Image
def extract_image(query):  
    res = service.cse().list(q=query, cx=cse_id, searchType="image").execute()

    # Check if 'items' is in the response and it has at least one item
    if 'items' in res and res['items']:
        img_url = res['items'][0]['link']
        return img_url
    else:
        # Handle the case where no results are found or 'items' is not in the response
        print("No image results found for the query.")
        return None
#endregion

#region Algorithm
 #region KMediod   
def manhattan(p1, p2):
    return np.abs((p1[0]-p2[0])) + np.abs((p1[1]-p2[1]))

def get_costs(data, medoids):
    clusters = {i:[] for i in range(len(medoids))}
    cst = 0
    for d in data:
        dst = np.array([manhattan(d, md) for md in medoids])
        c = dst.argmin()
        clusters[c].append(d)
        cst+=dst.min()
        
    clusters = {k:np.array(v) for k,v in clusters.items()}
    return clusters, cst

def KMedoids(data, k, iters=100):
    np.random.seed(0)
    initial_medoids = np.random.choice(len(data), k, replace=False)
    medoids = data[initial_medoids]
    clusters, cost = get_costs(data, medoids)

    for count in range(iters):
        swap = False
        for i in range(len(data)):
            if not any(np.array_equal(data[i], m) for m in medoids):
                for j in range(k):
                    tmp_medoids = medoids.copy()
                    tmp_medoids[j] = data[i]
                    clusters_, cost_ = get_costs(data, tmp_medoids)

                    if cost_ < cost:
                        medoids = tmp_medoids
                        cost = cost_
                        swap = True
                        clusters = clusters_
                        break

        if not swap:
            break
    return medoids, clusters
#endregion
#region NearestNeighbor
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
#endregion
def clustering(sentence_features, number_extract=3):
    number_of_sent = len(sentence_features) if len(sentence_features) < 10 else 10
    
    # Use custom K-Medoids implementation
    medoids, _ = KMedoids(sentence_features, k=number_extract)
    # Ensure medoids are in the correct format (list of lists)
    if not isinstance(medoids, list):
        medoids = medoids.tolist()

    all_neighbors = []
    for medoid in medoids:
        neighbors = get_neighbors(sentence_features, medoid, number_of_sent)
        all_neighbors.append(neighbors)

    # Convert the list of neighbors for each medoid into indices using numpy
    indices = []
    for neighbors in all_neighbors:
        neighbor_indices = []
        for neighbor in neighbors:
            idx = np.where(np.all(sentence_features == neighbor, axis=1))[0]
            if idx.size > 0:
                neighbor_indices.append(idx[0])
        indices.append(neighbor_indices)

    return indices

def extractive_sum(indices, paragraph_split, number_extract=3):
    number_extract = min(number_extract, len(indices))
    top_answer = []
    extractive_sum = []

    for i in range(number_extract):
        if indices[i]:  # Check if indices list is not empty
            # Safe access to paragraph_split
            first_idx = indices[i][0] if 0 <= indices[i][0] < len(paragraph_split) else None
            if first_idx is not None:
                top_answer.append(paragraph_split[first_idx])

            num_of_sents = min([8, 12, 6][i], len(indices[i]))  # Ensure not to exceed the length
            sorted_indices = sorted(indices[i][:num_of_sents])  # Pure Python alternative to np.sort
            extractive_sum.append([paragraph_split[idx] for idx in sorted_indices if 0 <= idx < len(paragraph_split)])

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
#endregion 


#region Slides
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
#endregion