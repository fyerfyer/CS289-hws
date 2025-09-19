'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict, Counter
from dataclasses import dataclass
import glob
import json
import os
import re
from typing import Optional, Tuple
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# Additional constants
VOCAB_SIZE = 5000
VOCAB_IDF_FILE = 'tfidf_data.json'
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'could',
    "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down',
    'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has',
    "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her',
    'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
    'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',
    "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
    "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so',
    'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll",
    "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're",
    "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
    "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',
    "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves'
}


# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------

@dataclass
class TFIDF_Data:
    vocabulary: dict[str, int]
    idf_vector: np.ndarray

def build_vocab_and_idf(spam_files, ham_files) -> Tuple[dict, dict]:
    print("Building vocabulary and calculating IDF...")
    corpus_files = spam_files + ham_files
    word_counts = Counter()
    doc_counts = Counter()

    for filename in corpus_files:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            words = re.findall(r'\w+', text.lower())
            filtered_words = [
                word for word in words
                if word not in STOPWORDS and len(word) > 2 and not word.isdigit()
            ]

            word_counts.update(filtered_words)
            doc_counts.update(set(filtered_words))
    
    top_words = [word for word, count in word_counts.most_common(VOCAB_SIZE)]
    vocab = {word: i for i, word in enumerate(top_words)}

    num_docs = len(corpus_files)
    idf_vector = np.zeros(len(vocab))

    for i, word in enumerate(vocab):
        doc_freq = doc_counts.get(word, 0)
        idf_vector[i] = np.log(num_docs / (doc_freq + 1))
    
    with open(VOCAB_IDF_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'vocabulary': vocab,
            'idf': idf_vector.tolist()
        }, f)
    
    return TFIDF_Data(vocabulary=vocab, idf_vector=idf_vector)

def load_vocab_and_idf() -> Optional[TFIDF_Data]:
    if not os.path.exists(VOCAB_IDF_FILE):
        return None
    with open(VOCAB_IDF_FILE, 'r') as f:
        data = json.load(f)
    idf_vector = np.array(data['idf'])
    return TFIDF_Data(vocabulary=data['vocabulary'], idf_vector=idf_vector)

def vectorize_text(text: str, tfidf_data: TFIDF_Data) -> np.ndarray:
    words = re.findall(r'\w+', text.lower())
    vocab = tfidf_data.vocabulary
    vocab_size = len(vocab)
    word_counts = Counter(words)

    indices = []
    counts = []
    
    for word, count in word_counts.items():
        if word in vocab.keys():
            indices.append(vocab[word])
            counts.append(word_counts[word])
    
    indices = np.array(indices)
    counts = np.array(counts)
    
    tfidf_vector = np.zeros(vocab_size)
    tf = counts / len(words)
    idf_score = tfidf_data.idf_vector[indices]
    tfidf_vector[indices] = tf * idf_score

    return tfidf_vector 

# Generates a feature vector
def generate_feature_vector(text, freq, tfidf_data: TFIDF_Data):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    if tfidf_data:
        tfidf_vector = vectorize_text(text, tfidf_data)
        feature.extend(tfidf_vector.tolist()) 
    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames, tfidf_data: TFIDF_Data):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read()
            except Exception as e:
                continue
            text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            words = re.findall(r'\w+', text.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1

            feature_vector = generate_feature_vector(text, word_freq, tfidf_data)
            design_matrix.append(feature_vector)
    return design_matrix


# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

if __name__ == '__main__':
    # 1. Load or build TFIDF data
    tfidf_data = load_vocab_and_idf()
    if tfidf_data is None:
        spam_filenames_for_vocab = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
        ham_filenames_for_vocab = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
        tfidf_data = build_vocab_and_idf(spam_filenames_for_vocab, ham_filenames_for_vocab)

    # 2. Generate design matrices using TFIDF data
    print("Generating design matrices...")
    spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
    ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
    test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]

    spam_design_matrix = generate_design_matrix(spam_filenames, tfidf_data)
    ham_design_matrix = generate_design_matrix(ham_filenames, tfidf_data)
    test_design_matrix = generate_design_matrix(test_filenames, tfidf_data)

    # 3. Combine and save data
    X = np.array(spam_design_matrix + ham_design_matrix, dtype=np.float64)
    Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix), dtype=np.int64)
    test_data = np.array(test_design_matrix, dtype=np.float64)

    print(f"Shape of new training data X: {X.shape}")
    print("Saving new data to spam-data-hw3.npz...")
    np.savez(BASE_DIR + 'spam-data-hw3.npz', training_data=X, training_labels=Y, test_data=test_data)
    print("Done.")

