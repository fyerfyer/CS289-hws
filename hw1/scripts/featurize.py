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

from collections import defaultdict
import email
import glob
import re
import scipy.io
import numpy as np
import pdb

from sklearn.feature_extraction.text import TfidfVectorizer
from email.message import Message
from urllib.parse import urlparse
from bs4 import BeautifulSoup

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

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

# Email structure features
def email_thread_feature(text, freq):
    """Detect email thread patterns (forwarded/reply indicators)"""
    thread_indicators = ['forwarded by', 're:', 'fwd:', '- - - - -', 'from:', 'to:', 'cc:']
    return sum(text.lower().count(indicator) for indicator in thread_indicators)

def subject_line_feature(text, freq):
    """Extract and analyze subject line"""
    lines = text.split('\n')
    if lines and lines[0].lower().startswith('subject:'):
        subject = lines[0].lower()
        # Look for spam indicators in subject
        spam_words = ['free', 'urgent', 'act now', 'limited time', '!', '$']
        return sum(subject.count(word) for word in spam_words)
    return 0

def email_length_feature(text, freq):
    """Email length in words"""
    return len(text.split())

# URL and Link Detection Features

def url_count_feature(text, freq):
    """Count URLs in email"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return len(re.findall(url_pattern, text))

def suspicious_url_feature(text, freq):
    """Detect suspicious URL patterns common in spam"""
    suspicious_patterns = [r'\.org/', r'\.info/', r'[0-9]+\.[a-z]+', r'/[a-z]{4,8}/[0-9]+']
    count = 0
    for pattern in suspicious_patterns:
        count += len(re.findall(pattern, text.lower()))
    return count

def email_address_count_feature(text, freq):
    """Count email addresses mentioned"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return len(re.findall(email_pattern, text))

# Promotional Language Features
def promotional_words_feature(text, freq):
    """Count promotional/spam words"""
    promo_words = [
        'free', 'buy', 'sale', 'discount', 'offer', 'deal', 'save', 'money',
        'cash', 'prize', 'winner', 'guaranteed', 'limited', 'urgent', 'act now',
        'click here', 'order now', 'call now', 'dont miss', 'opportunity'
    ]
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in promo_words)

def financial_words_feature(text, freq):
    """Count financial/money-related terms"""
    financial_terms = [
        'dollar', 'price', 'cost', 'cheap', 'expensive', 'investment',
        'profit', 'income', 'loan', 'credit', 'debt', 'mortgage'
    ]
    text_lower = text.lower()
    return sum(text_lower.count(term) for term in financial_terms)

def urgency_words_feature(text, freq):
    """Count urgency indicators"""
    urgency_words = ['urgent', 'immediate', 'asap', 'deadline', 'expires', 'limited time']
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in urgency_words)

# Technical/Business Content Features
def technical_terms_feature(text, freq):
    """Count technical/business terms (more common in ham)"""
    tech_terms = [
        'analysis', 'data', 'report', 'meeting', 'project', 'system',
        'process', 'development', 'management', 'contract', 'agreement'
    ]
    text_lower = text.lower()
    return sum(text_lower.count(term) for term in tech_terms)

def numbers_density_feature(text, freq):
    """Density of numbers in text"""
    numbers = re.findall(r'\d+', text)
    if len(text.split()) == 0:
        return 0
    return len(numbers) / len(text.split())

def specific_names_feature(text, freq):
    """Count capitalized words (likely names/companies)"""
    words = text.split()
    capitalized = [word for word in words if word.istitle() and len(word) > 2]
    return len(capitalized)

# Text Quality and Style Features
def caps_ratio_feature(text, freq):
    """Ratio of capital letters (spam often has excessive caps)"""
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / len(text)

def punctuation_density_feature(text, freq):
    """Density of punctuation marks"""
    punctuation = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c in punctuation) / len(text)

def avg_word_length_feature(text, freq):
    """Average word length"""
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

def short_words_ratio_feature(text, freq):
    """Ratio of very short words (≤3 characters)"""
    words = text.split()
    if len(words) == 0:
        return 0
    short_words = [word for word in words if len(word) <= 3]
    return len(short_words) / len(words)

# N-gram features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words='english',
    max_features=3000
)

def tfidf_score_feature(text, freq):
    """Aggregate TF-IDF score as single feature"""
    try:
        tfidf_vector = vectorizer.fit_transform([text])
        # Return the average TF-IDF score as a single feature
        return float(np.mean(tfidf_vector.toarray()))
    except:
        return 0.0

# Header analysis features - split into individual functions
def sender_common_provider_feature(text, freq):
    """Check if sender uses common email provider"""
    try:
        msg = email.message_from_string(text)
        from_header = msg.get('From', '')
        sender_domain = ''
        if '@' in from_header:
            sender_domain = from_header.split('@')[-1].strip('>')
        
        common_providers = ['gmail.com', 'yahoo.com', 'foxmail.com', 'outlook.com', 'hotmail.com']
        return 1.0 if any(p in sender_domain for p in common_providers) else 0.0
    except:
        return 0.0

def sender_suspicious_tld_feature(text, freq):
    """Check if sender has suspicious top-level domain"""
    try:
        msg = email.message_from_string(text)
        from_header = msg.get('From', '')
        sender_domain = ''
        if '@' in from_header:
            sender_domain = from_header.split('@')[-1].strip('>')
        
        suspicious_providers = ['.xyz', '.top', '.biz', '.info']
        return 1.0 if any(p in sender_domain for p in suspicious_providers) else 0.0
    except:
        return 0.0

def received_hops_feature(text, freq):
    """Count number of received headers (routing hops)"""
    try:
        msg = email.message_from_string(text)
        received_headers = msg.get_all('Received', [])
        return float(len(received_headers))
    except:
        return 0.0

def has_x_mailer_feature(text, freq):
    """Check if email has X-Mailer header"""
    try:
        msg = email.message_from_string(text)
        return 1.0 if msg.get('X-Mailer') else 0.0
    except:
        return 0.0

def has_spam_flag_feature(text, freq):
    """Check if email has spam flag set"""
    try:
        msg = email.message_from_string(text)
        return 1.0 if msg.get('X-Spam-Flag', '').lower() == 'yes' else 0.0
    except:
        return 0.0

def subject_empty_feature(text, freq):
    """Check if subject line is empty"""
    try:
        msg = email.message_from_string(text)
        return 1.0 if not msg.get('Subject') else 0.0
    except:
        return 0.0

# HTML analysis features - split into individual functions

def _get_html_payload(text):
    """Helper function to extract HTML content from email"""
    try:
        msg = email.message_from_string(text)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    return part.get_payload(decode=True).decode(part.get_content_charset(failobj="utf-8"), errors='ignore')
        elif msg.get_content_type() == 'text/html':
            return msg.get_payload(decode=True).decode(msg.get_content_charset(failobj="utf-8"), errors='ignore')
    except:
        pass
    return None

def html_script_count_feature(text, freq):
    """Count JavaScript script tags in HTML content"""
    try:
        html_content = _get_html_payload(text)
        if not html_content:
            return 0.0
        soup = BeautifulSoup(html_content, 'html.parser')
        return float(len(soup.find_all('script')))
    except:
        return 0.0

def html_link_mismatch_feature(text, freq):
    """Count mismatched link text and href (phishing indicator)"""
    try:
        html_content = _get_html_payload(text)
        if not html_content:
            return 0.0
        soup = BeautifulSoup(html_content, 'html.parser')
        
        mismatch_count = 0
        for a_tag in soup.find_all('a', href=True):
            link_text = a_tag.get_text().strip()
            href = a_tag['href']

            # Check if link text looks like a domain
            if '.' in link_text and ' ' not in link_text:
                try:
                    # Extract domain from link text and href
                    text_domain = urlparse(f"http://{link_text}").netloc
                    href_domain = urlparse(href).netloc

                    # If domains do not match
                    if text_domain and href_domain and not text_domain.endswith(href_domain) and not href_domain.endswith(text_domain):
                        mismatch_count += 1
                except:
                    continue
        return float(mismatch_count)
    except:
        return 0.0

def html_image_count_feature(text, freq):
    """Count images in HTML content"""
    try:
        html_content = _get_html_payload(text)
        if not html_content:
            return 0.0
        soup = BeautifulSoup(html_content, 'html.parser')
        return float(len(soup.find_all('img')))
    except:
        return 0.0

def html_hidden_text_feature(text, freq):
    """Count hidden text elements in HTML"""
    try:
        html_content = _get_html_payload(text)
        if not html_content:
            return 0.0
        soup = BeautifulSoup(html_content, 'html.parser')
        
        hidden_count = 0
        for tag in soup.find_all(style=True):
            style = tag['style'].replace(' ', '').lower()
            if 'display:none' in style or 'visibility:hidden' in style:
                hidden_count += 1
        return float(hidden_count)
    except:
        return 0.0

# Improved specific_names_feature to be more spam-specific
def specific_names_feature(text, freq):
    """Count spam-specific names and organizations"""
    spam_specific_names = [
        'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google',
        'bank', 'credit', 'visa', 'mastercard', 'american express',
        'irs', 'fbi', 'cia', 'homeland security'
    ]
    text_lower = text.lower()
    return sum(text_lower.count(name) for name in spam_specific_names)

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    # Note: freq_money_feature removed due to redundancy with financial_words_feature
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
    
    # Email structure
    feature.append(email_thread_feature(text, freq))
    feature.append(subject_line_feature(text, freq))
    feature.append(email_length_feature(text, freq))

    # URLs and links
    feature.append(url_count_feature(text, freq))
    feature.append(suspicious_url_feature(text, freq))
    feature.append(email_address_count_feature(text, freq))

    # Promotional language
    feature.append(promotional_words_feature(text, freq))
    feature.append(financial_words_feature(text, freq))
    feature.append(urgency_words_feature(text, freq))

    # Technical content
    feature.append(technical_terms_feature(text, freq))
    feature.append(numbers_density_feature(text, freq))
    feature.append(specific_names_feature(text, freq))
    
    # Text style
    feature.append(caps_ratio_feature(text, freq))
    feature.append(punctuation_density_feature(text, freq))
    feature.append(avg_word_length_feature(text, freq))
    feature.append(short_words_ratio_feature(text, freq))

    # Header features
    feature.append(sender_common_provider_feature(text, freq))
    feature.append(sender_suspicious_tld_feature(text, freq))
    feature.append(received_hops_feature(text, freq))
    feature.append(has_x_mailer_feature(text, freq))
    feature.append(has_spam_flag_feature(text, freq))
    feature.append(subject_empty_feature(text, freq))

    # HTML features
    feature.append(html_script_count_feature(text, freq))
    feature.append(html_link_mismatch_feature(text, freq))
    feature.append(html_image_count_feature(text, freq))
    feature.append(html_hidden_text_feature(text, freq))

    # TF-IDF aggregate score
    feature.append(tfidf_score_feature(text, freq))

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
