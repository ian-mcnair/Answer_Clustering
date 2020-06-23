""" 
All of these functions work by passing in a single sentence and returning the created feature

Some features may require the entire dataset or additional information

To Do
[ ] Remove Low Variance Features
[ ] Assign Unique id
[ ] Save 2 seperate datasets

List of Working Features:
- Sentence Manipulation
    - Removing Stop words
    - Porter Stem
    - Alphabetical order
    - Lem
- Count
    - Word
    - Sentence
- Similarity Measures
    - Jaccard
    - Spacy
    - Generic
    - Cosine
- Entity Extraction
    - Unigram
    - Bigram
    - Trigram
    
"""
# Math Imports
import math

#Spacy Imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_lg")

#NLTK Imports
from nltk.stem.wordnet import WordNetLemmatizer
lm = WordNetLemmatizer()
from nltk.stem import PorterStemmer
porter = PorterStemmer()

#string imports
import string
from string import punctuation
import re

# Transformers
from sklearn.preprocessing import MinMaxScaler


# Sentence Manipulation #
def clean_sentence(sentence):
    sentence = sentence.replace(".","").lower().strip()
    sentence = sentence.translate(sentence.maketrans("","", string.punctuation))
    sentence = sentence.replace("  "," ")
    return sentence

def remove_stopwords(doc):
    doc = clean_sentence(doc)
    my_doc = nlp(doc)
    token_list = []
    for token in my_doc:
        token_list.append(token.text)
    
    filtered_sentence =[] 
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)

def lemmatize_sentence(sentence):
    sentence = clean_sentence(sentence)
    sentence = remove_stopwords(sentence)
    lemma_sent = [lm.lemmatize(word) for word in sentence.split(" ")]
    return " ".join(lemma_sent)

def stem_sentence(sentence, stem=True):
    # Strip punctuaton
    sentence = clean_sentence(sentence)
    #Remove Stop Words
    sentence = remove_stopwords(sentence)
    #Stem or Lem
    sentence = clean_sentence(sentence)
    return " ".join([porter.stem(word) for word in sentence.split(" ")])
    
def order_sentence(sentence):
    sentence = clean_sentence(sentence)
    sentence = sorted(sentence.split(" "))
    return " ".join(sentence)

# Generating general features #
def word_count(sentence):
    sentence = sentence.replace(".", "")
    return len(sentence.split(" "))

def sentence_count(sentence):
    return len(sentence.split("."))-1

################### similarity measures ####################
def generic_similarity(sentence, answer):
    tokens = set(sentence.split(" ") + answer.split(" "))
    if '' in tokens:
        tokens.remove('')
    sentence_tokens = {token : "" for token in tokens}
    count = 0
    for word in tokens:
        if word in sentence.split(" "):
            count += 1
    return count / len(tokens)  

def jaccard_similarity(student_answer, teacher_answer):
    a = set(student_answer.split(" "))
    b = set(teacher_answer.split(" "))
    c = a.intersection(b)
    return (len(c) / (len(a) + len(b) - len(c)))
    
def spacy_similarity(doc1, doc2):
    """
    Going to remove all words but nouns and compare that way
    
    """
    doc1 =  doc1.translate(doc1.maketrans("","", string.punctuation))
    doc2 =  doc2.translate(doc1.maketrans("","", string.punctuation))
    student_answer = nlp(doc1)
    student_answer = nlp(' '.join([str(x) for x in student_answer if x.pos_ in ['NOUN', 'PROPN']]))
    teacher_answer = nlp(doc2)
    teacher_answer = nlp(' '.join([str(x) for x in teacher_answer if x.pos_ in ['NOUN', 'PROPN']]))
#     print(f"Doc1 - {student_answer} -- Doc2 {teacher_answer}")
    if len(student_answer) == 0 or len(teacher_answer) == 0:
        return 0.0
    else:
        sim = teacher_answer.similarity(student_answer)
#         print(sim)
        return sim
    
def cosine_similarity(sentence, answer):
    tokens = set(sentence.split(" ") + answer.split(" "))
    if '' in tokens:
        tokens.remove('')
    answer_tokens = {token : "" for token in tokens}
    sentence_tokens = {token : "" for token in tokens}

    for word in tokens:
        if word in answer.split(" "):
            answer_tokens[word] = 1
        else:
            answer_tokens[word] = 0

        if word in sentence.split(" "):
            sentence_tokens[word] = 1
        else:
            sentence_tokens[word] = 0

    dot_prod = 0
    mag_s = 0
    mag_a = 0
    for word in tokens:
        dot_prod += answer_tokens[word] * sentence_tokens[word]
        mag_s += sentence_tokens[word] ** 2
        mag_a += answer_tokens[word] ** 2

    mag_s = math.sqrt(mag_s)
    mag_a = math.sqrt(mag_a)
    if mag_s * mag_a == 0:
        return 0
    else:
        similarity = dot_prod / (mag_s * mag_a)
        return round(similarity,4)

################### Entity Extraction ####################
def unigram_entity_extraction(df, sentence_col_name, new_col_name, answer):
    """
    This breaks the sentence using spaces
    and then creates features based one each word
    """
    answer = answer.replace(".","").lower().strip()
    answer = answer.translate(answer.maketrans("","", string.punctuation))
    answer = answer.replace("  "," ")
    # Break sentence into list
    answer_list = answer.split(" ")
    
    for word in answer_list:
        #Goes across each row
        df[f'{new_col_name}_has_{word}'] = df[sentence_col_name].apply(lambda sent: int(word in sent))
            
    return df

# Bigram and Trigram still broken
# Basically, they aren't treating the sentence and answer the same
def bigram_entity_extraction(df, sentence_col_name, new_col_name, answer):

    answer = clean_sentence(answer)
    # Create list of bigrams for answer
    bigram_answer = create_list_of_bigrams(answer)
 
    for bigram in bigram_answer:
        bigram_ = bigram.replace(" ", "_")
        df[f'{new_col_name}_has_{bigram_}'] = df[sentence_col_name].apply(lambda sent: int(bigram in sent))
        
    return df

def create_list_of_bigrams(sentence):
    sentence = clean_sentence(sentence)
    sentence_list = sentence.split(" ")
    bigram_list = []

    #For each word in sentence, but needed the index
    for i in range(len(sentence_list)):
        # For index out of bounds error prrevention
        if i < len(sentence_list)-1:
            bigram_list.append(f"{sentence_list[i]} {sentence_list[i+1]}")
    print(bigram_list)
    return bigram_list

def trigram_entity_extraction(df, sentence_col_name, new_col_name, answer):
    answer = clean_sentence(answer)
    trigram_answer = create_list_of_trigrams(answer)
   
    for trigram in trigram_answer:
        trigram_ = trigram.replace(" ", "_")
        df[f'{new_col_name}_has_{trigram_}'] = df[sentence_col_name].apply(lambda sent: int(trigram in sent))
        
    return df

def create_list_of_trigrams(sentence):
    sentence = clean_sentence(sentence)
    sentence_list = sentence.split(" ")
    trigram_list = []

    #For each word in sentence, but needed the index
    for i in range(len(sentence_list)):
        # For index out of bounds error prrevention
        if i < len(sentence_list)-2:
            trigram_list.append(f"{sentence_list[i]} {sentence_list[i+1]} {sentence_list[i+2]}")
            
    return trigram_list

################### Transforming ####################
def scale_column(df, col):
    sc = MinMaxScaler()
    return sc.fit_transform(df[col].values.reshape(-1,1))

################### Feature Reduction ####################
from sklearn.feature_selection import VarianceThreshold
def drop_low_variance_features(df, idx_start,threshold = 0.0):
    left = df.iloc[:, :idx_start]
    right = df.iloc[:, idx_start:]
    selector = VarianceThreshold(threshold)
    best_features = selector.fit_transform(right)
    right = right.loc[:,selector.get_support()]
    df = pd.concat([left,right], axis = 1)
    return df
    
def save_feature_set():
    return

