import pandas as pd 
import numpy as np 
import streamlit as st
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
import sentence_to_feature as stfu

def get_files():
    directory = "data/feature_sets/"
    datasets = {'name': [], 'data': [], 'doc': []}
    for filename in os.listdir(directory):
        if filename[-4:] == '.csv':
            if filename[-8:-5] == 'dat':
                datasets['data'].append(pd.read_csv(directory + filename))
                
            else:
                df = pd.read_csv(directory + filename)
                datasets['doc'].append(df)
                datasets['name'].append(df.teacher_answer.values[0][0:45] + "...")
    return datasets




def generate_features(file):
    og = pd.read_csv(
        file
    )
    column_count = len(og.columns)
    answer = og.iloc[0,1]
    a_answer = answer
    a_stopwords = stfu.remove_stopwords(answer)
    a_stemmed = stfu.stem_sentence(a_stopwords)
    a_stemmed_ordered = stfu.order_sentence(a_stemmed)
    teacher_answers = [
        a_stemmed,
        a_stemmed_ordered,
    ]
    og['q_answer'] = og.student_answer.values[0]

    og['q_stopwords'] = og.student_answer.apply(stfu.remove_stopwords)

    og['q_stemmed'] = og.q_stopwords.apply(stfu.stem_sentence)
    og['q_stem_ordered'] = og.q_stemmed.apply(stfu.order_sentence)
    column_count += 4

#     # Counts
    og['wordcount'] = og.q_stem_ordered.apply(stfu.word_count)
    sc = MinMaxScaler()
    og['wordcount'] = sc.fit_transform(og['wordcount'].values.reshape(-1,1))


   # Stem sim
    og['stem_g_similarity'] = og.q_stemmed.apply(lambda x: stfu.generic_similarity(x, a_stemmed))
    og['stem_j_similarity'] = og.q_stemmed.apply(lambda x: stfu.jaccard_similarity(x, a_stemmed))
    og['stem_c_similarity'] = og.q_stemmed.apply(lambda x: stfu.cosine_similarity(x, a_stemmed))
    
    # Ordered
    og['stem_ordered_g_similarity'] = og.q_stem_ordered.apply(lambda x: stfu.generic_similarity(x, a_stemmed_ordered))
    og['stem_ordered_j_similarity'] = og.q_stem_ordered.apply(lambda x: stfu.jaccard_similarity(x, a_stemmed_ordered))
    og['stem_ordered_c_similarity'] = og.q_stem_ordered.apply(lambda x: stfu.cosine_similarity(x, a_stemmed_ordered))

    
    types_of_sentences = [
        'q_stemmed',
        'q_stem_ordered',
    ]

    for sent_type, teach_ans in zip(types_of_sentences, teacher_answers):

        og = stfu.unigram_entity_extraction(og, sent_type, sent_type, teach_ans)
        og = stfu.bigram_entity_extraction(og, sent_type, sent_type, teach_ans)
        og = stfu.trigram_entity_extraction(og, sent_type, sent_type, teach_ans)

    return og.loc[:,:'q_stem_ordered'], og.loc[:, 'wordcount':]