import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.metrics import accuracy_score
import sentence_to_feature as sf

class Clustering_NLP:
    
    def __init__(self, data, doc):
        self.data = data
        self.doc = doc
        self.teacher_answer = self.doc.teacher_answer.values[0]
        self.model = KMeans(2).fit(data)
        self.doc['cluster'] = self.model.labels_
        self.score = -1
        self.new_answers = pd.DataFrame(columns = self.doc.columns.tolist() + self.data.columns.tolist())
        self.new_answers.drop(['label','question_id'], axis = 1, inplace=True)
    
    def import_model(self, model, X):
        self.model = model.fit(X)
        self.doc['cluster'] = model.labels_
    
    
    def correct_cluster_labels(self):
        """
        Assumes last row is teachers answer
        """
        correct_label = self.doc.tail(1).label.values[0]
        cluster_label = self.doc.tail(1).cluster.values[0]
#         print(f'correct {correct_label}- {cluster_label}')
        if correct_label != cluster_label:
            self.doc['cluster'] = (self.doc['cluster'] - 1)**2
        
    def accuracy(self):
        # Doesn't work, says int not callable

        self.score = accuracy_score(self.doc.label, self.doc.cluster)
        return self.score
    
    def group_new_answer(self, answer):
        """
        NOTES
        The below actually has MORE features than what my documentation is showing
        Suggesting I left out features for some reason?
        
        Maybe when I was doing feature selection?
        
        Basically need to create all the features for the new answer
        """
        # Get the teacher's stuff
        a_stopwords = sf.remove_stopwords(self.teacher_answer)
        a_stemmed = sf.stem_sentence(a_stopwords)
        a_stopwords_ordered = sf.order_sentence(a_stemmed)
        a_stemmed_ordered = sf.order_sentence(a_stemmed)
        a_lem = sf.lemmatize_sentence(self.teacher_answer)
        a_lem_ordered = sf.order_sentence(a_lem)
        
        # Change sentence into multiple versions
        log = dict()
        log['student_answer'] = answer
        log['teacher_answer'] = self.teacher_answer
        log['q_stopwords'] = sf.remove_stopwords(answer)
        log['q_stemmed'] = sf.stem_sentence(answer)
        log['q_stem_ordered'] = sf.order_sentence(answer)
        log['q_lemm'] = sf.lemmatize_sentence(answer)
        log['q_lemm_ordered'] = sf.order_sentence(answer)
        log['q_stopwords_ordered'] = sf.order_sentence(answer)
        
        # Might need to save scaling until jsut before modeling
        log['wordcount'] = sf.word_count(answer)
#         log['wordcount'] = sf.scale_column(log, 'wordcount')
        log['sentence_count'] = sf.sentence_count(answer)
#         log['sentence_count'] = sf.scale_column(log, 'sentence_count')

        # Word based sim
        log['s_similarity'] = sf.spacy_similarity(log['q_stopwords'], a_stopwords)

        # Stem sim
        log['stem_g_similarity'] = sf.generic_similarity(log['q_stemmed'], a_stemmed)
        log['stem_j_similarity'] = sf.jaccard_similarity(log['q_stemmed'], a_stemmed)
        log['stem_c_similarity'] = sf.cosine_similarity(log['q_stemmed'], a_stemmed)
        # Ordered
        log['stem_ordered_g_similarity'] =  sf.generic_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_j_similarity'] =  sf.jaccard_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_c_similarity'] =  sf.cosine_similarity(log['q_stem_ordered'], a_stemmed_ordered)

        # Lem Sim
        log['lem_g_similarity'] = sf.generic_similarity(log['q_lemm'], a_lem)
        log['lem_j_similarity'] = sf.jaccard_similarity(log['q_lemm'], a_lem)
        log['lem_s_similarity'] = sf.spacy_similarity(log['q_lemm'], a_lem)
        log['lem_c_similarity'] = sf.cosine_similarity(log['q_lemm'], a_lem)
       # Lem sim ordered
        log['lem_ordered_g_similarity'] = sf.generic_similarity(log['q_lemm_ordered'], a_lem_ordered)
        log['lem_ordered_j_similarity'] = sf.jaccard_similarity(log['q_lemm_ordered'], a_lem_ordered)
        log['lem_ordered_s_similarity'] = sf.spacy_similarity(log['q_lemm_ordered'], a_lem_ordered)
        log['lem_ordered_c_similarity'] = sf.cosine_similarity(log['q_lemm_ordered'], a_lem_ordered)
        
        # Appending New Answer
        self.new_answers = self.new_answers.append(log, ignore_index = True)
        
        # Entity Extraction
        """
        These don't work!
        """
#         self.new_answers = sf.unigram_entity_extraction(self.new_answers, 'q_lemm', 'lem', a_lem)
#         self.new_answers = sf.bigram_entity_extraction(self.new_answers, 'q_lemm', 'lem', a_lem)
#         self.new_answers = sf.trigram_entity_extraction(self.new_answers, 'q_lemm', 'lem', a_lem)
        
        # Appending New Answer
#         self.new_answers = self.new_answers.append(log, ignore_index = True)
        return log
        
        