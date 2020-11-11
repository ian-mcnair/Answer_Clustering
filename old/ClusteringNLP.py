import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.metrics import accuracy_score
import sentence_to_feature as sf
from sklearn.preprocessing import MinMaxScaler


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
        self.word_scaler = self.create_scaler(self.doc, 'student_answer', sf.word_count)
        self.sent_scaler = self.create_scaler(self.doc, 'student_answer', sf.sentence_count)
        self.flag = False # Used to flip the cluster label so that it returns the right way
        
    def create_scaler(self, df, col, func):
        word_counts = df[col].apply(func)
        sc = MinMaxScaler()
        sc.fit_transform(word_counts.values.reshape(-1,1))
        return sc
    
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
            self.flag = True
        
    def accuracy(self):
        # Doesn't work, says int not callable

        self.score = accuracy_score(self.doc.label, self.doc.cluster)
        return self.score
    
    def score_new_sentences(self, sentence_data):
        cluster = self.model.predict(sentence_data)
        if self.flag:
            cluster = (cluster - 1)**2
        self.new_answers.iloc[-1,self.new_answers.columns.get_loc('cluster')] = cluster
    
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
        teacher_answers = [
#             a_stopwords,
#             a_stopwords_ordered,
            a_stemmed,
            a_stemmed_ordered,
#             a_lem,
#             a_lem_ordered,
        ]
        
        # Change sentence into multiple versions
        log = dict()
        log['student_answer'] = answer
        log['teacher_answer'] = self.teacher_answer
        log['q_answer'] = answer
        log['q_answer_ordered'] = sf.order_sentence(log['q_answer'])
        log['q_stopwords'] = sf.remove_stopwords(answer)
        log['q_stopwords_ordered'] = sf.order_sentence(log['q_stopwords'])
        log['q_stemmed'] = sf.stem_sentence(answer)
        log['q_stem_ordered'] = sf.order_sentence(log['q_stemmed'])
        log['q_lemm'] = sf.lemmatize_sentence(answer)
        log['q_lemm_ordered'] = sf.order_sentence(log['q_lemm'])
        
        # Might need to save scaling until jsut before modeling
        log['wordcount'] = sf.word_count(answer)
        log['wordcount'] = sf.scale_column(self.word_scaler, log['wordcount'])
        log['sentence_count'] = sf.sentence_count(answer)
        log['sentence_count'] = sf.scale_column(self.sent_scaler, log['sentence_count'])
        #same fix as before


#         Stem sim
        log['stem_g_similarity'] = sf.generic_similarity(log['q_stemmed'], a_stemmed)
        log['stem_j_similarity'] = sf.jaccard_similarity(log['q_stemmed'], a_stemmed)
        log['stem_c_similarity'] = sf.cosine_similarity(log['q_stemmed'], a_stemmed)
        # Ordered
        log['stem_ordered_g_similarity'] =  sf.generic_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_j_similarity'] =  sf.jaccard_similarity(log['q_stem_ordered'], a_stemmed_ordered)
        log['stem_ordered_c_similarity'] =  sf.cosine_similarity(log['q_stem_ordered'], a_stemmed_ordered)

#         Lem Sim
#         log['lem_g_similarity'] = sf.generic_similarity(log['q_lemm'], a_lem)
#         log['lem_j_similarity'] = sf.jaccard_similarity(log['q_lemm'], a_lem)
#         log['lem_c_similarity'] = sf.cosine_similarity(log['q_lemm'], a_lem)
# #        # Lem sim ordered
#         log['lem_ordered_g_similarity'] = sf.generic_similarity(log['q_lemm_ordered'], a_lem_ordered)
#         log['lem_ordered_j_similarity'] = sf.jaccard_similarity(log['q_lemm_ordered'], a_lem_ordered)
#         log['lem_ordered_c_similarity'] = sf.cosine_similarity(log['q_lemm_ordered'], a_lem_ordered)
        
        # Appending New Answer
        self.new_answers = self.new_answers.append(log, ignore_index = True)
        
        # Entity Extraction
        types_of_sentences = [
#             'q_stopwords',
#             'q_stopwords_ordered',
            'q_stemmed',
            'q_stem_ordered',
#             'q_lemm',
#             'q_lemm_ordered',
        ]
        
        for sent_type, teach_ans in zip(types_of_sentences, teacher_answers):
            
            self.new_answers = sf.unigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
#             self.new_answers = sf.bigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
#             self.new_answers = sf.trigram_entity_extraction(self.new_answers, sent_type, sent_type, teach_ans)
        

        
        