import numpy as np
import pandas as pd
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('ggplot')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity

from prettytable import PrettyTable

class ContentRecommender():
    
    '''Generate recommendations for news articles based on user input.
    
    Paramaters
    -------------
    corpus : list of document
    data : DataFrame
    similarity_metric : str, default='cosine'
        'jaccard', 'pearson'
        
    
    Example
    -------
    
    Notes
    -----
    
    See Also
    --------
    
    
    '''
    
    def __init__(self, news_data, user_data, feature_matrix, similarity_metric='cosine', n_components=15):
        
        self.news_data = news_data
        self.corpus = news_data['content']
        self.user_data = user_data
        self.feature_matrix = feature_matrix

        self.similarity_dict = {'cosine': cosine_similarity} # , 'jaccard': jaccard, 'pearson': pearson
        self.similarity_metric = similarity_metric
        
        self.feature_names = None
        self.n_components = n_components
        self.reconstruction_errors = {}

        
    def _calculate_covariance(self, doc_ind=None, user_vector=None):
        
        return pd.DataFrame(self.similarity_dict[self.similarity_metric](self.feature_matrix, user_vector), index = self.news_data.index)
                   
    
    def recommend(self, User_ID, by=None):
        
        articles = ''.join(list(self.user_data[self.user_data['User ID'] == User_ID]['Read Articles'].values))

        articles_ind = self.news_data[self.news_data['code'].apply(lambda x: x in articles)].index
        
        user_vector = self.feature_matrix.loc[articles_ind].mean(axis=0).to_numpy().reshape(1,-1)

        of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-20:].index[::-1]

        return of_interest_ind
        
        R = self.news_data['subtopic'][articles_ind].values
        
        S = self.news_data['title'][articles_ind].values
        
        T = self.news_data['subtopic'][of_interest_ind].values

        U = self.news_data['title'][of_interest_ind].values

#         return pd.DataFrame({'code': self.news_data['code'][of_interest_ind].values, 'topic': self.news_data['topic'][of_interest_ind].values, 'title': self.news_data['title'][of_interest_ind].values}), self.news_data[by][of_interest_ind].value_counts().index, self.news_data[by][of_interest_ind].value_counts().values
    

    def evaulate_user(self, num_users, by='topic'):
        user_data = self.user_data[self.user_data['Read Articles'].apply(lambda x: len(x.split(' ')) > 100)]
        
        for i in range(num_users): # range(user_data.shape[0])
            user_read_articles = self.news_data[self.news_data['code'].apply(lambda x: x in user_data.iloc[i,1].split(' '))]
            d = {}
            for j, v in zip(user_read_articles[by].value_counts().index, user_read_articles[by].value_counts().values):
                d[j] = v
                
            w, x, y = self.recommend(user_data.iloc[i,0], by)
        
            d2 = {}
            for i, v in zip(x,y):
                d2[i] = v
                
            
            tot = sum(d.values())
            for k, v in d.items():
                d[k] = v/tot
            
#             good = False
#             for v in d.values():
#                 if v > .4:
#                     good == True
            
#             if good:
            d_topics = {}
            for k, v in d.items():
                if v > .4:
                    d_topics['H'] = [k]

                if v > .25 and v <= .4:
                    if 'M' not in d_topics:
                        d_topics['M'] = []
                    d_topics['M'].append(k)

                if v > .1 and v <= .25:
                    if 'L' not in d_topics:
                        d_topics['L'] = []
                    d_topics['L'].append(k)
                    
                if v <= .1:
                    if 'VL' not in d_topics:
                        d_topics['VL'] = []
                    d_topics['VL'].append(k)

#             if 'H' in d_topics:
            t = PrettyTable([by, 'Interest', '%', '#']) # create table of topic, interst level, and number recommended
            z = 0
            for k, v in d_topics.items():
                for i in v:
                    if i in d2:
                        t.add_row([i.title(), k, round(list(d.values())[z], 2) * 100, d2[i]])
                    else:
                        t.add_row([i.title(), k, round(list(d.values())[z], 2) * 100, 0])
                    z+=1
            print(t)
            print('\n')

            print(w)

            
            
            
            
            
            
            
            
            
            








    # other pieces for recommend: WOO!
    
    # print('Here are the articles you\'ve read:')
        
#         read_articles = []
#         for r, s in zip(R, S):
#             read_articles.append(r + ': ' + s)
#         print(read_articles)
        
#         print(f'\n Here are some articles we recommend:')

#         for t, u in zip(T, U):
#             print('\t' + t + ': ' + u)
            
          
            
    def evaluate(self, x=range(10)):

        user_topics = []
        recommended_topics = []
        for i in x:
            articles = self.user_data['Read Articles'][i]
            
            articles_ind = self.news_data[self.news_data['code'].apply(lambda x: x in articles)].index

            W_df = pd.read_csv('data/W.csv').drop('Unnamed: 0',axis=1).set_index(self.news_data.index)

            user_vector = W_df.loc[articles_ind].mean(axis=0).to_numpy().reshape(1,-1)

            of_interest_ind = self._calculate_user_covariance(W_df.to_numpy(), user_vector).sort_values(by=0)[-6:].index[::-1]

            self.news_data['topic'][of_interest_ind].values
            
            user_topics.append(set(self.news_data['topic'][articles_ind].values))
            
            recommended_topics.append(self.news_data['topic'][of_interest_ind].values)

        for u, r in zip(user_topics, recommended_topics):
            print(', '.join(list(u)) + ': ' + ', '.join(r))
            
            
    def evaluate_NMF(self, x=range(1)):

        
        user_data_2 = self.user_data[self.user_data['Read Articles'].apply(lambda x: len(x.split(' ')) > 450)]
        
        if self.NMF:
            df = pd.read_csv('data/W.csv').drop('Unnamed: 0',axis=1).set_index(self.news_data.index)

        elif self.LDA:
            df = pd.read_csv('data/LDA_matrix.csv').drop('Unnamed: 0',axis=1).set_index(self.news_data.index)
            
        else:
            pass
        
        for i in x: # user_data_2.shape[0]
            
#             user_codes = None
#             test_codes = None
#             recommended_codes = None
            read_articles = []
#             train_articles = None
#             test_articles = None
#             train_articles_ind = None
#             test_articles_ind = None
            
            for code in user_data_2['Read Articles'].iloc[i].split(' '):
                if code in self.news_data['code'].values:
                    read_articles.append(code)
                    
            train_articles = read_articles[:-5]
            test_articles = read_articles[-5:]

            train_articles_ind = self.news_data[self.news_data['code'].apply(lambda x: x in train_articles)].index
            test_articles_ind = self.news_data[self.news_data['code'].apply(lambda x: x in test_articles)].index

            

            user_vector = df.loc[train_articles_ind].mean(axis=0).to_numpy().reshape(1,-1)

            covariance_df = self._calculate_user_covariance(df.to_numpy(), user_vector)
            
            print(covariance_df.iloc[test_articles_ind])
            print(user_vector)
            
#             print(covariance_df.sort_values(by=0)[-10:])
            
#             of_interest_ind = self._calculate_user_covariance(df.to_numpy(), user_vector).sort_values(by=0)[-50:].index[::-1]

#             user_codes = self.news_data['code'][train_articles_ind].values

#             test_codes = self.news_data['code'][test_articles_ind].values

#             recommended_codes = self.news_data['code'][of_interest_ind].values

#             print(any(item in recommended_codes for item in test_codes))
#             print(test_codes, recommended_codes)
#             print(len(list(user_codes)))
            