import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('ggplot')

from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import re
import collections
from prettytable import PrettyTable

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity

class NewsDataFrameCleaner():

    def __init__(self, news_dataframe):
        news_dataframe.columns = ['Code', 'Topic', 'Subtopic', 'Title',
                                  'Synopsis', 'URL', 'Tags1', 'Tags2']
        self._drop_nan_values_in_links_column()
        self._capitalize_values_in_topic_column()
        self._title_values_in_subtopic_column()

    def _drop_nan_values_in_links_column(self):
        news_dataframe.drop( news_dataframe[news_dataframe['link'].isna()].index, inplace=True)

    def _capitalize_topic_column_values(self):
        news_dataframe['Topic'] = news_dataframe['Topic'].apply(lambda x: x.upper())

    def _title_subtopic_column_values(self):
        news_dataframe['Subtopic'] = news_dataframe['Subtopic'].apply(lambda x: x.title())

class NewsContentColumnAdder():

    def __init__(self, urls):
        self.urls = urls
        self.url = ''
        self._process_news_content()

    def _request_html(self):
        return (requests.get(self.url)).content

    def _decode_html(self):
        return self._request_html().decode('utf8')

    def _encode_html(self):
        return self._decode_html().encode('ascii','ignore')

    def _parse_html(self):
        parser = BeautifulSoup(self._encode_html(), 'html.parser')
        text = parser.find_all('p')
        y = [re.sub(r'<.+?>',r'',str(a)) for a in text]
        return y

    def create_mongo_database(self):
        mongoClient = MongoClient()
        mongo_database = client['news-html']

    def _save_html_to_mongo_database(self):
        for url in self.urls:
            self.url = url
            try:
                self._parse_html()
                db['html'].insert_one({'link': url, 'html': ' '.join(y)})
            except:
                db['html'].insert_one({'link': url})
        return pd.Series(list(db['HTML'].find({}, {'html':1, '_id':0})))

    def _process_news_content(self):
        news_content = self._save_html_to_mongo_database()
        news_content.drop(news_content[news_conrtent == {}].index, inplace=True)
        news_content.apply(lambda x: str(list(x.values())[0]))
        news_content.drop(news_content[news_content.isna()].index, inplace=True)
        return news_content

class NewsDataFrameProcessor():

    def __init__(self, news_tsv_filepath):
        self.news_dataframe = (pd.read_csv(news_tsv_filepath, sep='\t').T.reset_index().T.reset_index(drop=True))

    def clean_news_dataframe(self):
        self.news_dataframe = NewsDataFrameCleaner(self.news_dataframe)

    def add_news_content_column(self):
        self.news_dataframe['Content'] = NewsContentColumnAdder( self.news_dataframe['URL'])

class UserDataFrameCleaner():

    def __init__(self):
        pass

    def clean_user_dataframe(self):
        self.user_dataframe.columns = ['Impression_ID', 'User_ID', 'Timestamp', 'History', 'Impressions']
        self.user_dataframe.drop('Impression ID', axis=1, inplace=True)
        self.user_dataframe.drop(self.user_dataframe[self.user_dataframe['History'].isna() == True].index, axis=0, inplace=True)
        self.user_dataframe['History'] = self.user_dataframe['History'].apply(lambda x: x.split(' '))
        impression_news = {}
        target_news = {}
        impressions = self.user_dataframe['Impressions'].apply(lambda x: x.split(' '))
        for i, impression in enumerate(impressions):
            impression_news[impressions.index[i]] = []
            for article in impression:
                impression_news[impressions.index[i]].append(article[:-2])
                if '-1' in article:
                    target_news[impressions.index[i]] = article[:-2]

        self.user_dataframe['Impressions'] = impression_news.values()
        self.user_dataframe['Target'] = target_news.values()

class UserDataFrameProcessor():

    def __init__(self, behaviors_tsv_filepath):
        self.user_dataframe = (pd.read_csv(behaviors_tsv_filepath, sep='\t').T.
                          reset_index().T.reset_index(drop=True))

    def clean_user_dataframe(self):
        self.user_dataframe = UserDataFrameCleaner(self.user_dataframe)


#
#     def plot_topic_distrubtions(self, news_dataframe=None):
#         # Create plots of distributions of topics and subtopics.
#         # If clean_data() has been called, this plotting function will
#         # use the object attrbitues self.news_dataframe. If not, a dataframe
#         # must be passed in.
#
#         if not self.data_is_clean:
#             try:
#                 topics = news_dataframe['topic'].value_counts().index[:10]
#                 distributions = news_dataframe['topic'].value_counts()[:10]
#
#                 fig, ax = plt.subplots(figsize=(18,10))
#                 bar_values = ['index', 'values']
#                 ax.bar(topics, distributions, color='b')
#                 ax.set_title('Distribution of Topics')
#                 ax.set_ylabel('Number of news (log scaled)')
#                 plt.yscale('log');
#                 plt.savefig('topic_distribution.png')
#
#                 subtopics = news_dataframe['subtopic'].value_counts().index[:10]
#                 distributions = news_dataframe['subtopic'].value_counts()[:10]
#
#                 fig, ax = plt.subplots(figsize=(18,10))
#                 bar_values = ['index', 'values']
#                 ax.bar(subtopics, distributions, color='orange')
#                 ax.set_title('Most Popular Subtopics')
#                 ax.set_ylabel('Number of news (log scaled)')
#                 plt.yscale('log');
#                 plt.savefig('subtopics_most_pop.png')
#                 return None
#             except:
#                 pass
#
#
#         try:
#             topics = self.news_dataframe['topic'].value_counts().index[:10]
#         except KeyError:
#             return 'ERROR: A dataframe must be passed if clean_news_dataframe() is not called'
#
#         distributions = self.news_dataframe['topic'].value_counts()[:10]
#
#         fig, ax = plt.subplots(figsize=(18,10))
#         bar_values = ['index', 'values']
#         ax.bar(topics, distributions, color='b')
#         ax.set_title('Distribution of Topics')
#         ax.set_ylabel('Number of news (log scaled)')
#         plt.yscale('log');
#         plt.savefig('topic_distribution.png')
#
#         subtopics = self.news_dataframe['subtopic'].value_counts().index[:10]
#         distributions = self.news_dataframe['subtopic'].value_counts()[:10]
#
#         fig, ax = plt.subplots(figsize=(18,10))
#         bar_values = ['index', 'values']
#         ax.bar(subtopics, distributions, color='orange')
#         ax.set_title('Most Popular Subtopics')
#         ax.set_ylabel('Number of news (log scaled)')
#         plt.yscale('log');
#         plt.savefig('subtopic_most_pop.png')
#
#
# class CreateFeatureMatrix():
#     '''Takes in language data and creates a feature matrix stored as an attribute of the name feature_matrix.
#     Option to utilize the MIND dataset directly or other strings in list or Series form.
#
#     Paramaters
#     ----------
#     features : str, ['LDA', 'NMF', 'TFIDF']
#     n_components : int, must be at least 2
#     ngram_range : tuple of two integers, first int must be equal to or less than the second
#
#     Methods
#     -------
#     featurize : featurize corpus as TFIDF, LDA, or NMF vectors
#
#     See Also
#     --------
#
#     Examples
#     --------
#     >>> data = ['This is a tool for building a content recommender',
#                 'Content recommenders can be challenging to evaluate',
#                 'Sports readers typically enjoy sports recommendations'
#                 'MIND is a userful dataset for studying recommenders',
#                 'medium.com is difficult to scrape from']
#     >>> create_matrix = CreateFeatureMatrix(data, MIND=False, n_components=3)
#     >>> create_matrix.featurize()
#     >>> create_matrix.feature_matrix
#         array([[0.70385031, 0.14807349, 0.1480762 ],
#                [0.18583332, 0.64621002, 0.16795666],
#                [0.33333333, 0.33333333, 0.33333333],
#                [0.18583223, 0.16795668, 0.64621109],
#                [0.33333333, 0.33333333, 0.33333333]])
#     '''
#
#     def __init__(self, data, MIND=True, ngram_range=(1,1), features='LDA', n_components=15):
#
#         self.MIND = MIND
#         if self.MIND:
#             self.data = data
#             self.corpus = data['content']
#             self._add_stopwords()
#         else:
#             self.corpus = data
#
#         self.vectorized = None
#         self.ngram_range = ngram_range
#         self.features = features
#         self.stopwords = set(stopwords.words('english'))
#
#         self.model = None
#         self.feature_matrix = None
#
#         self.n_components = n_components
#         self.reconstruction_errors = {}
#         self.feature_names = None
#
#
#     def _add_stopwords(self):
#
#         self.additional_stopwords = ['said', 'trump', 'just', 'like', '2019']
#         for word in self.additional_stopwords:
#             self.stopwords.add(word)
#
#     def featurize(self):
#
#         if self.features == 'LDA':
#             self._LDA()
#
#         elif self.features == 'NMF':
#             self._NMF()
#
#         elif self.features == 'TFIDF':
#             self._vectorize()
#
#
#     def _vectorize(self):
#     #
#         if self.features == 'LDA':
#             tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
#                                 stop_words='english', ngram_range = self.ngram_range) # max_features=n_features
#             self.vectorized = tf_vectorizer.fit_transform(self.corpus)
#             self.feature_names = tf_vectorizer.get_feature_names()
#
#         else:
#             tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=20,
#                                                stop_words='english', ngram_range=self.ngram_range)
#             self.vectorized = tfidf_vectorizer.fit_transform(self.corpus)
#             self.feature_names = tfidf_vectorizer.get_feature_names()
#             if self.features == 'TFIDF':
#                 self.feature_matrix = self.vectorized
#
#
#     def _LDA(self):
#
#         self._vectorize()
#
#         self.model = LDA(n_components = self.n_components).fit(self.vectorized)
#         self.feature_matrix = self.model.transform(self.vectorized)
#
#
#     def _NMF(self):
#
#         self._vectorize()
#
#         self.model = NMF(n_components=self.n_components, max_iter=400, random_state=1,
#               alpha=.1, l1_ratio=.5).fit(self.vectorized)
#
#         self.feature_matrix = self.model.transform(self.vectorized)
#
#
#     def _reconstruction_error(self): # get reconstruction error per n_components
#
#         for n_components in range(5,30,5):
#
#             if self.features == 'LDA':
#                 model = LDA(n_components=n_components).fit(self.vectorized)
#                 self.reconstruction_errors[n_components] = model.reconstruction_err_
#
#             else:
#                 model = NMF(n_components=n_components, max_iter=400, random_state=1,
#                   alpha=.1, l1_ratio=.5).fit(self.vectorized)
#                 self.reconstruction_errors[n_components] = model.reconstruction_err_
#
#
#     def plot_reconstruction_error(self): #plots elbow plot to find optimal n_components
#
#         self._reconstruction_error()
#
#         plt.plot(self.reconstruction_errors.keys(), self.reconstruction_errors.values(), color='orange')
#         plt.title('Reconstruction Error per Num Components')
#         plt.xlabel('Num Components')
#         plt.ylabel('Error')
#         plt.xticks(ticks=[5,10,15,20,25], labels=[5,10,15,50,25]);
#
#
#
#     def plot_top_words(self, n_top_words, title): # from sklearn documentation
#
#         fig, axes = plt.subplots(3, 5, figsize=(30, 25), sharex=True)
#         axes = axes.flatten()
#         for topic_idx, topic in enumerate(self.model.components_):
#             top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
#             top_features = [self.feature_names[i] for i in top_features_ind]
#             weights = topic[top_features_ind]
#
#             ax = axes[topic_idx]
#             if self.features == 'LDA':
#                 ax.barh(top_features, weights, height=0.7)
#             else:
#                 ax.barh(top_features, weights, height=0.7, color='b')
#             ax.set_title(f'Topic {topic_idx +1}',
#                          fontdict={'fontsize': 30})
#             ax.invert_yaxis()
#             ax.tick_params(axis='both', which='major', labelsize=20)
#             for i in 'top right left'.split():
#                 ax.spines[i].set_visible(False)
#             fig.suptitle(title, fontsize=40)
#
#         plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
#         plt.show();
#         plt.savefig('nmf_15.png')
#
#
# class ContentRecommender():
#
#     '''Generate recommendations for news news based on user input.
#     Built on mind.py and feature_matrix.py. Can compare recommendations to
#     target column to evaluate recommender performance.
#
#     Paramaters
#     -------------
#     corpus : list of document
#     data : DataFrame
#     feature_matrix : TFIDF, LDA, or NMF matrix (num_documents x num_features)
#     similarity_metric : str, default='cosine'
#         'jaccard', 'pearson'
#
#     Methods
#     -------
#     recommend :
#
#
#     Example
#     -------
#
#     Notes
#     -----
#
#     See Also
#     --------
#
#
#     '''
#
#     def __init__(self, news_dataframe, user_data, feature_matrix, similarity_metric='cosine'):
#
#         self.news_dataframe = news_dataframe
#         self.corpus = news_dataframe['content']
#         self.user_dataframe = user_data
#         self.feature_matrix = feature_matrix
#
#         self.similarity_dict = {'cosine': cosine_similarity} # , 'jaccard': jaccard, 'pearson': pearson
#         self.similarity_metric = similarity_metric
#
#         self.feature_names = None
#
#
#     def _calculate_covariance(self, user_vector=None):
#
#         return pd.DataFrame(self.similarity_dict[self.similarity_metric](self.feature_matrix, user_vector), index = self.news_dataframe.index)
#
#
#     def recommend(self, User_ID, by=None):
#
#         news = ''.join(list(self.user_dataframe[self.user_dataframe['User ID'] == User_ID]['Read news'].values))
#
#         news_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in news)].index
#
#         user_vector = self.feature_matrix.loc[news_ind].mean(axis=0).to_numpy().reshape(1,-1)
#
#         of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-10:].index[::-1]
#
#         print(f'We recommend the following news: ')
#         t = PrettyTable([' ', 'Title']) # create table of topic, interst level, and number recommended
#         for i, title in enumerate(list(self.news_dataframe['title'][of_interest_ind].values)):
#             t.add_row([i + 1, title])
#         t.align = 'l'
#         print(t)
#
#
#     def evaluate(self):
#
#         history_news = ''.join(list(self.user_dataframe[self.user_dataframe['User_ID'] == User_ID]['History'].values))
#         history_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in history_news)].index
#         impression_news = ''.join(list(self.user_dataframe[self.user_dataframe['User_ID'] == User_ID]['Impressions'].values))
#         impression_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in impression_news)].index
#
#         user_vector = self.feature_matrix.loc[history_ind].mean(axis=0).to_numpy().reshape(1,-1)
#         impressions_vectors = self.feature_matrix.loc[impression_ind].mean(axis=0).to_numpy().reshape(1,-1)
#         df = pd.DataFrame(self.similarity_dict[self.similarity_metric](impressions_vectors, user_vector)).sort_values(by=0)[-10:].index[::-1]
#
#         return df, self.user_dataframe['Target'][self.user_dataframe['User_ID'] == User_ID],
#
# #     def recommend(self, User_ID, by=None):
#
# #         news = ''.join(list(self.user_dataframe[self.user_dataframe['User ID'] == User_ID]['Read news'].values))
#
# #         news_ind = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in news)].index
#
# #         user_vector = self.feature_matrix.loc[news_ind].mean(axis=0).to_numpy().reshape(1,-1)
#
# #         of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-20:].index[::-1]
#
# #         return of_interest_ind
#
# #         R = self.news_dataframe['subtopic'][news_ind].values
#
# #         S = self.news_dataframe['title'][news_ind].values
#
# #         T = self.news_dataframe['subtopic'][of_interest_ind].values
#
# #         U = self.news_dataframe['title'][of_interest_ind].values
#
# #         return pd.DataFrame({'code': self.news_dataframe['code'][of_interest_ind].values, 'topic': self.news_dataframe['topic'][of_interest_ind].values, 'title': self.news_dataframe['title'][of_interest_ind].values}), self.news_dataframe[by][of_interest_ind].value_counts().index, self.news_dataframe[by][of_interest_ind].value_counts().values
#
#
#     def evaulate_user(self, num_users, by='topic'):
#         user_data = self.user_dataframe[self.user_dataframe['Read news'].apply(lambda x: len(x.split(' ')) > 100)]
#
#         for i in range(num_users): # range(user_data.shape[0])
#             user_read_news = self.news_dataframe[self.news_dataframe['code'].apply(lambda x: x in user_data.iloc[i,1].split(' '))]
#             d = {}
#             for j, v in zip(user_read_news[by].value_counts().index, user_read_news[by].value_counts().values):
#                 d[j] = v
#
#             w, x, y = self.recommend(user_data.iloc[i,0], by)
#
#             d2 = {}
#             for i, v in zip(x,y):
#                 d2[i] = v
#
#
#             tot = sum(d.values())
#             for k, v in d.items():
#                 d[k] = v/tot
#
# #             good = False
# #             for v in d.values():
# #                 if v > .4:
# #                     good == True
#
# #             if good:
#             d_topics = {}
#             for k, v in d.items():
#                 if v > .4:
#                     d_topics['H'] = [k]
#
#                 if v > .25 and v <= .4:
#                     if 'M' not in d_topics:
#                         d_topics['M'] = []
#                     d_topics['M'].append(k)
#
#                 if v > .1 and v <= .25:
#                     if 'L' not in d_topics:
#                         d_topics['L'] = []
#                     d_topics['L'].append(k)
#
#                 if v <= .1:
#                     if 'VL' not in d_topics:
#                         d_topics['VL'] = []
#                     d_topics['VL'].append(k)
#
# #             if 'H' in d_topics:
#             t = PrettyTable([by, 'Interest', '%', '#']) # create table of topic, interst level, and number recommended
#             z = 0
#             for k, v in d_topics.items():
#                 for i in v:
#                     if i in d2:
#                         t.add_row([i.title(), k, round(list(d.values())[z], 2) * 100, d2[i]])
#                     else:
#                         t.add_row([i.title(), k, round(list(d.values())[z], 2) * 100, 0])
#                     z+=1
#             print(t)
#             print('\n')
#
#             print(w)
#
#
#     def clean_user_data_general(self):
#
#         self.user_dataframe.columns = ['Impression_ID', 'User_ID', 'Timestamp', 'History', 'Impressions']
#         self.user_dataframe.drop(self.user_dataframe[self.user_dataframe['History'].isna()]
#                             .index, inplace=True)
#         self.user_dataframe.drop('Impression ID', axis=1, inplace=True)
#
#         d={} # collect user impressions
#         for row in self.user_dataframe.iterrows():
#             if row[1][0] not in d.keys():
#                 d[row[1][0]] = []
#             d[row[1][0]].append(row[1][3])
#         od = collections.OrderedDict(sorted(d.items()))
#         self.user_dataframe.drop_duplicates(subset='User ID', inplace=True)
#         self.user_dataframe.sort_values(by='User ID', inplace=True)
#         self.user_dataframe.drop('Impressions', axis=1, inplace=True)
#         self.user_dataframe['Impressions'] = od.values()
#
#         news_read = {}
#         impressions = self.user_dataframe['Impressions'].apply(lambda x: (' '.join(x).split(' ')))
#         for i, impression in enumerate(impressions):
#             news_read[i] = []
#             for article in impression:
#                 if '-1' in article:
#                     news_read[i].append(article[:-2])
#
#         self.user_dataframe['Read news'] = list(news_read.values())
#         self.user_dataframe['Read news'] = (self.user_dataframe['Read news']
#                                            .apply(lambda x: ' '.join(x)))
#         self.user_dataframe['Read news'] = (self.user_dataframe['History'] + ' ' +
#                                            self.user_dataframe['Read news'])
#         self.user_dataframe.drop(['History', 'Time', 'Impressions'], axis=1,
#                             inplace=True)
#
#
# '''Clean and prepare MIND news and behaviors datasets for analaysis.
# Various methods associated with the class.
# User IDs and History values are not unique in behaviors.tsv, only
# Timestamp, Impression IDs and Impressions are.
#
# Parameters
# ----------
# news_filepath : str
# behaviors_filepath: str
#
# Methods
# -------
# clean_news_dataframe : cleans and prepares news.tsv
# get_body_of_news : requests HTML for each article in links column of news.tsv;
#     saving html to mongoDB requires running docker container
# add_content :
# clean_user_data : cleans and prepares behaviors.tsv
# plot_topic_distributions : plot topic and subtopic distributions
#
# See More
# --------
# Read about the data at https://msnews.github.io/#about-mind.
#
# '''
#
