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


class SmallData():
    '''Clean and prepare MIND news and behaviors datasets for analaysis.
    Various methods associated with the class.
    User IDs and History values are not unique in behaviors.tsv, only 
    Timestamp, Impression IDs and Impressions are.
    
    Parameters
    ----------
    news_filepath : str
    behaviors_filepath: str
    
    Methods
    -------
    clean_news_data : cleans and prepares news.tsv
    get_content : requests HTML for each article in links column of news.tsv;
        saving html to mongoDB requires running docker container
    add_content : 
    clean_user_data : cleans and prepares behaviors.tsv
    plot_topic_distributions : plot topic and subtopic distributions
    
    See More
    --------
    Read about the data at https://msnews.github.io/#about-mind.
    
    '''
    
    def __init__(self, news_filepath, behaviors_filepath):
        self.news_data = (pd.read_csv(news_filepath, sep='\t').T.reset_index()
                          .T.reset_index(drop=True))
        self.user_data = (pd.read_csv(behaviors_filepath, sep='\t').T.
                          reset_index().T.reset_index(drop=True))
        self.clean = False

        
    def clean_news_data(self):
        # clean news data frame
        
        self.news_data.columns = ['code', 'topic', 'subtopic', 'title', 
                                  'abstract', 'link', 'tags1', 'tags2']
        self.news_data.drop(self.news_data[self.news_data['link'].isna()].
                            index, inplace=True)
        self.news_data['topic'] = self.news_data['topic'].apply(lambda x: 
                                                                x.upper())
        self.news_data['subtopic'] = self.news_data['subtopic'].apply(lambda x: 
                                                                      x.title())
        self.clean = True

        
    def get_content(self): 
        # Request html for each article, parse, save to mongo database
        
        urls = self.news_data['link'].values
       
        client = MongoClient() # request html, parse, save to mongo
        db = client['news-html']
        for i, url in enumerate(urls):
            try:
                res = requests.get(url)
                html = res.content
                unicode_str = html.decode("utf8")
                encoded_str = unicode_str.encode("ascii",'ignore')
                soup = BeautifulSoup(encoded_str, "html.parser")
                text = soup.find_all('p')
                y = [re.sub(r'<.+?>',r'',str(a)) for a in text]
                db['html'].insert_one({'link': url, 'html': ' '.join(y)})
            except:
                db['html'].insert_one({'link': url})
                
                
    def add_content(self):
        # add article content from mongodb to article data frame
        
        client = MongoClient()
        db = client['news-html']
        content = pd.Series(list(db['html'].find({}, {'html':1, '_id':0})))
        self.news_data['content'] = list(db['html'].find({}, {'html':1, '_id':0}))
        self.news_data.drop(self.news_data[self.news_data['content'] == {}]
                            .index, inplace=True)
        self.news_data['content'] = self.news_data['content'].apply(lambda x: 
                                                                    str(list(x.values())[0]))
        self.news_data.drop(self.news_data[self.news_data['content'].isna()]
                            .index, inplace=True)

        
    def clean_user_data(self):
        
        self.user_data.columns = ['Impression ID', 'User ID', 'Time', 
                                  'History', 'Impressions'] # set column names
        self.user_data.drop(self.user_data[self.user_data['History'].isna()]
                            .index, inplace=True)
        self.user_data.drop('Impression ID', axis=1, inplace=True)
        
        d={} # collect user impressions
        for row in self.user_data.iterrows():
            if row[1][0] not in d.keys():
                d[row[1][0]] = []
            d[row[1][0]].append(row[1][3])
        od = collections.OrderedDict(sorted(d.items()))
        self.user_data.drop_duplicates(subset='User ID', inplace=True)
        self.user_data.sort_values(by='User ID', inplace=True)
        self.user_data.drop('Impressions', axis=1, inplace=True)
        self.user_data['Impressions'] = od.values()
            
        articles_read = {}
        impressions = self.user_data['Impressions'].apply(lambda x: (' '.join(x).split(' ')))
        for i, impression in enumerate(impressions):
            articles_read[i] = []
            for article in impression:
                if '-1' in article:
                    articles_read[i].append(article[:-2])
        
        self.user_data['Read Articles'] = list(articles_read.values())
        self.user_data['Read Articles'] = (self.user_data['Read Articles']
                                           .apply(lambda x: " ".join(x)))
        self.user_data['Read Articles'] = (self.user_data['History'] + ' ' + 
                                           self.user_data['Read Articles'])
        self.user_data.drop(['History', 'Time', 'Impressions'], axis=1, 
                            inplace=True)
        
        
    def plot_topic_distrubtions(self, news_data=None):
        # Create plots of distributions of topics and subtopics. 
        # If clean_data() has been called, this plotting function will 
        # use the object attrbitues self.news_data. If not, a dataframe
        # must be passed in.

        if not self.clean:
            try:
                topics = news_data['topic'].value_counts().index[:10]
                distributions = news_data['topic'].value_counts()[:10]

                fig, ax = plt.subplots(figsize=(18,10))
                bar_values = ['index', 'values']
                ax.bar(topics, distributions, color='b')
                ax.set_title('Distribution of Topics')
                ax.set_ylabel('Number of articles (log scaled)')
                plt.yscale('log');
                plt.savefig('topic_distribution.png')

                subtopics = news_data['subtopic'].value_counts().index[:10]
                distributions = news_data['subtopic'].value_counts()[:10]

                fig, ax = plt.subplots(figsize=(18,10))
                bar_values = ['index', 'values']
                ax.bar(subtopics, distributions, color='orange')
                ax.set_title('Most Popular Subtopics')
                ax.set_ylabel('Number of articles (log scaled)')
                plt.yscale('log');
                plt.savefig('subtopics_most_pop.png')
                return None
            except:
                pass
 

        try:
            topics = self.news_data['topic'].value_counts().index[:10]
        except KeyError:
            return 'ERROR: A dataframe must be passed if clean_news_data() is not called'
        
        distributions = self.news_data['topic'].value_counts()[:10]

        fig, ax = plt.subplots(figsize=(18,10))
        bar_values = ['index', 'values']
        ax.bar(topics, distributions, color='b')
        ax.set_title('Distribution of Topics')
        ax.set_ylabel('Number of articles (log scaled)')
        plt.yscale('log');
        plt.savefig('topic_distribution.png')

        subtopics = self.news_data['subtopic'].value_counts().index[:10]
        distributions = self.news_data['subtopic'].value_counts()[:10]

        fig, ax = plt.subplots(figsize=(18,10))
        bar_values = ['index', 'values']
        ax.bar(subtopics, distributions, color='orange')
        ax.set_title('Most Popular Subtopics')
        ax.set_ylabel('Number of articles (log scaled)')
        plt.yscale('log');
        plt.savefig('subtopic_most_pop.png')


class CreateFeatureMatrix():
    '''Takes in language data and creates a feature matrix stored as an attribute of the name feature_matrix.
    Option to utilize the MIND dataset directly or other strings in list or Series form.
    
    Paramaters
    ----------
    features : str, 'LDA', 'NMF', or 'TFIDF'
    n_components : int, must be at least 2
    ngram_range : tuple of two integers, first int must be equal to or less than the second
    
    Methods
    -------
    featurize : featurize corpus as TFIDF, LDA, or NMF vectors
    
    See Also
    --------
    
    Examples
    --------
    >>> data = ['This is a tool for building a content recommender',
                'Content recommenders can be challenging to evaluate',
                'Sports readers typically enjoy sports recommendations'
                'MIND is a userful dataset for studying recommenders',
                'medium.com is difficult to scrape from']
    >>> create_matrix = CreateFeatureMatrix(data, MIND=False, n_components=3)
    >>> create_matrix.featurize()
    >>> create_matrix.feature_matrix
        array([[0.70385031, 0.14807349, 0.1480762 ],
               [0.18583332, 0.64621002, 0.16795666],
               [0.33333333, 0.33333333, 0.33333333],
               [0.18583223, 0.16795668, 0.64621109],
               [0.33333333, 0.33333333, 0.33333333]])
    '''
    
    def __init__(self, data, MIND=True, ngram_range=(1,1), features='LDA', n_components=15):
        
        self.MIND = MIND
        if self.MIND:
            self.data = data
            self.corpus = data['content']
            self._add_stopwords()
        else:
            self.corpus = data
        
        self.vectorized = None
        self.ngram_range = ngram_range
        self.features = features
        self.stopwords = set(stopwords.words('english'))
        
        self.model = None
        self.feature_matrix = None
        
        self.n_components = n_components
        self.reconstruction_errors = {}
        self.feature_names = None
        

    def _add_stopwords(self):
        
        self.additional_stopwords = ['said', 'trump', 'just', 'like', '2019']
        for word in self.additional_stopwords:
            self.stopwords.add(word)
            
    def featurize(self):
        
        if self.features == 'LDA':
            self._LDA()
            
        elif self.features == 'NMF':
            self._NMF()
            
        else:
            self._vectorize()
    
    
    def _vectorize(self):
    # 
        if self.features == 'LDA':
            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english', ngram_range = self.ngram_range) # max_features=n_features
            self.vectorized = tf_vectorizer.fit_transform(self.corpus)
            self.feature_names = tf_vectorizer.get_feature_names()
            
        else:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=20,
                                               stop_words='english', ngram_range=self.ngram_range)
            self.vectorized = tfidf_vectorizer.fit_transform(self.corpus)
            self.feature_names = tfidf_vectorizer.get_feature_names()
            if self.features == 'TFIDF':
                self.feature_matrix = self.vectorized
                
                
    def _LDA(self):

        self._vectorize()
        
        self.model = LDA(n_components = self.n_components).fit(self.vectorized)
        self.feature_matrix = self.model.transform(self.vectorized)

    
    def _NMF(self):
        
        self._vectorize()
        
        self.model = NMF(n_components=self.n_components, max_iter=400, random_state=1,
              alpha=.1, l1_ratio=.5).fit(self.vectorized)
        
        self.feature_matrix = self.model.transform(self.vectorized)

    
    def _reconstruction_error(self): # get reconstruction error per n_components
        
        for n_components in range(5,30,5):
            
            if self.features == 'LDA':
                model = LDA(n_components=n_components).fit(self.vectorized)
                self.reconstruction_errors[n_components] = model.reconstruction_err_
                
            else:
                model = NMF(n_components=n_components, max_iter=400, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(self.vectorized)
                self.reconstruction_errors[n_components] = model.reconstruction_err_

            
    def plot_reconstruction_error(self): #plots elbow plot to find optimal n_components

        self._reconstruction_error()
        
        plt.plot(self.reconstruction_errors.keys(), self.reconstruction_errors.values(), color='orange')
        plt.title('Reconstruction Error per Num Components')
        plt.xlabel('Num Components')
        plt.ylabel('Error')
        plt.xticks(ticks=[5,10,15,20,25], labels=[5,10,15,50,25]);
        
        
        
    def plot_top_words(self, n_top_words, title): # from sklearn documentation

        fig, axes = plt.subplots(3, 5, figsize=(30, 25), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            if self.features == 'LDA':
                ax.barh(top_features, weights, height=0.7)
            else:
                ax.barh(top_features, weights, height=0.7, color='b')
            ax.set_title(f'Topic {topic_idx +1}',
                         fontdict={'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show();
        plt.savefig('nmf_15.png')


class ContentRecommender():
    
    '''Generate recommendations for news articles based on user input.
    Built on mind.py and feature_matrix.py. Can pass
    
    Paramaters
    -------------
    corpus : list of document
    data : DataFrame
    feature_matrix : TFIDF, LDA, or NMF matrix (num_documents x num_features)
    similarity_metric : str, default='cosine'
        'jaccard', 'pearson'
        
    Methods
    -------
    recommend : 
        
    
    Example
    -------
    
    Notes
    -----
    
    See Also
    --------
    
    
    '''
    
    def __init__(self, news_data, user_data, feature_matrix, similarity_metric='cosine'):
        
        self.news_data = news_data
        self.corpus = news_data['content']
        self.user_data = user_data
        self.feature_matrix = feature_matrix

        self.similarity_dict = {'cosine': cosine_similarity} # , 'jaccard': jaccard, 'pearson': pearson
        self.similarity_metric = similarity_metric
        
        self.feature_names = None

        
    def _calculate_covariance(self, doc_ind=None, user_vector=None):
        
        return pd.DataFrame(self.similarity_dict[self.similarity_metric](self.feature_matrix, user_vector), index = self.news_data.index)

    
    def recommend(self, User_ID, by=None):
        
        articles = ''.join(list(self.user_data[self.user_data['User ID'] == User_ID]['Read Articles'].values))

        articles_ind = self.news_data[self.news_data['code'].apply(lambda x: x in articles)].index
        
        user_vector = self.feature_matrix.loc[articles_ind].mean(axis=0).to_numpy().reshape(1,-1)

        of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-10:].index[::-1]

        print(f'We recommend the following articles: ')
        t = PrettyTable([' ', 'Title']) # create table of topic, interst level, and number recommended
        for i, title in enumerate(list(self.news_data['title'][of_interest_ind].values)):
            t.add_row([i + 1, title])
        t.align = 'l'
        print(t)
    
    
#     def recommend(self, User_ID, by=None):
        
#         articles = ''.join(list(self.user_data[self.user_data['User ID'] == User_ID]['Read Articles'].values))

#         articles_ind = self.news_data[self.news_data['code'].apply(lambda x: x in articles)].index
        
#         user_vector = self.feature_matrix.loc[articles_ind].mean(axis=0).to_numpy().reshape(1,-1)

#         of_interest_ind = self._calculate_covariance(self.feature_matrix.to_numpy(), user_vector).sort_values(by=0)[-20:].index[::-1]

#         return of_interest_ind
        
#         R = self.news_data['subtopic'][articles_ind].values
        
#         S = self.news_data['title'][articles_ind].values
        
#         T = self.news_data['subtopic'][of_interest_ind].values

#         U = self.news_data['title'][of_interest_ind].values

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