import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('ggplot')

from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import re
import collections

class MINDSmallData():
    
    def __init__(self):
        self.news_data = pd.read_csv('MINDsmall_train/news.tsv', sep='\t')
        self.user_data = pd.read_csv('MINDsmall_train/behaviors.tsv', sep='\t')
        self.clean = False

    def clean_article_data(self):
        '''
        Clean news data frame
        '''
        # adds columns values as first row of dataframe
        self.news_data.loc[-1] = self.news_data.columns
        self.news_data.index = self.news_data.index + 1
        self.news_data = self.news_data.sort_index()

        # sets column values
        self.news_data.columns = ['code', 'topic', 'subtopic', 'title', 'abstract', 'link', 'tags1', 'tags2']

        # drop 'content' na values
        self.news_data.drop(self.news_data[self.news_data['abstract'].isna()].index, inplace=True)
        
        # drops rows with subtopics that only appear once
        one_time_subtopics = list(self.news_data['subtopic'].value_counts()[(self.news_data['subtopic'].value_counts() == 1).values].index) 
        self.news_data.drop(self.news_data[self.news_data['subtopic'].apply(lambda x: x in one_time_subtopics)].index, inplace=True)

        # clean up topic names
        self.news_data['topic'].replace('foodanddrink','FOOD & DRINK', inplace=True)
        self.news_data['topic'].replace('autos','CARS', inplace=True)

        self.news_data['topic'] = self.news_data['topic'].apply(lambda x: x.upper())
        self.news_data['topic'].replace('MIDDLEEAST', 'NEWS', inplace=True)
        
        self.news_data.drop(self.news_data[self.news_data['topic'] == 'KIDS'].index, inplace=True)
        self.news_data.drop(self.news_data[self.news_data['topic'] == 'VIDEO'].index, inplace=True)

        self.news_data['subtopic'] = self.news_data['subtopic'].apply(lambda x: x.upper())
        subtopic_dict = {'WEATHERTOPSTORIES': 'WEATHER', 'FOOTBALL_NFL': 'NFL', 'NEWSSCIENCEANDTECHNOLOGY': 'SCIENCE & TECHNOLOGY',
                        'NEWSPOLITICS': 'POLITICS', 'BASEBALL_MLB': 'MLB', 'NEWSUS': 'US NEWS', 'BASKETBALL_NBA': 'NBA', 'NEWSCRIME': 'CRIME',
                        'NEWSWORLD': 'WORLD NEWS', 'FOOTBALL_NCAA': 'NCAA FOOTBALL', 'LIFESTYLEROYALS': 'ROYALTY LIFESTYLE'}
        
        self.news_data['subtopic'].replace(subtopic_dict, inplace=True)
        self.news_data['subtopic'] = self.news_data['subtopic'].apply(lambda x: x.title())
        
        self.clean = True

    def get_content(self): 
        '''
        Request html for each article, parse, save to mongo
        '''
        
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
        '''
        Add article content from mongodb to article data frame
        '''
        
        client = MongoClient()
        db = client['news-html']
        content = pd.Series(list(db['html'].find({}, {'html':1, '_id':0})))
        self.news_data['content'] = list(db['html'].find({}, {'html':1, '_id':0}))
        self.news_data.drop(self.news_data[self.news_data['content'] == {}].index, inplace=True)
        self.news_data['content'] = self.news_data['content'].apply(lambda x: str(list(x.values())[0]))
        self.news_data.drop(self.news_data[self.news_data['content'].isna()].index, inplace=True)

        
    def clean_user_data(self):
        '''
        Clean user behaviors datatframe
        '''
        
        self.user_data.loc[-1] = self.user_data.columns # add columns values as first row of dataframe
        self.user_data.index = self.user_data.index + 1
        self.user_data = self.user_data.sort_index()

        self.user_data.columns = ['Impression ID', 'User ID', 'Time', 'History', 'Impressions'] # set column names

        self.user_data.drop(self.user_data[self.user_data['History'].isna()].index, inplace=True) # drop na values in 'History' column
        self.user_data.drop('Impression ID', axis=1, inplace=True) # drop 'Impression ID' column
        
        d={} # collect user impressions
        for row in self.user_data.iterrows():
            if row[1][0] not in d.keys():
                d[row[1][0]] = []
            d[row[1][0]].append(row[1][3])
        od = collections.OrderedDict(sorted(d.items())) # order dictionary by keys
        self.user_data.drop_duplicates(subset='User ID', inplace=True) # removes duplicate user appearances
        self.user_data.sort_values(by='User ID', inplace=True)
        self.user_data.drop('Impressions', axis=1, inplace=True) # drop 'Impression ID' column
        self.user_data['Impressions'] = od.values()
            
        articles_read = {} # collect articles read for each impression
        impressions = self.user_data['Impressions'].apply(lambda x: (' '.join(x).split(' ')))
        for i, impression in enumerate(impressions):
            articles_read[i] = []
            for article in impression:
                if '-1' in article:
                    articles_read[i].append(article[:-2])

        
        self.user_data['Read Articles'] = list(articles_read.values()) # create column Articles Read in dataframe
        self.user_data['Read Articles'] = self.user_data['Read Articles'].apply(lambda x: " ".join(x))

        self.user_data['Read Articles'] = self.user_data['History'] + ' ' + self.user_data['Read Articles']
        self.user_data.drop(['History', 'Time', 'Impressions'], axis=1, inplace=True)
        
        
    def plot_topic_distrubtions(self, article_data=None):

        '''
        Creates two plots: distributions of topics and subtopics. If clean_data() has been called,
        this plotting function will use the object attrbitues self.news_data. If not, a dataframe
        must be passed in
        '''

        if not self.clean:
            try:
                topics = article_data['topic'].value_counts().index[:10]
                distributions = article_data['topic'].value_counts()[:10]

                fig, ax = plt.subplots(figsize=(18,10))
                bar_values = ['index', 'values']
                ax.bar(topics, distributions, color='b')
                ax.set_title('Distribution of Topics')
                ax.set_ylabel('Number of articles (log scaled)')
                plt.yscale('log');
                plt.savefig('topic_distribution.png')

                topics = article_data['subtopic'].value_counts().index[:10]
                distributions = article_data['subtopic'].value_counts()[:10]

                fig, ax = plt.subplots(figsize=(18,10))
                bar_values = ['index', 'values']
                ax.bar(topics, distributions, color='orange')
                ax.set_title('Most Popular Subtopics')
                ax.set_ylabel('Number of articles (log scaled)')
                plt.yscale('log');
                plt.savefig('subtopic_most_pop.png')
                return None
            except:
                pass
 

        try:
            topics = self.news_data['topic'].value_counts().index[:10]
        except KeyError:
            return 'ERROR: A dataframe must be passed if clean_article_data() is not called'
        
        distributions = self.news_data['topic'].value_counts()[:10]

        fig, ax = plt.subplots(figsize=(18,10))
        bar_values = ['index', 'values']
        ax.bar(topics, distributions, color='b')
        ax.set_title('Distribution of Topics')
        ax.set_ylabel('Number of articles (log scaled)')
        plt.yscale('log');
        plt.savefig('topic_distribution.png')

        topics = self.news_data['subtopic'].value_counts().index[:10]
        distributions = self.news_data['subtopic'].value_counts()[:10]

        fig, ax = plt.subplots(figsize=(18,10))
        bar_values = ['index', 'values']
        ax.bar(topics, distributions, color='orange')
        ax.set_title('Most Popular Subtopics')
        ax.set_ylabel('Number of articles (log scaled)')
        plt.yscale('log');
        plt.savefig('subtopic_most_pop.png')