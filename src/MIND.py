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
    '''Clean and prepare MIND news and behaviors datasets for analaysis.
    Various methods associated with the class.
    User IDs and History values are not unique in behaviors.tsv, only Timestamp, Impression IDs and Impressions are.
    
    Parameters
    ----------
    news_filepath : str
    behaviors_filepath: str
    
    See More
    --------
    Read about the data at https://msnews.github.io/#about-mind.
    
    '''
    
    def __init__(self, news_filepath, behaviors_filepath):
        self.news_data = pd.read_csv(news_filepath, sep='\t').T.reset_index().T.reset_index(drop=True)
        self.user_data = pd.read_csv(behaviors_filepath, sep='\t').T.reset_index().T.reset_index(drop=True)
        self.clean = False

    def clean_news_data(self):
        # clean news data frame
        
        self.news_data.columns = ['code', 'topic', 'subtopic', 'title', 'abstract', 'link', 'tags1', 'tags2']
        self.news_data.drop(self.news_data[self.news_data['link'].isna()].index, inplace=True)
        self.news_data['topic'] = self.news_data['topic'].apply(lambda x: x.upper())
        self.news_data['subtopic'] = self.news_data['subtopic'].apply(lambda x: x.title())
        
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
        self.news_data.drop(self.news_data[self.news_data['content'] == {}].index, inplace=True)
        self.news_data['content'] = self.news_data['content'].apply(lambda x: str(list(x.values())[0]))
        self.news_data.drop(self.news_data[self.news_data['content'].isna()].index, inplace=True)

        
    def clean_user_data(self):
        # clean user behaviors datatframe
        
        self.user_data.columns = ['Impression ID', 'User ID', 'Time', 'History', 'Impressions'] # set column names
        self.user_data.drop(self.user_data[self.user_data['History'].isna()].index, inplace=True)
        self.user_data.drop('Impression ID', axis=1, inplace=True)
        
        d={} # collect user impressions
        for row in self.user_data.iterrows():
            if row[1][0] not in d.keys():
                d[row[1][0]] = []
            d[row[1][0]].append(row[1][3])
        od = collections.OrderedDict(sorted(d.items())) # order dictionary by keys
        self.user_data.drop_duplicates(subset='User ID', inplace=True) # removes duplicate user appearances
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
        
        self.user_data['Read Articles'] = list(articles_read.values()) # create column Articles Read in dataframe
        self.user_data['Read Articles'] = self.user_data['Read Articles'].apply(lambda x: " ".join(x))
        self.user_data['Read Articles'] = self.user_data['History'] + ' ' + self.user_data['Read Articles']
        self.user_data.drop(['History', 'Time', 'Impressions'], axis=1, inplace=True)
        
        
    def plot_topic_distrubtions(self, news_data=None):
        # Creates two plots: distributions of topics and subtopics. 
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
        
        
        
        
        
        
        
#         one_time_subtopics = list(self.news_data['subtopic'].value_counts()[(self.news_data['subtopic'].value_counts() == 1).values].index) 
#         self.news_data.drop(self.news_data[self.news_data['subtopic'].apply(lambda x: x in one_time_subtopics)].index, inplace=True)

#         self.news_data['topic'].replace('foodanddrink','FOOD & DRINK', inplace=True)
#         self.news_data['topic'].replace('autos','CARS', inplace=True)

#         self.news_data['topic'].replace('MIDDLEEAST', 'NEWS', inplace=True)
        
#         self.news_data.drop(self.news_data[self.news_data['topic'] == 'KIDS'].index, inplace=True)
#         self.news_data.drop(self.news_data[self.news_data['topic'] == 'VIDEO'].index, inplace=True)

#         self.news_data['subtopic'] = self.news_data['subtopic'].apply(lambda x: x.upper())
#         subtopic_dict = {'WEATHERTOPSTORIES': 'WEATHER', 'FOOTBALL_NFL': 'NFL', 'NEWSSCIENCEANDTECHNOLOGY': 'SCIENCE & TECHNOLOGY',
#                         'NEWSPOLITICS': 'POLITICS', 'BASEBALL_MLB': 'MLB', 'NEWSUS': 'US NEWS', 'BASKETBALL_NBA': 'NBA', 'NEWSCRIME': 'CRIME',
#                         'NEWSWORLD': 'WORLD NEWS', 'FOOTBALL_NCAA': 'NCAA FOOTBALL', 'LIFESTYLEROYALS': 'ROYALTY LIFESTYLE'}
#         self.news_data['subtopic'].replace(subtopic_dict, inplace=True)