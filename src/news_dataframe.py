import numpy as np
import pandas as pd
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import re

class NewsDataFrameProcessor():

    def __init__(self, news_tsv_filepath):
        self.news_dataframe = (pd.read_csv(news_tsv_filepath, sep='\t').T.reset_index().T.reset_index(drop=True))

    def clean_news_dataframe(self):
        newsDataFrameCleaner = NewsDataFrameCleaner(self.news_dataframe)
        self.news_dataframe = newsDataFrameCleaner.news_dataframe

    def add_news_content_column(self):
        newsContentColumnAdder = NewsContentColumnAdder(self.news_dataframe['URL'])
        self.news_dataframe['Content'] = newsContentColumnAdder.news_content

class NewsDataFrameCleaner():

    def __init__(self, news_dataframe):
        self.news_dataframe = news_dataframe
        self.news_dataframe.drop(0, axis=1, inplace=True)
        self.news_dataframe.columns = ['Code', 'Topic', 'Subtopic', 'Title',
                                  'Synopsis', 'URL', 'Tags1', 'Tags2']
        self._drop_nan_values_in_url_column()
        self._capitalize_values_in_topic_column()
        self._title_values_in_subtopic_column()

    def _drop_nan_values_in_url_column(self):
        self.news_dataframe.drop(self.news_dataframe[self.news_dataframe['URL'].isna()].index, inplace=True)

    def _capitalize_values_in_topic_column(self):
        self.news_dataframe['Topic'] = self.news_dataframe['Topic'].apply(lambda x: x.upper())

    def _title_values_in_subtopic_column(self):
        self.news_dataframe['Subtopic'] = self.news_dataframe['Subtopic'].apply(lambda x: x.title())

class NewsContentColumnAdder():

    def __init__(self, urls):
        self.urls = urls
        self.url = ''
        self.news_content = None
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
        return [re.sub(r'<.+?>',r'',str(a)) for a in text]

    def _save_html_to_mongo_database(self):
        mongoClient = MongoClient()
        mongo_database = mongoClient['news-html-test']
        for url in self.urls:
            self.url = url
            try:
                y = self._parse_html()
                mongo_database['html'].insert_one({'link': url, 'html': ' '.join(y)})
            except:
                mongo_database['html'].insert_one({'link': url})
        return pd.DataFrame(list(mongo_database['html'].find({}, {'html':1, '_id':0})))

    def _process_news_content(self):
        self.news_content = self._save_html_to_mongo_database()

if __name__ == "__main__":
    news_tsv_filepath = '../MINDsmall_train/news_test.tsv'
    newsDataFrameProcessor = NewsDataFrameProcessor(news_tsv_filepath)
    newsDataFrameProcessor.clean_news_dataframe()
    newsDataFrameProcessor.add_news_content_column()
