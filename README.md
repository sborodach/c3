### About
Utilizing the small [MIcrosoft News Dataset (MIND)](https://msnews.github.io/#about-mind), I sought to build my own recommender in order to explore the challenges of building effective recommendation systems. I learned to couple topic modeling techniques with similarity metrics to make content based recommendations for users given their reading history. Since no metric exists to evaluate such recommendations, assessing the recommender’s performance required qualitatively comparing individual instances of user recommendations with their reading history. I found the recommender performs well given topic modeling features matrices, specifically under LDA, indicating that dimensionality reduction captures underlying themes in the news articles.



### Process
1. Data Wrangling  
  MIND provides two datasets. 
  news.tsv, 51,000+ news articles: code, topic, subtopic, link to article. Saved as news_data.csv
  behaviors.tsv, 50,000 unique user IDs: history of articles read (article codes), impressions (explain). collect impressions by user, add to articles read to create history column. new frame has user ID and total history. Saved as user_data.csv

2. Data scraping  
  I looped through the links contained in news_data.csv, requesting the html, saving it to a mongo database. Once html was collected for all the articles, it was parsed, and cleaned, leaving only article bodies to be added to the dataframe.
  
3. Vectorization and Topic Modeling  
  The CreateFeatureMatrix class utilizes either a Count Vectorizer or a TFIDF vectorizer depending on the specification of the user. A Count Vectorizer is appropriate to use for LDA. See [this paper](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) for further explanation on LDA. The recommender can be run using a TFIDF feature matrix, or, once vectorized, the data can be reduced with an NMF model.

4. Recommendation  
  Finally, recommednations are used by passing in a single User ID into the ContentRecommender class, which creates a user vector by averaging together the feature values across all topics for all the articles read by the user. Then the cosine similarity is calculated between the newly generated user vector and the remaining corpus articles. Laslty, article titles with links are outputted for the user.

### Findings
Content recommenders are challenging to evaluate. See [this paper] 

This project pushed me to embrace object-oriented-programming, creating classes that can be easily utilized by myself and others in order to reproduce the work done here.

### Future Work
1. scraping medium.com
2. Deploying flask app
3. How to output topic model relevance per user
  A. take LDA output, give title to each topic
  B. find measures to make most relevant topics, genreate user description
  C. summarize output of top atricles recommended

### Gratitude
Thank you to the staff at Galvanize Austin for stimulating this project! Thanks Juliana Duncan, Dan Rupp, Kiara Hearn, and Kristen Grewe.

<img align="center" width="350" height="250" src="https://github.com/sborodach/news-content-recommender/blob/main/img/tech_stack1.png">
