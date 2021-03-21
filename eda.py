import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import extractTweet
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer

import numpy as np
import seaborn as sns
from plotly import graph_objs as go

def app():
    sentiment_dict = {
    0:'Strongly negative',
    1:'Strongly Positive',
    2:'negative',
    3:'positive',
    4:'sarcasm',
    5:'neutral'
    }
    def density_calculator(data, attribute):
        # declaring cvec object
        tk = TweetTokenizer()
        cvec = CountVectorizer(stop_words='english',tokenizer=tk.tokenize)
        # fitting data into cvec
        cvec.fit(data.reviews)

        # negative Data matrix

        # @st.cache(persist=True)
        def inner_helper(target):
            doc_matrix = cvec.transform(data[data[attribute] == target].reviews)
            doc_tf = np.sum(doc_matrix, axis=0)
            return np.squeeze(np.asarray(doc_tf))
        word_polarity_dict={}
        for key in sentiment_dict.keys():
            word_polarity_dict[sentiment_dict[key]] = inner_helper(key)

        # neg = inner_helper(0)
        # pos = inner_helper(4)
        return word_polarity_dict, cvec.get_feature_names()

    def createDocumentMatrix(data, max_features):
        tk = TweetTokenizer()
        cvec = CountVectorizer(stop_words='english',
                               max_features=max_features,tokenizer=tk.tokenize)  # Removing stop words this time and limiting studied words to 10k
        cvec.fit(data.reviews)
        document_matrix = cvec.transform(data.reviews)
        return document_matrix, cvec.get_feature_names()

    def calculate_term_frequency(data, document_matrix, target):
        batches = np.linspace(0, len(data[data['reviews'] == target]), 10).astype('int')
        b_tf = []
        i = 0
        while i < len(batches) - 1:
            batch_result = np.sum(document_matrix[batches[i]:batches[i + 1]].toarray(), axis=0)
            b_tf.append(batch_result)
            i += 1
        return b_tf

    def show_string(data, attribute, target):
        innertweets = data[data[attribute] == target]
        # st.write(data[[0]])

        # st.write(data[data[attribute] == 0])
        string = []
        for t in innertweets.reviews:
            string.append(t)
        # st.write(string)
        string = pd.Series(string).str.cat(sep='')
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(string)

        return wordcloud

    # if st.button('Analyse Data'):
    st.header('EDA ')
    tweets = pd.read_csv("./Data/reviews_clean.csv", encoding='latin-1')
    st.write(tweets)
    #Sentiment and reiviews count
    st.header("Various tweet counts")
    temp = tweets.groupby('sentiment').count().sort_values(by='reviews',ascending=False)['reviews'].reset_index()
   
    st.write(temp.style.background_gradient(cmap='Purples'))
    # Bar graph plot


    st.write("Bar graph plot ")
    figure = plt.figure(figsize=(12,6))
    counts = tweets["sentiment"].value_counts()
    plt.bar(counts.index, counts.values,color=['red','blue','green','pink','olive','cyan'])
    plt.show()
    st.pyplot(figure)
    

    st.header('Word Cloud')
    st.subheader(f'{sentiment_dict[0]} Tweets word cloud')
    figure = plt.figure()
    plt.imshow(show_string(tweets, 'int_category', 0), interpolation='bilinear')
    plt.show()
    st.pyplot(figure)
    
    st.subheader(f'{sentiment_dict[1]} Tweets word cloud')
    figure = plt.figure()
    plt.imshow(show_string(tweets, 'int_category', 1), interpolation='bilinear')
    plt.show()
    st.pyplot(figure)

    st.subheader(f'{sentiment_dict[2]} Tweets word cloud')
    figure = plt.figure()
    plt.imshow(show_string(tweets, 'int_category', 2), interpolation='bilinear')
    plt.show()
    st.pyplot(figure)

    st.subheader(f'{sentiment_dict[3]} Tweets word cloud')
    figure = plt.figure()
    plt.imshow(show_string(tweets, 'int_category', 3), interpolation='bilinear')
    plt.show()
    st.pyplot(figure)
   
    st.subheader(f'{sentiment_dict[4]} Tweets word cloud')
    figure = plt.figure()
    plt.imshow(show_string(tweets, 'int_category', 4), interpolation='bilinear')
    plt.show()
    st.pyplot(figure)
    
    st.subheader(f'{sentiment_dict[5]} Tweets word cloud')
    figure = plt.figure()
    plt.imshow(show_string(tweets, 'int_category', 5), interpolation='bilinear')
    plt.show()
    st.pyplot(figure)
   
    word_polarity_dict, feature_names = density_calculator(tweets, 'int_category')
    term_freq_df = pd.DataFrame([word_polarity_dict['Strongly negative'],word_polarity_dict['Strongly Positive'],word_polarity_dict['negative']
                     ,word_polarity_dict['positive'],word_polarity_dict['sarcasm'],
                     word_polarity_dict['neutral']], columns=feature_names).transpose()
    term_freq_df.columns = word_polarity_dict.keys()
    term_freq_df['total'] = term_freq_df['Strongly negative']+term_freq_df['Strongly Positive']+term_freq_df['negative']+term_freq_df['positive']+term_freq_df['sarcasm']+term_freq_df['neutral']
    #
    term_freq_df_sorted = term_freq_df.sort_values(by='total', ascending=False)
    st.write(term_freq_df_sorted)
    
    
    # st.header(f'Top 50 tokens in {sentiment_dict[0]} tweets')
    # reduced_matrix, feature_names = createDocumentMatrix(tweets, 10000)
    # neg_tf = calculate_term_frequency(tweets, reduced_matrix, 0)
    # pos_tf = calculate_term_frequency(tweets, reduced_matrix, 4)
    # neg = np.sum(neg_tf, axis=0)
    # pos = np.sum(pos_tf, axis=0)
    # term_freq_df2 = pd.DataFrame([neg, pos], columns=feature_names).transpose()
    # term_freq_df2.columns = ['negative', 'positive']
    # term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
    # st.write(term_freq_df2.sort_values(by='total', ascending=False).iloc[:10])
    for i in sentiment_dict.keys():
        positions = np.arange(50)
        fig = plt.figure()
        frequencies = term_freq_df.sort_values(by=sentiment_dict[i], ascending=False)[sentiment_dict[i]][:50]
        plt.bar(positions, frequencies)
        plt.xticks(positions, frequencies.index,
                   rotation='vertical',fontsize='small')
        plt.title(f'Top 50 words in {sentiment_dict[i]} tweets')
        plt.show()
        st.pyplot(fig)
    # fig = plt.figure()
    # frequencies = term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50]
    # plt.bar(positions, frequencies)
    # plt.xticks(positions, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,
    #            rotation='vertical', fontsize='small')
    # plt.title('Top 50 words in positive tweets')
    # plt.show()
    # st.pyplot(fig)