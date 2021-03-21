import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import clean_data_funcs
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from wordcloud import WordCloud



# pre process data


@st.cache(persist=True)
def pre_processing(text_array):
    clean_tweets = []
    chunk_range = [0, 400000, 800000, 1200000, 1600000]

    for i in range(chunk_range[0], chunk_range[3]):
        my_bar.progress(int((i/chunk_range[1])*100))
        clean_tweets.append(clean_data_funcs.clean_tweet(text_array[i]))

    return clean_tweets


@st.cache(persist=True)
def load_clean_data():
    cols = ['text', 'target']
    clean_train = pd.read_csv('./Data/clean_Tweets_1600000Tweet.csv', header=None, names=cols, encoding='latin-1')
    clean_train.index = np.arange(0,len(clean_train['target']))
    return clean_train


@st.cache(persist=True)
def load_data():
    # Declare columns
    cols = ['sentiment', 'text']
    # Initialize training dataset
    df_train = pd.read_csv('./Data/training.1600000.processed.noemoticon.csv', header=None, usecols=[0, 5], names=cols,
                           encoding='latin-1')
    # drop rows with neutral sentiment
    df_train.drop(df_train[df_train.sentiment == 2].index, inplace=True)
    # drop rows with retweet text
    df_train.drop(df_train[df_train.text.str.contains(' RT ')].index, inplace=True)
    return df_train


@st.cache(persist=True)
def show_string(data, attribute, target):
    tweets = data[data[attribute] == target]
    # st.write(data[[0]])

    # st.write(data[data[attribute] == 0])
    string = []
    for t in tweets.text:
        string.append(t)
    # st.write(string)
    string = pd.Series(string).str.cat(sep='')
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(string)

    return wordcloud


@st.cache(persist=True)
def density_calculator(data, attribute):
    # declaring cvec object
    cvec = CountVectorizer()
    # fitting data into cvec
    cvec.fit(data.text)
    # negative Data matrix

    @st.cache(persist=True)
    def inner_helper(target):
        doc_matrix = cvec.transform(data[data[attribute] == target].text)
        doc_tf = np.sum(doc_matrix, axis=0)
        return np.squeeze(np.asarray(doc_tf))
    neg = inner_helper(0)
    pos = inner_helper(4)
    return neg, pos, cvec.get_feature_names()


st.title('Twitter Sentiment analysis')
st.subheader('Raw chart Data')
train_data = load_data()
st.write(train_data.head(4000))

st.subheader('Clean Data')
my_bar = st.progress(0)
clean_df = load_clean_data()


st.write(clean_df.head(4000))
st.header('Word Cloud')
st.subheader('Negative Words')

fig = plt.figure()
plt.imshow(show_string(clean_df, 'target', 0), interpolation='bilinear')
plt.axis("off")
plt.show()


st.pyplot(fig)


st.subheader('Positive Words')


def word_cloud_generator():
    fig = plt.figure()
    plt.imshow(show_string(clean_df, 'target', 0), interpolation='bilinear')
    # plt.axis("off")
    plt.show()
    return fig


@st.cache
def pyplot_generator(plot):
    figure = plt.figure()
    st.write(plot[1]['x'], plot[1]['y'])
    plt.plot(plot[1]['x'], plot[1]['y'], color='r', linestyle='--', linewidth=2)
    plt.show()
    return figure


st.pyplot(word_cloud_generator())
#
st.subheader('Negative Words')

negative, positive, feature_names = density_calculator(clean_df, 'target')

term_freq_df = pd.DataFrame([negative, positive], columns=feature_names).transpose()
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
#
term_freq_df_sorted = term_freq_df.sort_values(by='total', ascending=False)
st.write(term_freq_df_sorted.head())

st.subheader('Number of words')
st.write(str(negative.shape[0]))

st.subheader('Zeph\'s law')
figure = plt.figure()
plt.plot(np.arange(500), [term_freq_df_sorted['total'][0]/(i+1) for i in np.arange(500)], color='r', linestyle='--', linewidth=2)
plt.bar(np.arange(500), term_freq_df_sorted['total'][:500])
plt.show()

st.pyplot(figure)

def logloggraph():
    counts = term_freq_df_sorted.total
    tokens = term_freq_df_sorted.index
    ranks = np.arange(1, len(counts)+1)
    indices = np.argsort(-counts)
    frequencies = counts[indices]
    fi = plt.figure()
    plt.ylim(1, 10**4)
    plt.xlim(1, 10**4)
    plt.loglog(ranks, frequencies, marker=".")
    plt.plot([1, frequencies[0]], [frequencies[0], 1], color='r')
    plt.title("Zipf plot for tweets tokens")
    plt.xlabel("Frequency rank of token")
    plt.ylabel("Absolute frequency of token")
    st.pyplot(fi)


logloggraph()




