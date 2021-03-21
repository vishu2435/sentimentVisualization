import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import extractTweet
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def app():
    st.subheader('Tweet hashtag')
    inputData = st.text_input('Enter Tweet hashtag')

    def density_calculator(data, attribute):
        # declaring cvec object
        cvec = CountVectorizer()
        # fitting data into cvec
        cvec.fit(data.text)

        # negative Data matrix

        # @st.cache(persist=True)
        def inner_helper(target):
            doc_matrix = cvec.transform(data[data[attribute] == target].text)
            doc_tf = np.sum(doc_matrix, axis=0)
            return np.squeeze(np.asarray(doc_tf))

        neg = inner_helper(0)
        pos = inner_helper(4)
        return neg, pos, cvec.get_feature_names()

    def createDocumentMatrix(data, max_features):
        cvec = CountVectorizer(stop_words='english',
                               max_features=max_features)  # Removing stop words this time and limiting studied words to 10k
        cvec.fit(data.text)
        document_matrix = cvec.transform(data.text)
        return document_matrix, cvec.get_feature_names()

    def calculate_term_frequency(data, document_matrix, target):
        batches = np.linspace(0, len(data[data['target'] == target]), 10).astype('int')
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
        for t in innertweets.text:
            string.append(t)
        # st.write(string)
        string = pd.Series(string).str.cat(sep='')
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(string)

        return wordcloud

    if st.button('Analyse Data'):
        loaded_model = joblib.load('trainedModel.pkl')
        st.write('Loaded Model', loaded_model)
        if not loaded_model:
            st.error('Please Train model first')
        elif not inputData:
            st.error('Please Enter Input field first')
        else:
            st.write('Extracting and cleaning Data .... ')
            tweets = extractTweet.extract_tweet(inputData)
            st.write(pd.DataFrame([tweets]).transpose())
            st.write('Now feeding tweets to model')
            prediction = loaded_model.predict(tweets)
            st.write('Table after prediction is ')
            tweets_dataframe = pd.DataFrame([tweets, prediction]).transpose()
            tweets_dataframe.columns = ['text', 'target']
            st.write(tweets_dataframe)
            st.write(tweets_dataframe.shape)
            st.header('Word Cloud')

            st.subheader('Negative Tweets word cloud')
            figure = plt.figure()
            plt.imshow(show_string(tweets_dataframe, 'target', 0), interpolation='bilinear')
            plt.show()
            st.pyplot(figure)

            st.subheader('Positive Tweets word cloud')
            figure = plt.figure()
            plt.imshow(show_string(tweets_dataframe, 'target', 4), interpolation='bilinear')
            plt.show()
            st.pyplot(figure)

            negative, positive, feature_names = density_calculator(tweets_dataframe, 'target')
            term_freq_df = pd.DataFrame([negative, positive], columns=feature_names).transpose()
            term_freq_df.columns = ['negative', 'positive']
            term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
            #
            term_freq_df_sorted = term_freq_df.sort_values(by='total', ascending=False)
            st.write(term_freq_df_sorted)

            st.subheader('Zeph\'s law')
            figure = plt.figure()
            plt.plot(np.arange(500), [term_freq_df_sorted['total'][0] / (i + 1) for i in np.arange(500)], color='r',
                     linestyle='--', linewidth=2)
            plt.bar(np.arange(200), term_freq_df_sorted['total'][:200])
            plt.show()

            st.pyplot(figure)

            st.header('Most Frequent positive and negative words')
            reduced_matrix, feature_names = createDocumentMatrix(tweets_dataframe, 10000)
            neg_tf = calculate_term_frequency(tweets_dataframe, reduced_matrix, 0)
            pos_tf = calculate_term_frequency(tweets_dataframe, reduced_matrix, 4)
            neg = np.sum(neg_tf, axis=0)
            pos = np.sum(pos_tf, axis=0)
            term_freq_df2 = pd.DataFrame([neg, pos], columns=feature_names).transpose()
            term_freq_df2.columns = ['negative', 'positive']

            term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
            st.write(term_freq_df2.sort_values(by='total', ascending=False).iloc[:10])

            positions = np.arange(50)
            fig = plt.figure()
            frequencies = term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50]
            plt.bar(positions, frequencies)
            plt.xticks(positions, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50].index,
                       rotation='vertical',fontsize='small')
            plt.title('Top 50 words in negative tweets')
            plt.show()
            st.pyplot(fig)

            fig = plt.figure()
            frequencies = term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50]
            plt.bar(positions, frequencies)
            plt.xticks(positions, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,
                       rotation='vertical', fontsize='small')

            plt.title('Top 50 words in positive tweets')
            plt.show()
            st.pyplot(fig)

