import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import accuracyMeasure
import extractTweet
import pickle
import joblib
from wordcloud import WordCloud
from nltk.tokenize import TweetTokenizer


sentiment_dict = {
    0:'Strongly negative',
    1:'Strongly Positive',
    2:'negative',
    3:'positive',
    4:'sarcasm',
    5:'neutral'
}
def app():
    st.write("# Twitter Sentiment Analysis")

    stModel = None

    @st.cache
    def load_training_dataset(vectorizer=None,tokenizer=None,ngrams=(1,1),stop_words=None,features=10000):
        # st.write("****************************************")
        # st.write("Parameters are ")
        # st.write("Features ",features)
        # st.write("Tokenizer ",tokenizer)
        
        dataset = pd.read_csv("./Data/reviews_clean.csv", usecols=[1, 3], encoding='latin-1')
        x = dataset.reviews
        y = dataset.int_category
        # tk = tokenizer()
        # print()
        if stop_words:
            stop_words='english'
        else:
            stop_words=None
        vectorizer.set_params(tokenizer=tokenizer.tokenize,stop_words=stop_words,ngram_range=ngrams,max_features=features)
        vectorizer.fit(x)
        vector = vectorizer.transform(x) 

        x_train, x_validation_and_test, y_train, y_validation_test = train_test_split(vector, y, test_size=0.3,
                                                                                      random_state=2000)

        x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_test,
                                                                      test_size=0.5, random_state=2000)
        return x_train, y_train, x_test, y_test, x_validation, y_validation

    options = {
        'ngram_dict': {
            'UniGram': (1, 1),
            'BiGram': (2, 2),
            'TriGram': (3, 3)
        },
        'classifier': {
            'Linear SVC': LinearSVC(max_iter=200),
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes':MultinomialNB(alpha=1.0,fit_prior=False),
            'AdaBoost':AdaBoostClassifier()
        },
        'Tokenizer':{
            'Tweet Tokenizer':TweetTokenizer()
        },
        'vectorizer': {
            'Count Vectorizer': CountVectorizer(),
            'TfidfVectorizer': TfidfVectorizer()
        }

    }
    tokenizer = st.sidebar.selectbox("Select Tokenizer",('Tweet Tokenizer',))
    vectorizer = st.sidebar.selectbox("Select Vectorizer", ('TfidfVectorizer', 'Count Vectorizer'))
    classifier = st.sidebar.selectbox("Select Classifier", ('Linear SVC', 'Logistic Regression','Naive Bayes','AdaBoost'))
    ngram_range = st.sidebar.selectbox("Select NGram", ('UniGram', 'BiGram', 'TriGram'))
    use_stop_words = st.sidebar.checkbox("Use Stop Words")
    features = st.sidebar.slider("Number of Features", min_value=10000, max_value=100000, step=10000)
    st.sidebar.write(f"Number of features are {features}")
    
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = load_training_dataset(vectorizer = options['vectorizer'][vectorizer],
                                                                                        tokenizer = options['Tokenizer'][tokenizer],
                                                                                        stop_words = use_stop_words,
                                                                                        ngrams=options['ngram_dict'][ngram_range],features=int(features))
    st.write(f'Shape of training input X is {X_train.shape}')
    st.write(f'Shape of training output variable y is {Y_train.shape}')

    if st.sidebar.button("Train Model"):
        # if use_stop_words:
        #     stModel = accuracyMeasure.nfeature_accuracy_checker(
        #         st=st,
        #         vectorizer=options['vectorizer'][vectorizer](options['Tokenizer'][tokenizer]),
        #         n_features=features,
        #         stop_words=None,
        #         ngram_range=options['ngram_dict'][ngram_range],
        #         classifier=options['classifier'][classifier],
        #         x_train=X_train,
        #         y_train=Y_train,
        #         x_test=X_validation,
        #         y_test=Y_validation
        #     )
        # else:
        #     # stModel = accuracyMeasure.nfeature_accuracy_checker(
            #     st=st,
            #     vectorizer=options['vectorizer'][vectorizer](options['Tokenizer'][tokenizer]),
            #     n_features=features,
            #     stop_words='english',
            #     ngram_range=options['ngram_dict'][ngram_range],
            #     classifier=options['classifier'][classifier],
            #     x_train=X_train,
            #     y_train=Y_train,
            #     x_test=X_validation,
            #     y_test=Y_validation
            # )
        stModel = accuracyMeasure.run_classification(
            st,
            X_train,
            Y_train,
            X_test,
            Y_test,
            options['classifier'][classifier]
        )
