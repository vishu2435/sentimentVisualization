import tweepy
import clean_data_funcs

# class StreamListener(tweepy.StreamListener):
#     def on_status(self, status):
#         print(status.text)
#
#     def on_error(self, status_code):
#         print("Encountered streaming error ( ", status_code," ) ")
#


# api = tweepy.API(auth)


# public_tweets = api.home_timeline()

# initialize stream
# streamListener = StreamListener()
# stream = tweepy.Stream(auth=api.auth, listener=streamListener, tweet_mode='extended')
#
# tags = ["covid-19"]
# stream.filter(track=tags)

# print(api.rate_limit_status())

def extract_tweet(searchArgument):
    # complete authorization and initialize Api endpoint
    auth = tweepy.OAuthHandler('g5bDo7rt1IFecdoM0et2wgS3e', 'fhsfGVq5WVTmDA7CTtAWBDzS6oxCm4qXLeglPXG3VuAGbS9AkZ')
    auth.set_access_token('1305347807478333442-gA9frH26CCtpBxcqcweWzAki4t8zI9',
                          'vxtdJlAbQbsFNEtmjW4iK9n3ysw2KTCc1rf50CpUKOL16')

    # Construct the API instance
    api = tweepy.API(auth)
    tweets = []
    for tweet in tweepy.Cursor(api.search, q=f'{searchArgument} -filter:retweets', count=200, lang='en', rpp=100).items(200):
        tweets.append(clean_data_funcs.clean_tweet(tweet.text))
    return tweets
