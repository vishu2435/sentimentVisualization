from bs4 import BeautifulSoup
import re

url = re.compile(r"(?:(http[s]?://\S+)|((//)?(\w+\.)?\w+\.\w+/\S+))")
usr_mention = re.compile(r"(?:(?<!\w)@\w+\b)")
number = re.compile(r"(?:\b\d+\b)")
repeated_char = '([a-zA-Z])\\1+'
length_repeated_char = '\\1\\1'
rt_mention = re.compile(r'RT @')
rt_small_mention = re.compile(r'rt @')

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :'),:D, : D, =)
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))|:\s?D | =\)', '', tweet)
    # Sad -- :-(, : (, :(, ):, )-: , :p
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)|:p', '', tweet)
    return tweet


def clean_tweet(raw):
    new_row = BeautifulSoup(raw, 'html.parser').get_text()
    # Change all text to lower case
    new_row = new_row.lower()

    # Replaces any url with ''
    new_row = re.sub(url, '', new_row)
    new_row = re.sub(rt_mention, '@', new_row)
    new_row = re.sub(rt_small_mention, '@', new_row)
    # Replace any username with ''
    new_row = re.sub(usr_mention, '', new_row)

    # Strip repeated chars
    new_row = re.sub(repeated_char, length_repeated_char, new_row)

    # replaces hashtag with
    new_row = re.sub(r'#(/S+)', r'\1', new_row)

    # Remove numbers
    new_row = re.sub(number, '', new_row)

    # decode text with 'utf-8-sig'
    try:
        temp_row = new_row.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        temp_row = new_row

    new_row = handle_emojis(temp_row)
    new_row = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", new_row)
    return new_row
