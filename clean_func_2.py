import re
from emoji import UNICODE_EMOJI


def is_emoji(s):
    count = 0
    for emoji in UNICODE_EMOJI:
        count += s.count(emoji)
        if count > 1:
            return False
    return bool(count)

def clean_line(line):
    line = ' '.join(re.sub("(@[A-Za-z0-9_]+)", "", line).split())

    wordlist = line.split()
    line = ' '.join(re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", line).split())


