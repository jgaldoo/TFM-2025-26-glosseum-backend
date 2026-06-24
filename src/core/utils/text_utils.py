import re

quote_characters = {'"', "'"}

"""Cleans quoting spacing from a given word list.

Appends quotes to the right word of the quote if the quoting is being opened,
and to the left word if the quoting is being closed. 

Args:
    words: A list of words to be cleaned.

Returns:
    words: The list of cleaned words.
"""
def clean_quoting(words):
    processed_words = []
    opened_quotes = {q: False for q in quote_characters}
    i = 0

    while i < len(words):
        word = words[i]

        # Word is an opening quote and the pair isn't opened
        if word in quote_characters and not opened_quotes[word]:
            if i + 1 < len(words):
                processed_words.append(word + words[i + 1])
                opened_quotes[word] = True
                i += 1
            else:
                processed_words.append(word)

        # Word is a closing quote and the pair is opened
        elif word in quote_characters and opened_quotes[word]:
            processed_words[-1] += word
            opened_quotes[word] = False

        # Word starts with a quote
        elif word[0] in quote_characters:
            processed_words.append(word)
            opened_quotes[word[0]] = True

        # Word ends with a quote
        elif word[-1] in quote_characters:
            processed_words.append(word)
            opened_quotes[word[-1]] = False

        # Normal word
        else:
            processed_words.append(word)

        i += 1

    return processed_words


"""Cleans punctuation spacing from a given text.

Applies regular expressions to remove extra spaces near different punctuation symbols.

Args:
    text: The text to be cleaned.

Returns:
    text: The text after cleaning.
"""
def clean_punctuation(text):
    # Remove space before punctuation for any of .,;:!?)]”’»›-
    text = re.sub(r"\s+([.,;:!?)\]”’»›-])", r"\1", text)
    # Remove space after opening punctuation for ({[“‘«‹-
    text = re.sub(r"([({[“‘«‹-])\s+", r"\1", text)
    return text