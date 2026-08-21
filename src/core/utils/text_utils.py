import re

quote_characters = {'"', "'"}

mid_sentence_words = [
    # Conjunctions
    "y", "e", "ni", "o", "u", "pero", "sino", "aunque", "como", "que", "donde", "cuando", "pues",
    "porque", "si",
    "and", "or", "nor", "because", "however",

    # Prepositions
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "del", "desde", "durante", "en", "entre", "hacia",
    "hasta", "mediante", "para", "por", "según", "sin", "so", "sobre", "tras", "versus", "vía",
    "about", "above", "across", "after", "against", "along", "among", "around", "as", "at", "before",
    "behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "despite",
    "down", "during", "except", "for", "from", "in", "inside", "into", "like", "near", "of", "off",
    "on", "onto", "opposite", "out", "outside", "over", "past", "through", "throughout", "to",
    "toward", "towards", "under", "underneath", "until", "up", "with", "within", "without",

    # Articles
    "el", "la", "los", "las", "su" ,
    "the", "her", "his"
]

def has_ended_abruptly(line_end, next_line_start):
    """Determines whether a line has finished without being complete or not.

    Applies heuristics to determine whether a line has finished without being complete or not based on
    the last word of this line and the first word of the next line, if any.

        :param line_end: Word that ends the line which has possibly been separated while incomplete.
        :type line_end: str

        :param next_line_start: First word in the next line, or None if there's no next line.
        :type next_line_start: str

        :return abrupt: True if the line has ended abruptly, False otherwise.
        :rtype: bool
    """
    if next_line_start is None:
        return False

    ends_with_comma = line_end.endswith(',')
    continues_with_lowercase = next_line_start.islower()
    continues_semantically = line_end in mid_sentence_words or next_line_start in mid_sentence_words

    return ends_with_comma or continues_with_lowercase or continues_semantically


def clean_quoting(words):
    """Cleans quoting spacing from a given word list.

    Appends quotes to the right word of the quote if the quoting is being opened,
    and to the left word if the quoting is being closed.

        :param words: A list of words to be cleaned.
        :type words: list[str]

        :return words: The list of cleaned words.
        :rtype: list[str]
    """
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



def clean_punctuation(text):
    """Cleans punctuation spacing from a given text.

    Applies regular expressions to remove extra spaces near different punctuation symbols.

        :param text: The text to be cleaned.
        :type text: str

        :return text: The text after cleaning.
        :rtype: str
    """
    # Remove space before punctuation for any of .,;:!?)]”’»›-
    text = re.sub(r"\s+([.,;:!?)\]”’»›-])", r"\1", text)
    # Remove space after opening punctuation for ({[“‘«‹-
    text = re.sub(r"([({[“‘«‹-])\s+", r"\1", text)
    return text