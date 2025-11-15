#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')   

text = "John enjoys playing football while Mary loves reading books in the library."

# 1. Tokenize
tokens = word_tokenize(text)

# 2. Remove stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]

# Helper: convert POS tags to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# 3. POS tagging
pos_tags = pos_tag(filtered_tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_words = []

# 4. Keep only verbs and nouns + lemmatize
for word, tag in pos_tags:
    wn_tag = get_wordnet_pos(tag)
    if wn_tag in (wordnet.VERB, wordnet.NOUN):
        lemma = lemmatizer.lemmatize(word, wn_tag)
        lemmatized_words.append(lemma)

print(lemmatized_words)


# In[ ]:




