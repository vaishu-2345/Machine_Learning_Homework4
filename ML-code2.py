#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk import pos_tag, word_tokenize, ne_chunk

# Download required NLTK datasets (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "Chris met Alex at Apple headquarters in California. He told him about the new iPhone launch."

# Tokenize + POS Tag
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# 1. Named Entity Recognition (NER)
ner_tree = ne_chunk(pos_tags)

print("Named Entities:")
for subtree in ner_tree:
    if hasattr(subtree, 'label'):   # means it's an entity
        entity = " ".join([token for token, pos in subtree])
        print(entity, "->", subtree.label())

# 2. Pronoun Ambiguity Detection
pronouns = {"he", "she", "they", "him", "her", "them"}

found_pronouns = [t.lower() for t in tokens if t.lower() in pronouns]

if found_pronouns:
    print("\nWarning: Possible pronoun ambiguity detected!")


# In[ ]:




