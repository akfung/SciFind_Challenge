"""
Part one of this script runs through all the authors on a pubmed paper using its url. 
Require chromedriver for use
It collects the abstracts of the 5 most recent papers of each author credited in that paper.
"""

# ask user for input of pubmet paper url
article_url = input("Enter pubmed paper URL\n")
# ask user what to title the output graph
fig_title = "Authors of " + input("What is this paper about?\n") + " Paper"

from splinter import Browser
from bs4 import BeautifulSoup as bs
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import networkx as nx
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities
from gensim.matutils import softcossim
import gensim.downloader as api
from gensim.utils import simple_preprocess


# Chromedriver setup


def init_browser():
    executable_path = {'executable_path': '/usr/local/bin/chromedriver'}
    return Browser('chrome', **executable_path, headless=True)


base_url = "https://pubmed.ncbi.nlm.nih.gov"  # pubmed base url
novavax_url = "https://pubmed.ncbi.nlm.nih.gov/32877576/"  # url for the novavax paper

# visit the web page
browser = init_browser()

# get list of authors and hrefs to their searches
browser.visit(article_url)
soup = bs(browser.html, "html.parser")
authors = []
for child in soup.find("div", class_="authors").find_all("a", class_='full-name'):
    author_name = child.get_text()
    author_href = child['href']
    authors.append({
        "name": author_name,
        "href": author_href
    })

for author in authors:
    print(f'Collecting abstracts for {author["name"]}')

    # go to date sorted results for author
    browser.visit(base_url + author['href'] + "&sort=date")
    soup = bs(browser.html, "html.parser")

    article_hrefs = []  # list to hold hrefs for 5 most recent articles by scientist
    for article in soup.find_all("a", class_="docsum-title", limit=5):
        article_hrefs.append(article['href'])
    abstract_string = ''

    # go to each article by href and add the abstract text to a string if it exists
    for article_href in article_hrefs:
        browser.visit(base_url + article_href)
        soup = bs(browser.html, "html.parser")
        abstract = soup.find(id="enc-abstract")
        if abstract:
            abstract_string += abstract.get_text()

    # add the concatenated string to authors dict
    author["abstracts"] = abstract_string
browser.quit()

# create df with data and cleaning
df = pd.DataFrame(authors)


def cleanHtml(sentence):  # regex to remove end of sentence punctuation
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence):  # function to clean the abstracts of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


# stem data to combine words with similar meanings w/ snowball stemmer
stemmer = SnowballStemmer("english")


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


# update list of stopwords from nltk
stop_words = set(stopwords.words('english'))
# remove some common terms used in pubmed abstract
stop_words.update(['background', 'methods', 'results', 'conclusions'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
# function to remove stop words


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


# apply cleaning functions to abstract text
df['abstracts'] = df['abstracts'].str.lower()
df['abstracts'] = df['abstracts'].apply(cleanHtml)
df['abstracts'] = df['abstracts'].apply(cleanPunc)
# optionally use stemming
# df['abstracts'] = df['abstracts'].apply(stemming)
df['abstracts'] = df['abstracts'].apply(removeStopWords)

'''
This part of the script uses gensim for calculating soft cosine similarity between authors by their recent abstract text.
Graphing a network of their relationships is performed by networkx and output to a png. Edges are only drawn between nodes with
a similarity higher than the mean similarity of all authors.
'''

w2v_model = api.load("glove-wiki-gigaword-50")  # use word2vec model

text_corpus = list(df['abstracts'])
author_list = list(df['name'])

# 2D matrix of whitespace separated words for each abstract
texts = [[word for word in document.split()] for document in text_corpus]

dictionary = corpora.Dictionary(texts)  # create tokens for each unique word
similarity_index = models.WordEmbeddingSimilarityIndex(
    w2v_model)  # similarity index using word 2 vec model
similarity_matrix = similarities.SparseTermSimilarityMatrix(
    similarity_index, dictionary)  # similarity matrix from dictionary

# tokenize each abstract and store all results in bow_sentences
bow_sentences = []
for sentence in texts:
    bow_sentences.append(dictionary.doc2bow(sentence))

# calculate similarities between all authors
import numpy as np
num_authors = len(texts)
sims = []
for i in range(0, len(texts)):
    row = []
    for j in range(0, len(texts)):
        similarity = similarity_matrix.inner_product(
            bow_sentences[i], bow_sentences[j], normalized=True)
        row.append(similarity)
    sims.append(row)

# write similarities matrix to a df and csv, set index to author names
df = pd.DataFrame(data=sims, columns=author_list)
df['Author'] = author_list
df = df.set_index("Author")
df.to_csv(f'{fig_title}_similarities_matrix.csv)

g = nx.Graph()
mean_value = df.mean().mean()  # calculate mean similarity value of entire matrix
for i in range(0, num_authors):
    author_1 = author_list[i]

    for j in range(0, num_authors):
        author_2 = author_list[j]
        relation = df.values[i][j]
        # draw edge only if the similarity value is over the mean similarity value
        if relation > mean_value:
            g.add_edge(author_1, author_2, weight=relation)

# draw and output the network graph
nx.draw_networkx(g)
plt.title(fig_title)
plt.margins(0.3, 0.3)
fig = plt.gcf()
fig.savefig(f'{fig_title}.png', dpi=300, bbox_inches='tight')
