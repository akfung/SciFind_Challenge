# SciFind_Challenge
This project attempts to visualize the spread of experience between a group of scientists as measured by the similarity of their most recent work, while mapping the relationships between each scientist. Web scraping and minor data processing is conducted with scrape.ipynb, while the NLP and network analysis is conducted with doc_network_analysis.ipynb. These have been collected into pubmed_scrape.py for simpler reusability of the scripts.

Given a collaborative paper with a list of authors on pubmed, I used splinter and beautiful soup collect and concatenate abstracts of each scientist's 5 most recent published papers. 

The abstracts for each scientist are processed using nltk to remove stopwords and punctuation. Each scientist's concatenated abstracts are compared for soft cosine similarity to each other scientist's abstracts using gensim and the word2vec model (I used this as a guide for using the library https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb). The resulting network is graphed using NetworkX. Edges are drawn between authors that have similarity scores higher than the mean similarity score, which allows for better visualization of clustering.

![alt text](https://github.com/akfung/SciFind_Challenge/blob/master/img/Authors\ of\ Novavax\ Paper.png?raw=true)

Below are examples of this script generating network graphs using the authors of a recently published Novavax paper on their successful phase 1 and phase 2 clinical trials for a Coronavirus vaccine, the authors of a review on mosquito vectors, and the authors of a microplastics paper. The Coronavirus vaccine paper (as a highly focused project) seems to have a more narrow spread between the authors compared to the broader mosquito and microplastics review papers. The review papers seem to show clustering between authors of similar specialities, which may be useful for identifying authors of a similar skillset.
