import json
import collections
import string
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log

stemmer = PorterStemmer()


with open('training_data.json') as f:
    reviews = [json.loads(line) for line in f]

reviewtext=[]
reviewlabel=[]
tokens=[]

for i in range(len(reviews)):
    text=reviews[i]['text'].lower()
    text = re.sub(r'[^\w\s]','',text)
    #text=stemmer.stem(text)
    #text.translate(None, string.punctuation)
    reviewtext.append(text)
    tokens=tokens + text.split()
    reviewlabel.append(reviews[i]['label'])

#print reviewtext
print("Total No of documents", len(reviewtext))
print("Total no of tokens obtained are:",len(tokens))

token_counts = collections.Counter(token)
for tok, count in token_counts.most_common(20):
    print '%s : %i' % (tok, count)

#Verifying Zipf's Law
token_rank = range(1, 21)
log_rank = [log(x,10) for x in token_rank]
term_counts = [i[1] for i in token_counts.most_common(20)]
log_term_counts = [log(y,10) for y in term_counts]
#print(rank)
#print(term_counts)

plt.plot(log_rank, log_term_counts)
plt.xlabel('log-base10 rank of top 20 token')
plt.ylabel('log-base10 freq of top 20 token')
plt.title('Zipfs law')
plt.show()
