import logging
import sys
from gensim import corpora,models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load vocabulary and convert to id2word
f = open('../vocab/vocabulary.txt')
wordlist=f.read().rstrip('\n').split('\n')
f.close()
#id2word=dict(zip(range(0,len(wordlist)), wordlist))

# creat dictionary
#dictionary=corpora.Dictionary.from_corpus(corpus, id2word)

# load stop words
#f = open('stoplist2.txt')
#stoplist=f.read().rstrip('\n').split('\n')
#f.close()

# get Word2Vec
#model = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#model = models.Word2Vec.load_word2vec_format('glove_6B_300d.bin', binary=True)
model = models.Word2Vec.load_word2vec_format('../python/' + sys.argv[1], binary=True);
dimension=len(model['the']);

filename='../vocab/' + sys.argv[2]
f=open(filename,'w+')
nolookups=0
for i in range(0,len(wordlist)):
    try:
        wordvec = model[wordlist[i]]
        for numbers in wordvec:
            f.write("%f " % numbers)
    except KeyError:
        #print "Oops! one word doesnot look up its feature vector: " + wordlist[i]
        for j in range(0,dimension):
            f.write("0 ")
        nolookups=nolookups + 1
    f.write("\n")
f.close()
print str(nolookups) + ' words removed because of no-shown'
print 'save to ' + filename
