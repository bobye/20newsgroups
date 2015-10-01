from gensim import corpora,models
import logging
from numpy import savetxt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus=corpora.MmCorpus('data.mm')
id2word=[line.rstrip() for line in open('../vocab/vocabulary.txt')]
id2word=dict([(i,id2word[i]) for i in range(0, len(id2word))])
lda=models.LdaModel(corpus, id2word=id2word, num_topics=120, alpha=0.1, eta=0.01, passes=100)
infer=lda.inference(corpus)
savetxt('corpus.topics120', infer[0])
