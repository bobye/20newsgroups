import logging
import sys
import itertools
from gensim import corpora,models


def generate_tfidf(version, isSemiSup=False):
    "take in .mm format data, process with tfidf, and output .data"
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # load training data in mm format
    trainmm = corpora.mmcorpus.MmCorpus('../mm/train_preproc' + str(version) + '.mm')
    testmm = corpora.mmcorpus.MmCorpus('../mm/test_preproc' + str(version) + '.mm')
    corpus=itertools.chain(trainmm, testmm) if isSemiSup else trainmm

    tfidf = models.TfidfModel(corpus)    
    train_tfidf = tfidf[trainmm]
    test_tfidf = tfidf[testmm]    
    tag=(str(version) + 's') if isSemiSup else str(version)

    f=open('../data/train_tfidf' + tag + '.data','w+')
    for i, doc in enumerate(train_tfidf):
        for word in doc:
            f.write("%d %d %f\n" % (i+1, word[0]+1, word[1]))
    f.close()
    f=open('../data/test_tfidf' + tag + '.data','w+')
    for i, doc in enumerate(test_tfidf):
        for word in doc:
            f.write("%d %d %f\n" % (i+1, word[0]+1, word[1]))
    f.close()


def main(argv):
    generate_tfidf(int(argv[0]))
    generate_tfidf(int(argv[0]), True)

if __name__ == "__main__":
    main(sys.argv[1:])
