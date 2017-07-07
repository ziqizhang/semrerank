import re
import time

import gensim
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords

from util import utils

import logging


def train_unigram_model(corpus_gz, out_model):
    sentences = utils.GeniaCorpusLoader(corpus_gz)  # a memory-friendly iterator
    print(time.ctime())

    # sentences=LineSentence("/home/zqz/Work/termgraphex/data/dbpedia-splotlight/Vitamin_E.txt")
    model = gensim.models.Word2Vec(sentences,
                                   size=100, alpha=0.025, window=3, min_count=1, workers=8, sg=0)
    model.save(out_model)
    print(time.ctime())


def train_trigram_model(corpus_gz, out_model):
    sentences = utils.GeniaCorpusLoader(corpus_gz)  # a memory-friendly iterator
    print(time.ctime())

    phrases = Phrases(sentences, min_count=2, threshold=5)
    bigram = gensim.models.phrases.Phraser(phrases)
    trigram = Phrases(bigram[sentences], min_count=2, threshold=5)
    # sentences=LineSentence("/home/zqz/Work/termgraphex/data/dbpedia-splotlight/Vitamin_E.txt")
    model = gensim.models.Word2Vec(trigram[sentences],
                                   size=100, alpha=0.025, window=5, min_count=2, workers=8, sg=1)
    model.save(out_model)
    print(time.ctime())


def check_vocab_size(model_file):
    model = Word2Vec.load(model_file)
    model_keys = model.wv.vocab.keys()
    vocab_size=0
    vocab_size_before_filter=len(model_keys)
    stop = stopwords.words('english')
    for v in model_keys:
        norm = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                      v).strip()
        if (norm in stop or len(norm) < 2):
            continue
        vocab_size += 1
    print(vocab_size)
    print(vocab_size_before_filter)


check_vocab_size('/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model')
print('done')
#acl 21664, genia 84640, ttcm 98272, ttcw 121668

logging.basicConfig(filename='word2vec.log', level=logging.INFO)
logging.info("Starting...")
# IN_CORPUS="/home/zqz/Work/data/semrerank/corpus/genia.tar.gz"
# IN_CORPUS="/home/zqz/Work/data/semrerank/corpus/genia+dbp.tar.gz"
#IN_CORPUS = "/home/zqz/Work/data/semrerank/corpus/ttc_mobile.tar.gz"
# IN_CORPUS="/home/zqz/Work/data/semrerank/corpus/ttc_wind.tar.gz"
IN_CORPUS="/home/zqz/Work/data/semrerank/corpus/aclrd-ver2.tar.gz"
# IN_CORPUS="/home/zqz/Work/data/semrerank/corpus/scienceie_test.tar.gz"
# OUT_MODEL="/home/zqz/Work/data/semrerank/embeddings/em_gdbp-uni-sg-100-w3-m1.model"
#OUT_MODEL = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# OUT_MODEL="/home/zqz/Work/data/semrerank/embeddings/em_ttcw-uni-sg-100-w3-m1.model"
OUT_MODEL="/home/zqz/Work/data/semrerank/embeddings/em_aclv2_test-uni-sg-100-w3-m1.model"
# OUT_MODEL="/home/zqz/Work/data/semrerank/embeddings/em_science-uni-sg-100-w3-m2.model"
# train_trigram_model(IN_CORPUS, OUT_MODEL)
train_unigram_model(IN_CORPUS, OUT_MODEL)
