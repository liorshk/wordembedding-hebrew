import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time

def train(inp = "wiki.he.text",out_model = "wiki.he.word2vec.model"):

    start = time.time()

    model = Word2Vec(LineSentence(inp), sg = 0, # 0=CBOW , 1= SkipGram
                     size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    print(time.time()-start)

    model.save(out_model)


def test(model = "wiki.he.word2vec.model"):

    model = Word2Vec.load("wiki.he.word2vec.model")
 
    print(model.most_similar(positive=["מלכה","גבר"],negative=["מלך"])
