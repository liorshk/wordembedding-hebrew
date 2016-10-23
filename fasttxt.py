import fasttext
import time

def train(inp = "wiki.he.text",out_model = "wiki.he.fasttext.model",
          alg = "CBOW"):

    start = time.time()

    if alg == "skipgram":
        # Skipgram model
        model = fasttext.skipgram(inp, out_model)
        print(model.words) # list of words in dictionary
    else:
        # CBOW model
        model = fasttext.cbow(inp, out_model)
        print(model.words) # list of words in dictionary

    print(time.time()-start)
          
    model.save(out_model)



def getModel(model = "wiki.he.fasttext.model.bin"):

    model = fasttext.load_model(model)

    return model
