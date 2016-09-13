import fasttext
import time

def train(inp = "wiki.he.text",out_model = "wiki.he.fasttext.model",
          alg = "cbow"):

    print("Training Fasttext")

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

    print("Saving model")
    model.save(out_model)


def test(model = "wiki.he.fasttext.model"):

    model = fasttext.load_model(model)

    print model.words
    # todo



train()
