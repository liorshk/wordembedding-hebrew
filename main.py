import word2vec
import fasttxt
import numpy as np
from gensim.matutils import unitvec

TRAIN = True

def test(model,positive,negative,test_words):

    mean = []
    for pos_word in positive:
        mean.append(1.0 * np.array(model[pos_word]))

    for neg_word in negative:
        mean.append(-1.0 * np.array(model[neg_word]))

    # compute the weighted average of all words
    mean = unitvec(np.array(mean).mean(axis=0))

    for word in test_words:
        test_word = unitvec(np.array(model[word]))

        # Cosine Similarity
        print(np.dot(test_word, mean))

if TRAIN:
    print("Training Word2vec")
    word2vec.train()

    print("Training fasttext")
    fasttxt.train()


positive_words = ["פריז","גרמניה"]#["מלכה","גבר"]

negative_words = ["צרפת"]#["מלך"]

test_words = ["ברלין"]#["אישה"]


# Test Word2vec
print("Testing Word2vec")
test(word2vec.getModel(),positive_words,negative_words,test_words)

# Test Fasttext
print("Testing Fasttext")
test(fasttxt.getModel(),positive_words,negative_words,test_words)