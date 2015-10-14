import pickle

def pickler(object, file_name):
    with open(file_name, "w") as f:
        pickle.dump(object, f)


def depickler(file_name):
    with open(file_name, "r") as f:
        return pickle.load(f)


filename = 'data/words_embeddings_32.pkl'

embeddings = Embedding.load("/home/rmyeid/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
print(embeddings)