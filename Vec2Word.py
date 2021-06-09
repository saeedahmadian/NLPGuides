from scipy.spatial import distance
import numpy as np
import spacy
import en_core_web_md

# Load the spacy vocabulary
nlp = en_core_web_md.load()

def Vect2Word(nlp,input_words):
    output_words=[]
    ids = [x for x in nlp.vocab.vectors.keys()]
    vectors = [nlp.vocab.vectors[x] for x in ids]
    vectors = np.array(vectors)
    for input_word in input_words:
        p = np.array([nlp.vocab[input_word].vector])
        closest_index = distance.cdist(p, vectors).argmin()
        word_id = ids[closest_index]
        output_words.append(nlp.vocab[word_id].text)
    return output_words


a=1