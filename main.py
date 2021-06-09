import torch
import tensorflow as tf
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import spacy
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import en_core_web_md
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from torchtext.data import Field, ReversibleField
from torchtext.data import TabularDataset, BucketIterator
import pandas as pd
nlp= en_core_web_md.load()

df_train=pd.read_csv('train.csv',header='infer')
import re
loss=torch.nn.CosineSimilarity()
loss(torch.tensor([1,2,4],torch.tensor([8,9,0])))
nltk.download('stopwords')
stop_words=nltk.corpus.stopwords.words('english')
stop_words.remove('not')
def CustomTokenizerSpacy(sentence):
    sentence= re.sub('n\'t',r' not',sentence)
    sentence=re.sub(r'[^a-zA-Z0-9!?,\.]',' ',sentence)
    sentence= re.sub(r'([\?!,\.]+)',lambda m: m.group(1)[0],sentence)
    sentence=re.sub(r'[ ]+',' ',sentence)
    return [token.lemma_ for token in nlp.tokenizer(sentence) if token.lemma_ not in nlp.Defaults.stop_words]#

def CustomTokenizerNltk(sentence):
    sentence=  re.sub(r'[^a-zA-Z0-9!?,\.\']',' ',sentence)
    sentence= re.sub(r'([\?!,\.]+)',lambda m: m.group(1)[0],sentence)
    sentence=re.sub(r'[ ]+',' ',sentence)
    sentence= re.sub('n\'t',r' not',sentence)
#     stemmer=nltk.stem.WordNetLemmatizer()
    stemmer= nltk.stem.PorterStemmer()
    return [stemmer.stem(token) for token in nltk.tokenize.word_tokenize(sentence) if token not in stop_words]

from torchtext.vocab import GloVe

comment_text= Field(sequential=True,use_vocab=True,
                    lower=True,batch_first=True,tokenize=CustomTokenizerSpacy,
                    fix_length=100,pad_first=False,truncate_first=False)
toxic= Field(sequential=False,use_vocab=False)
severe_toxic= Field(sequential=False,use_vocab=False)
obscene= Field(sequential=False,use_vocab=False)
threat= Field(sequential=False,use_vocab=False)
insult= Field(sequential=False,use_vocab=False)
identity_hate= Field(sequential=False,use_vocab=False)


class Classification(nn.Module):
  def __init__(self,vocab_size,n_class,pretrained_weights=None):
    super(Classification,self).__init__()
    self.embed= torch.nn.Embeding



















# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')


