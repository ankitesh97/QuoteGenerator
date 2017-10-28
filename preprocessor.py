
import json
import numpy as np
import pickle
import re
import nltk
import itertools
import time
import pandas as pd


params = json.loads(open("params.json").read())
VOCAB_SIZE = params['preprocess']['vocab_size']
SENTENCE_START = 'SENTENCE_START'
SENTENCE_END = 'SENTENCE_END'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
FILE_NAME = 'dataVectorizedv0.0'

class preprocess():

    def __init__(self):

        self.data = []
        self.X = []
        self.y = []

        # this contains the actual data in numerical
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_validate = []
        self.y_validate = []

        # meta data about the vocab
        self.index_to_word = []
        self.word_to_index = {}
        self.unknown = []
        self.vocab = []


    def makeData(self):
        self.getData()
        self.makeXy()
        self.equalize()


    def load(self):
        picklefile = open('pickledfiles/'+FILE_NAME,'r')
        obj = pickle.loads(picklefile.read())
        return obj

    def getData(self):
        dps = params["preprocess"]['total']
        df = pd.read_csv('quotes.csv')
        from_csv = list(np.array(df.get('quote')))
        from_csv =  list(set(from_csv))

        data = open('quotes.txt','r').read().split('\n')[:-1]
        data = [x.split('\t')[-1] for x in data]
        final = from_csv + data
        data = list(set(final))
        print len(data)
        final = []
        for i in range(len(data)):
            data[i] = data[i].replace("'", " ' ")
            data[i] = data[i].replace('"', ' " ')
            data[i] = data[i].replace('.', ' . ')
            data[i] = data[i].lower()
            l = len(nltk.word_tokenize(data[i]))
            if l<=50:
                final.append(data[i])

        print "total dp " + str(len(final))
        self.data = final[:dps]

    def makeXy(self):
        train = []
        data = []
        #add start token and end token
        for sent in self.data:
            data.append(SENTENCE_START+" "+sent+" "+SENTENCE_END)

        #this contains tokens for all the sentences
        tokenized = [nltk.word_tokenize(data[i]) for i in range(len(data))]
        frequency = nltk.FreqDist(itertools.chain(*tokenized))
        self.vocab = frequency.most_common(VOCAB_SIZE-1)
        #CREATE TWO MAPPING WORD_TO_INDEX AND INDEX_TO_WORD
        #we also have an UNKNOWN_TOKEN i.e if the word is not found in our vocab then we replace it by UNKNOWN
        self.index_to_word = [x[0] for x in self.vocab]
        self.index_to_word.append(UNKNOWN_TOKEN)
        for i,w in enumerate(self.index_to_word):
            self.word_to_index[w] = i
        #
        # #replace the words that are not in our vocab with the UNKNOWN_TOKEN
        #
        for i in xrange(len(tokenized)):
            for j in xrange(len(tokenized[i])):
                if(self.word_to_index.has_key(tokenized[i][j])==False):
                    self.unknown.append(tokenized[i][j])
                    tokenized[i][j] = UNKNOWN_TOKEN
        #

        #
        #now to create arrays of the input
        for i in xrange(len(tokenized)):
            X,y = self.convertToXy(tokenized[i])
            self.X.append(X)
            self.y.append(y)




    def convertToXy(self, tokenized):
        X = []
        y = []
        for i in xrange(len(tokenized)-1):
            X.append(self.word_to_index[tokenized[i]])
            y.append(self.word_to_index[tokenized[i+1]])

        return np.array(X,dtype=np.int32),np.array(y,dtype=np.int32)


    def equalize(self):
        # print self.word_to_index
        max_l = 0
        for x in self.X:
            max_l = max(len(x),max_l)
        end = self.word_to_index["SENTENCE_END"]
        for i in range(len(self.X)):
            l = max_l - len(self.X[i])
            to_append = [end]*l
            self.X[i] = np.concatenate((self.X[i],to_append))
            self.y[i] = np.concatenate((self.y[i],to_append))




if __name__ == '__main__':
    start = time.time()
    obj = preprocess()
    obj.makeData()
    pickle_file_sampled_data = open('pickledfiles/'+FILE_NAME,'w')
    pickle.dump(obj,pickle_file_sampled_data)
    print "seconds ---------- "+str(time.time()-start)
