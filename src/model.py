# -*- coding: utf-8 -*-
#TRAINING THE MODEL

import cPickle
import numpy as np
from collections import Counter
from sklearn.cross_validation import train_test_split
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing import sequence
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from prepare_data import split_unicode_chrs

def train_model(x, y, x_test=None, y_test=None, num_chars=100, max_len=32, nb_epoch=10):
    """
    :param num_chars: number of different chars in data
    :param max_len: max length of input to RNN
    :return: the model
    """
    embedding_W = np.identity(num_chars+1)

    model=Sequential()
    model.add(Embedding(num_chars+1, num_chars+1, weights=[embedding_W], input_length=max_len, trainable=False))
    model.add(LSTM(512, return_sequences=True, input_shape=(max_len, num_chars+1)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    MODEL_FILE = './try_model.h5'
    early_stop = EarlyStopping(verbose=True, patience=3, monitor='val_loss')
    model_check = ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=True, save_best_only=True)

    model.fit(x, y, batch_size=256, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(x_test, y_test), callbacks = [early_stop, model_check])

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def make_poem_index(model, start_chars):
    """
    :param model: the model used
    :param start_chars: the chars want to use at the beginning of each sentence, have to be string, eg: '新年快乐'
    :return: a list of index of chars
    """
    text = [0]*32
    start_chars = split_unicode_chrs(start_chars.decode('utf-8'))
    num_sentences = len(start_chars)
    start_index = [char_to_index[i] for i in start_chars]

    text.append(start_index.pop(0))
    count = 1
    get_length=False
    for i in range(40):
        count+=1
        inputs = np.asarray([text[-32:]])
        preds = model.predict(inputs, verbose=0)[0]
        next_index = sample(preds, 0.9)+1

        #get the sentence length
        if next_index == 1 and not get_length:
            get_length = count

        #if not come to the end, but get , or . , resample
        if get_length and (next_index == 1 or 2) and count % get_length !=0:
            preds = preds[2:]
            next_index = sample(preds, 0.9)+3

        #check whether come to the end of a sentence
        if get_length and count % get_length == 0:
            if count % (2*get_length) == 0:
                next_index = 2
            else:
                next_index = 1

        text.append(next_index)

        #append the next index when come to an end
        if get_length and count % get_length == 0:
            if start_index:
                text.append(start_index.pop(0))
                count+=1
            else:
                break
    return text[32:]

def print_poem(str_list):
    """
    A helper function to print the poems nicely
    :param str_list: a list of strings, each is a unicode string
    :return:
    """
    col_index = [i for i, j in enumerate(str_list) if j == u'\uff0c' or j == u'\u3002']
    sent_length = col_index[0]+1
    while col_index:
        end_sent = col_index.pop(0)
        cur_sent = str_list[end_sent-sent_length+1: end_sent+1]
        print ''.join(cur_sent)

if __name__ == "__main__":
    char_to_index, index_to_char = cPickle.load(open('../data/dict.pkl', 'r+'))
    data, label = cPickle.load(open('../data/data_train.pkl', 'r+'))
    num_chars = len(char_to_index)

    data = sequence.pad_sequences(data, maxlen=32)
    label = to_categorical([i-1 for i in label])
    print data.shape

    X_train = data[: 1800000]
    X_test = data[1800000:]
    y_train = label[:1800000]
    y_test = label[1800000:]

    model = train_model(X_train, y_train, X_test, y_test, num_chars=num_chars)
    #TODO, ADD YUNJIAO function: to make sure it yayun, if not, resample the last index

    #dumple the model
    json_string = model.to_json()
    open('../data/my_model_architecture.json', 'w').write(json_string)
    model.save_weights('../data/my_model_weights.h5')

    #sample use
    s = make_poem_index(model, '新年快乐')
    s = [index_to_char[i] for i in s]
    print_poem(s)

