#GENERATE THE TRAINIGN DATA

import codecs
from collections import Counter
import cPickle
from re import compile as _Re

def split_unicode_chrs( text ):
	_unicode_chr_splitter = _Re( '(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)' ).split
	return [ chr for chr in _unicode_chr_splitter( text ) if chr ]
	
def read_file(file_name):
	f = codecs.open(file_name).readlines()
	f = [i.strip() for i in f]
	f = [i.decode('utf-8') for i in f]
	f = [split_unicode_chrs(i) for i in f]
	return f
	
def build_index(f):
	all_chars = Counter()
	for i in f:
		all_chars.update(i)
	char_to_index = {x[0]: i+1 for i, x in enumerate(all_chars.most_common())}
	index_to_char = {i+1: x[0] for i, x in enumerate(all_chars.most_common())}
	return char_to_index, index_to_char

def build_data(f, feed_length=32, step=1):
	"""
	construct the training data
	:param feed_length: the length of input of RNN
	:param step: timesteps of sampling
	:return:
	"""


	data = []
	label = []
	
	for sent in f:
		cur_data = [sent[max(i, 0):i+feed_length] for i in xrange(1-feed_length, len(sent)-feed_length, step)]
		cur_data = [[char_to_index[i] for i in j] for j in cur_data]
		cur_label = [char_to_index[sent[i+feed_length]] for i in xrange(1-feed_length, len(sent)-feed_length, step)]       
		data.extend(cur_data)
		label.extend(cur_label)
	return data, label

if __name__ == "__main__":
    new_file = read_file('../data/quan_tang_shi.txt')
    char_to_index, index_to_char = build_index(new_file)
    print "%d chars found in data" %(len(char_to_index))

    data, label = build_data(new_file)
    print "%d training samples generated" %(len(data))
    cPickle.dump([char_to_index, index_to_char], open('../data/dict.pkl', 'w+'))
    cPickle.dump([data, label], open('../data/data_train.pkl', 'w+'))

