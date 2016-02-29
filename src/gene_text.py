# -*- coding: utf-8 -*-
import cPickle
from keras.models import model_from_json
from model import make_poem_index, print_poem
import argparse

parser = argparse.ArgumentParser(description='Generate new poems.')
parser.add_argument("text", type=str, help="Chinese characters in the poem")
parser.add_argument('-m', '--model', default='../data/my_model_architecture.json',
                    help='Model file location')
parser.add_argument('-w', '--weight', default='../data/my_model_weights.h5',
                    help='weight file location')

args = parser.parse_args()

#load model
model = model_from_json(open(args.model).read())
model.load_weights(args.weight)
char_to_index, index_to_char = cPickle.load(open('../data/dict.pkl', 'r+'))

#predict
s = make_poem_index(model, args.text)
s = [index_to_char[i] for i in s]
print_poem(s)


