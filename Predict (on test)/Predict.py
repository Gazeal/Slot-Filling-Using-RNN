#Ignore Warning
import warnings
warnings.filterwarnings('ignore')

#Import Lib
import random
import gzip
import _pickle as cPickle
import os
import theano
import numpy as np
import json

from os.path import  join
from keras.models import Sequential
from keras.layers import (Input, Embedding, SimpleRNN, Dense, Activation, TimeDistributed)
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Read file JSON
def read_JSON(filein):
    with open(filein) as json_file:
        json_data = json.load(json_file)
        return json_data

# Write file JSON
def write_JSON(data, fileout):
    with open(fileout, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)

# Load test set and dicts for test only
def  load_test(test, dicts):
    f1 = open(str(test),'rb')
    test_set = cPickle.load(f1,encoding='latin1')
    f2 = open(str(dicts),'rb')
    dicts = cPickle.load(f2,encoding='latin1')
    f1.close()
    f2.close()
    return test_set, dicts

# Compute error
def acc_measurement(p, g, w, filename):
    out = ''
    error = 0
    sumup = 0
    # Write down result file for predicted ones
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
            sumup+=1
            if(wl!=wp):
                error+=1
        out += 'EOS O O\n\n'
    error_prob = str(error/sumup)
    #print(str(error/sumup))
    f = open(filename,'w')
    f.writelines(out)
    f.close()
    # error (val or test)
    return error_prob

if __name__ == '__main__':

    # Read config json file
    s = read_JSON('model_config.json')
    
    # Make result folder, contain: model file, result.json, output file for validation and test, also save model_config.json again
    folder = os.path.basename("Result").split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)
    write_JSON(s,folder + '/model_config.json')
    
    # load the dataset
    test_set, dic = load_test('test.pkl', 'dicts.pkl')
    # Convert for index
    # Vocabulary of meaningful words and labels are covered in dic data of Atis datset
    idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].items())
    print('------------------------------------------------')
    print('Total Labels:')
    for key, value in idx2label.items() :
        print (key,':', value)
    print('------------------------------------------------')
    # Seperate each element of test set
    #words2idx, tables2idx and labels2idx (use only vocabs and labels. Tables will list all meaning of words_which not neccessary)
    test_words,  test_tables,  test_labels  = test_set
    
    # Print some info
    print('Test set:',str(len(test_words)), 'sentences')
    
    # Some para use in 'for loop'
    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    print('Nums of vocabulary get from words of each sentence: ', vocsize)
    print('Nums of slot: ', nclasses)
    print('------------------------------------------------')
    #Re-build hyper parameter
    model = Sequential()    #Init
    model.add(Embedding(vocsize, s['emb_dimension']))    # Word Embedding
    model.add(SimpleRNN(s['nhidden'], activation='sigmoid', return_sequences=True))    # Recurrent use Sigmoid Activation
    model.add(TimeDistributed(Dense(output_dim=nclasses)))    # For making Dense Layer (Context Layer) keep updating 
    model.add(Activation("softmax"))    # Softmax activation for classification
    adam = Adam(lr=s['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)    # Adam optimizer (some hyperparameter will be locked)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])    # Lost funct: Cross entropy
    
    #Load model
    model.load_weights('model_weight.h5', by_name=False)
    
    # Test
    predictions_test = [map(lambda x: idx2label[x], model.predict_on_batch( np.asarray([x])).argmax(2)[0]) for x in test_words]
    groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_labels ]
    words_test = [ map(lambda x: idx2word[x], w) for w in test_words]
    
    # Evaluation 
    test_error  = acc_measurement(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
    
    # Save weight in file 'model.h5' as HDF5 file (default), can be load by: model.load_weights('model.h5', by_name=False)
    model.save_weights(folder +'/model_weight.h5', overwrite=True)
    s['current_test_error'] = test_error
    
    #Make output file
    if os.path.exists(folder + '/valid.txt'):
        os.remove(folder + '/valid.txt')
    if os.path.exists(folder + '/test.txt'):
        os.remove(folder + '/test.txt')
    os.rename(folder + '/current.test.txt',folder + '/test.txt')
    result = read_JSON('result.json')
    result['test error'] = float(s['current_test_error'])
    result['model'] = str('model_config.json and model_weight.h5')
    result['test prediction result'] = str('test.txt')
    write_JSON(result,folder + '/result.json')
    #Print final result model
    print('------------------------------------------------')
    print ('RESULT: error in test set = ', s['current_test_error'])
    print('------------------------------------------------')