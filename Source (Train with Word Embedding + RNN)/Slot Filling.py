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
from keras import optimizers

# Shuffle
def shuffle(data, batch):
    #data : list of  data list use as input
    #batch : batch size for the shuf (or seed)
    for d in data:
        random.seed(batch)
        random.shuffle(d)

# Read file JSON
def read_JSON(filein):
    with open(filein) as json_file:
        json_data = json.load(json_file)
        return json_data

# Write file JSON
def write_JSON(data, fileout):
    with open(fileout, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)

# Load file (1 file)
def load_atis(filein):
    f = open(str(filein),'rb')
    train_set, test_set, dicts = cPickle.load(f,encoding='latin1')
    f.close()
    return train_set, test_set, dicts

# Separate file (from original file into 3 file: train, test, dicts)
def write_atis(filein):
    # Read original file
    train_set, test_set, dicts = load_atis(filein)
    # Write train.pkl
    train = open('train.pkl', 'wb')
    cPickle.dump(train_set, train)
    # Write test.pkl
    test = open('test.pkl', 'wb')
    cPickle.dump(test_set, test)
    # Write dicts.pkl
    dic = open('dicts.pkl', 'wb')
    cPickle.dump(dicts, dic)
    train.close()
    test.close()
    dic.close()
    print('Separation successful !')

# Load file (3 file: train, test, dicts)
def  load_atis_splitted(train, test, dicts):
    f1 = open(str(train),'rb')
    train_set = cPickle.load(f1,encoding='latin1')
    f2 = open(str(test),'rb')
    test_set = cPickle.load(f2,encoding='latin1')
    f3 = open(str(dicts),'rb')
    dicts = cPickle.load(f3,encoding='latin1')
    f1.close()
    f2.close()
    f3.close()
    return train_set, test_set, dicts

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
# Stuff funct for Statistic and find the best value of error in validation set
# alpha: learning rate
# nums_hidden: numbers of hidden node
# seed: batch size
# dim: embedding dimension
# nums_epochs: numbers of epoch
def stuff(alpha, nums_hidden, seed, dim, nums_epochs):
    # Config model
    s = {'dataset':'atis.pkl',
         'lr':alpha,
         'nhidden': nums_hidden, # number of hidden units
         'batch': seed,
         'emb_dimension': dim, # dimension of word embedding
         'nepochs': nums_epochs}
    
    # Make result folder, contain: model file, result.json, output file for validation and train, also save config.json again
    folder = os.path.basename("Result").split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)
    write_JSON(s,folder + '/config.json')
    
    # load the dataset
    train_set, test_set, dic = load_atis(s['dataset'])    
    
    # Convert for index
    # Vocabulary of meaningful words and labels are covered in dic data of Atis datset
    idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].items())
    
    # Seperate data for hold-out: train set, validation set
    #words2idx, tables2idx and labels2idx (use only vocabs and labels. Tables will list all meaning of words_which not neccessary)
    train_words, train_tables, train_labels = train_set
    valid_words = train_words[0:499]
    valid_tables = train_tables[0:499]
    valid_labels = train_labels[0:499]
    
    train_words = train_words[500:len(train_words)]
    train_tables = train_tables[500:len(train_tables)]
    train_labels = train_labels[500:len(train_labels)]
    
    # Print some info
    #print('Train set:',str(len(train_words)), 'sentences')
    #print('Validation set:',str(len(valid_words)),'sentences')
    
    # Some para use in 'for loop'
    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_words)
    #print('Nums of vocabulary get from words of each sentence: ', vocsize)
    #print('Nums of slot: ', nclasses)
    #print('Nums of sentence use for training: ', nsentences)
    
    # instanciate the model (for randomize duel to bath size_ optimization convergence)
    np.random.seed(s['batch'])
    random.seed(s['batch'])
    
    #Making model
    model = Sequential()    #Init
    model.add(Embedding(vocsize, s['emb_dimension']))    # Word Embedding
    model.add(SimpleRNN(s['nhidden'], activation='sigmoid', return_sequences=True))    # Recurrent use Sigmoid Activation
    model.add(TimeDistributed(Dense(output_dim=nclasses)))    # For making Dense Layer (Context Layer) keep updating 
    model.add(Activation("softmax"))    # Softmax activation for classification
    adam = Adam(lr=s['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)    # Adam optimizer (some hyperparameter will be locked)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])    # Lost funct: Cross entropy

    # Train
    for e in range(s['nepochs']):
        # shuffle
        shuffle([train_words, train_tables, train_labels], s['batch'])
        s['ce'] = e
        for i in range(nsentences):
            X = np.asarray([train_words[i]])
            Y = to_categorical(np.asarray(train_labels[i])[:, np.newaxis],nclasses)[np.newaxis, :, :]
            if X.shape[1] == 1:
                continue # bug with X, Y of len 1
            model.train_on_batch(X, Y)

            
        # Evaluation // back into meaning word for output prediction : idx -> words
       
       # Train
        predictions_train = [map(lambda x: idx2label[x], model.predict_on_batch( np.asarray([x])).argmax(2)[0]) for x in train_words]
        groundtruth_train = [ map(lambda x: idx2label[x], y) for y in train_labels ]
        words_train = [ map(lambda x: idx2word[x], w) for w in train_words]
        
        # Validatae
        predictions_valid = [map(lambda x: idx2label[x], model.predict_on_batch( np.asarray([x])).argmax(2)[0]) for x in valid_words]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_labels ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_words]
        
        # Evaluation 
        valid_error = acc_measurement(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')
        train_error  = acc_measurement(predictions_train, groundtruth_train, words_train, folder + '/current.train.txt')
        
        # Save weight in file 'model.h5' as HDF5 file (default), can be load by: model.load_weights('model.h5', by_name=False)
        model.save_weights(folder +'/model_weight.h5', overwrite=True)
        #print ('MODEL built at epoch = ', e, ', error in validation set = ', valid_error)
        s['current_valid_error'] = valid_error
        s['current_train_error'] = train_error
    
    # Make output file
    if os.path.exists(folder + '/valid.txt'):
        os.remove(folder + '/valid.txt')
    if os.path.exists(folder + '/train.txt'):
        os.remove(folder + '/train.txt')
    os.rename(folder + '/current.valid.txt',folder + '/valid.txt')
    os.rename(folder + '/current.train.txt',folder + '/train.txt')
    result = read_JSON('result.json')
    result['validation error'] = float(s['current_valid_error'])
    result['train error'] = float(s['current_train_error'])
    write_JSON(result,folder + '/result.json')
    #Print final result model
    print ('RESULT MODEL built at epoch = ', e, ',error in validation set = ', s['current_valid_error'], ',error in train set = ', s['current_train_error'])
    print('\n')

# Build model
def train_model():
    # Read config json file
    s = read_JSON('config.json')
    
    # Make result folder, contain: model file, result.json, output file for validation and test, also save config.json again
    folder = os.path.basename("Result").split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)
    write_JSON(s,folder + '/model_config.json')
    
    # load the dataset
    train_set, test_set, dic = load_atis(s['dataset'])    
    
    # Convert for index
    # Vocabulary of meaningful words and labels are covered in dic data of Atis datset
    idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].items())
    
    #words2idx, tables2idx and labels2idx (use only vocabs and labels. Tables will list all meaning of words_which not neccessary)
    train_words, train_tables, train_labels = train_set
    test_words,  test_tables,  test_labels  = test_set
    
    # Some para use in 'for loop'
    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_words)
    print('Nums of vocabulary get from words of each sentence: ', vocsize)
    print('Nums of slot: ', nclasses)
    print('Nums of sentence use for training: ', nsentences)
    print('------------------------------------------------')
    # instanciate the model (for randomize duel to bath size_ optimization convergence)
    np.random.seed(s['batch'])
    random.seed(s['batch'])
    
    #Making model
    model = Sequential()    #Init
    model.add(Embedding(vocsize, s['emb_dimension']))    # Word Embedding
    model.add(SimpleRNN(s['nhidden'], activation='sigmoid', return_sequences=True))    # Recurrent use Sigmoid Activation
    model.add(TimeDistributed(Dense(output_dim=nclasses)))    # For making Dense Layer (Context Layer) keep updating 
    model.add(Activation("softmax"))    # Softmax activation for classification
    adam = Adam(lr=s['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)    # Adam optimizer (some hyperparameter will be locked)
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)        #SGD + momentum
    #adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)        #AdaGrad
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])    # Lost funct: Cross entropy

    # Train
    print('------------------------------------------------')
    print('Training...')
    for e in range(s['nepochs']):
        # shuffle
        shuffle([train_words, train_tables, train_labels], s['batch'])
        s['ce'] = e
        for i in range(nsentences):
            X = np.asarray([train_words[i]])
            Y = to_categorical(np.asarray(train_labels[i])[:, np.newaxis],nclasses)[np.newaxis, :, :]
            if X.shape[1] == 1:
                continue # bug with X, Y of len 1
            model.train_on_batch(X, Y)
        # Save weight in file 'model.h5' as HDF5 file (default), can be load by: model.load_weights('model.h5', by_name=False)
        model.save_weights(folder +'/model_weight.h5', overwrite=True)
        print(str(e + 1),' epoches done!...')
    # Print sign
    print('Finished!...')
    print('------------------------------------------------')
if __name__ == '__main__':
    # Statistic for  choosing best hyper parameter
    #stuff(0.01, 10, 234, 10, 10)
    #stuff(0.01, 10, 234, 10, 20)
    #stuff(0.01, 10, 234, 100, 10)
    #stuff(0.01, 10, 234, 100, 20)
    #stuff(0.01, 10, 345, 10, 10)
    #stuff(0.01, 10, 345, 10, 20)
    #stuff(0.01, 10, 345, 100, 10)
    #stuff(0.01, 10, 345, 100, 20)
    #stuff(0.01, 100, 234, 10, 10)
    #stuff(0.01, 100, 234, 10, 20)
    #stuff(0.01, 100, 234, 100, 10)
    #stuff(0.01, 100, 234, 100, 20)
    #stuff(0.01, 100, 345, 10, 10)
    #stuff(0.01, 100, 345, 10, 20)
    #stuff(0.01, 100, 345, 100, 10)
    #stuff(0.01, 100, 345, 100, 20)
    #stuff(0.001, 10, 234, 10, 10)
    #stuff(0.001, 10, 234, 10, 20)
    #stuff(0.001, 10, 234, 100, 10)
    #stuff(0.001, 10, 234, 100, 20)
    #stuff(0.001, 10, 345, 10, 10)
    #stuff(0.001, 10, 345, 10, 20)
    #stuff(0.001, 10, 345, 100, 10)
    #stuff(0.001, 10, 345, 100, 20)
    #stuff(0.001, 100, 234, 10, 10)
    #stuff(0.001, 100, 234, 10, 20)
    #stuff(0.001, 100, 234, 100, 10)
    #stuff(0.001, 100, 234, 100, 20)
    #stuff(0.001, 100, 345, 10, 10)
    #stuff(0.001, 100, 345, 10, 20)
    #stuff(0.001, 100, 345, 100, 10)
    #stuff(0.001, 100, 345, 100, 20)
    train_model()
    