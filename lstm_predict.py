#!/usr/bin/env python3
# coding: utf-8
# File: lstm_predict.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import matplotlib.pyplot as plt
import os
import math
from ast import literal_eval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMNER:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        #self.embedding_file = os.path.join(cur, 'model/zh.tsv')
        #self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/model_token_300_16.h5')
        self.word_dict = self.load_worddict()
        self.class_dict ={
                         'O':0,
                         'TREATMENT-I': 1,
                         'TREATMENT-B': 2,
                         'BODY-B': 3,
                         'BODY-I': 4,
                         'SIGNS-I': 5,
                         'SIGNS-B': 6,
                         'CHECK-B': 7,
                         'CHECK-I': 8,
                         'DISEASE-I': 9,
                         'DISEASE-B': 10
                        }
        self.label_dict = {j:i for i,j in self.class_dict.items()}
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 500
        self.embedding_matrix = self.build_embedding_matrix()
        self.model = self.tokenvec_bilstm2_crf_model()
        self.model.load_weights(self.model_path)

    '加载词表'
    def load_worddict(self):
        vocabs = [line.strip() for line in open(self.vocab_path, 'r', encoding='UTF-8')]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

    '''构造输入，转换成所需形式'''
    def build_input(self, text):
        x = []
        for char in text:
            if char not in self.word_dict:
                char = 'UNK'
            x.append(self.word_dict.get(char))
        x = pad_sequences([x], self.TIME_STAMPS)
        return x

    def predict(self, text):
        str = self.build_input(text)
        raw = self.model.predict(str)[0][-self.TIME_STAMPS:]
        result = [np.argmax(row) for row in raw]
        chars = [i for i in text]
        tags = [self.label_dict[i] for i in result][len(result)-len(text):]
        return chars, tags

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r', encoding='UTF-8') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    '''使用预训练向量进行模型训练'''
    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        return model

    def collect_entities_bio(self, sent, tags):

        append = lambda x, y, z: x.append('{}/{}'.format(''.join(y), z[0])) \
            if len(y) != 0 else None

        entities = []
        entity = []
        last_tag = '<T_START>'
        for word, tag in zip(sent, tags):
            if tag == 'O':
                tag = 'o'
                if last_tag != tag:
                    append(entities, entity, last_tag)
                    entity = [word]
                    last_tag = tag
                else:
                    entity.append(word)
                    last_tag = tag

            else:
                prefix, cls = tag.split('-')
                if cls == 'B':
                    append(entities, entity, last_tag)
                    entity = [word]
                    last_tag = tag
                else:
                    if prefix[0] == last_tag[0] and tag[-1] == 'I':#(last_tag[-1] == 'B' or last_tag[-1] == 'I'):
                        entity.append(word)
                        last_tag = tag
                    else:
                        append(entities, entity, last_tag)
                        entity = [word]
                        last_tag = tag

        append(entities, entity, last_tag)
        return entities

    def count_correct_entities(self, inputs, golden_tags, predict_tags):

        for sent, gold_tags, pred_tags in zip(inputs, golden_tags, predict_tags):
            # gold_tags = gold_tags[:l]
            # pred_tags = pred_tags[:l]
            # sent = sent[:l]
            print(sent)
            print(gold_tags)
            print(pred_tags)
            pred_entities = self.collect_entities_bio(sent, pred_tags)
            gold_entities = self.collect_entities_bio(sent, gold_tags)

            for pred_entity in pred_entities:
                if pred_entity in gold_entities:
                    self.entity_right_num += 1
            self.entity_gold_num += len(gold_entities)
            self.entity_pred_num += len(pred_entities)

        acc = sum(list(self.correct_tags_number.values())) / sum(list(self.predict_tags_counter.values()))
        return acc

if __name__ == '__main__':
    ner = LSTMNER()
    '''
    while 1:
        s = input('enter an sent:').strip()
        ner.predict(s)
    '''
    file_path = 'data-unlabeled-1'
    file_result_path = 'data-labeled-2'
    for root, dirs, files in os.walk(file_path):
        for dir in dirs:
            if not os.path.exists(file_result_path+'/'+dir):
                os.makedirs(file_result_path+'/'+dir)
        for file in files:
            filepath = os.path.join(root, file)
            result_path = filepath.replace(file_path, file_result_path).replace('.txtoriginal','')
            #if os.path.exists(result_path):
            #    continue
            print(filepath)
            f = open(filepath, 'r', encoding='UTF-8')
            words = ''.join(f.readlines()).replace('\n','').replace('\r','')
            f.close()
            result_words, result_tags = ner.predict(words)
            #print(result_tags)
            result = ner.collect_entities_bio(result_words, result_tags)
            leng = 1
            bios = []
            for bio in result:
                if bio[-1] != 'o':
                    bios.append(bio[:-2]+' '+str(leng)+' '+str(leng+len(bio)-3)+' '+bio[-1])
                leng = leng + len(bio) - 2
            print(bios)
            bios = '\n'.join(bios)
            bios = bios.replace('C', '检查与检验').replace('S', '症状和体征').replace('D', '疾病和诊断').replace('T', '治疗').replace('B', '身体部位')
            result_path = filepath.replace(file_path,file_result_path).replace('.txtoriginal','')
            '''
            if os.path.exists(result_path):
                continue
            '''
            f_result = open(result_path, 'w', encoding='UTF-8')
            f_result.write(bios)
            f_result.close()
