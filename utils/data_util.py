import h5py
import pickle
import torch
import numpy as np
# from pytorch_pretrained_bert import BertTokenizer, BertModel  #pytorch = v1.10.0
# from transformers import BertTokenizer, BertModel
import torch
import re
import pickle
import pandas as pd
import os

# Get item2id and write into txt
def write_index_dict(datasets):
    path = 'datasets/' + datasets + '/'
    print(path)
    entities = set()
    relations = set()
    with open(path + datasets + '_EntityTriples.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split(' ')
            entities.add(instance[0])
            relations.add(instance[1])
            entities.add(instance[2])

    with open(path + 'entity2id.txt', 'w', encoding='utf-8') as f:
        for index, entity in enumerate(entities):
            f.write(entity + ' ' + str(index) + '\n')

    with open(path + 'relation2id.txt', 'w', encoding='utf-8') as f:
        for index, relation in enumerate(relations):
            f.write(relation + ' ' + str(index) + '\n')

# Bind img features with id
def write_img_vec(datasets):
    path = 'datasets/' + datasets + '/'
    entities = {}
    with open(path + datasets + '_ImageIndex.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split('\t')
            entities[instance[0]] = instance[1]

    img_features = []
    with open(path + 'entity2id.txt', 'r', encoding='utf-8') as f:
        with h5py.File(path + datasets + '_ImageData.h5', 'r') as img:  # img.key()= [...,'DBIMG12841']
            img_all = np.array([feats for feats in img.values()])
            img_mean = np.mean(img_all.reshape(-1, img_all.shape[2]), 0)
            for line in f:
                instance = line.strip().split(' ')
                entity = instance[0]  # entity => '<http://dbpedia.org/resource/Wedding_Crashers>'
                if entity in entities.keys():  # entities.keys() => <http://dbpedia.org/resource/Andy_Griffith> DBIMG00001 ...
                    img_features.append(np.array(img[entities[entity]]).flatten())
                else:
                    img_features.append(img_mean)

    img_features = np.array(img_features)
    pickle.dump(img_features, open(path + 'img_features.pkl', 'wb'))

def data_preprocess(datasets):
    # write_index_dict(datasets)
    # write_img_vec(datasets)
    # write_attr_vec(datasets)

def read_relation_from_id(path):
    relation2id = {}
    with open(path + 'relation2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])

    return relation2id


# Calculate adjacency matrix
def get_adj(path, split):
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    triples = []
    rows, cols, data = [], [], []
    unique_entities = set()
    with open(path + split + '.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # instance = line.strip().split('\t')
            instance = line.strip().split(' ')
            e1, r, e2 = instance[0], instance[1], instance[2]
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
            rows.append(entity2id[e2])
            cols.append(entity2id[e1])
            data.append(relation2id[r])

    return triples, (rows, cols, data), unique_entities


def read_entity_from_id(path):
    entity2id = {}
    with open(path + 'entity2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])

    return entity2id

# Load data triples and adjacency matrix
def load_data(datasets):
    path = 'datasets/' + datasets + '/'
    train_triples, train_adj, train_unique_entities = get_adj(path, 'train')
    val_triples, val_adj, val_unique_entities = get_adj(path, 'val')
    test_triples, test_adj, test_unique_entities = get_adj(path, 'test')
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    img_features = pickle.load(open(path + 'img_features.pkl', 'rb'))  # img_features.shape ==> (12842, 4096)
    text_features = pickle.load(open(path + 'text_features.pkl', 'rb'))  # 768维
    attr_features = None
    return entity2id, relation2id, img_features, attr_features, text_features, \
        (train_triples, train_adj, train_unique_entities), \
        (val_triples, val_adj, val_unique_entities), \
        (test_triples, test_adj, test_unique_entities)


# Split data into train, val and test
def dataset_split(datasets):
    path = 'datasets/' + datasets + '/'
    with open(path + datasets + '_EntityTriples.txt', 'r', encoding='utf-8') as f:
        triples = f.readlines()

    np.random.shuffle(triples)
    nb_val = round(0.05 * len(triples))
    nb_test = round(0.05 * len(triples))
    val_triples, test_triples, train_triples = triples[:nb_val], triples[nb_val: nb_val + nb_test], triples[
                                                                                                    nb_val + nb_test:]

    with open(path + 'train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_triples)
    with open(path + 'val.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_triples)
    with open(path + 'test.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_triples)


def data_loader(datasets):
    path = 'datasets/' + datasets
    with open(path + '/' + datasets + '_EntityTriples.txt', 'r', encoding='utf-8') as f:
        for line in f:
            print(line)
