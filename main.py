from models.model import *
from utils.data_util import load_data
from utils.data_loader import *
import numpy as np
import argparse
import torch
import time
import os
from tqdm import tqdm
import torch.nn.functional as F


def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout': 0.4,
        'cuda': 7,
        'epochs': 1000,
        'weight_decay': 0,
        'seed': 1000010,
        'model': 'MoCi',
        'dim': 256,
        'r_dim': 256,
        'dataset': 'YAGO15K',
        'image_features': 1,
        'text_features': 1,
        'eval_freq': 10,
        'temp': 2,
        'bias': 1,
        'batch_size': 256,
        'save': 1
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", action="append", default=val)
    args = parser.parse_args()
    return args

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, attr_features, text_features,  train_data, val_data, test_data = load_data(args.dataset)

print("Training data {:04d}".format(len(train_data[0])))

if args.model in ['ConvE', 'TuckER', 'MoCi']:
    corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)
else:
    corpus = ConvKBCorpus(args, train_data, val_data, test_data, entity2id, relation2id)
if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.attr_features:
    args.attr = F.normalize(torch.Tensor(attr_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)


args.entity2id = entity2id
args.relation2id = relation2id

model_name = {
    'OnlyConvKB': OnlyConvKB,
    'IKRLConvKB': IKRLConvKB,
    'ConvE': ConvE,
    'TuckER': TuckER,
    'MoCi': MoCi,
    'IKRL': IKRL,
    'MKGC': MKGC
}

def pre_train_decoder(args):
    model = model_name[args.model](args, corpus.train_head_entities, corpus.train_relations, corpus.train_tail_entities, corpus.entity_matrix, corpus.relation_matrix)
    print(str(model))
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    # Train Model
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num

    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = []
        t = time.time()
        corpus.shuffle()

        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
            output = model.forward(train_indices, train_values, model_split='pre_train_decoder')
            loss = model.pre_loss_func(output,  train_values)

            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss.append(loss.data.item())
        lr_scheduler.step()

        if args.save and epoch == args.epochs-1:
            torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}_pre.pth')
            print('Saved model!')


def train_decoder(args):
    model = model_name[args.model](args, corpus.train_head_entities, corpus.train_relations, corpus.train_tail_entities, corpus.entity_matrix, corpus.relation_matrix)
    model.load_state_dict(
        torch.load('./checkpoint/{}/{}_pre.pth'.format(args.dataset, args.model)), strict=False)
    print(str(model))
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')

    # Train Model
    t_total = time.time()
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num

    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = []
        t = time.time()
        corpus.shuffle()

        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
            output = model.forward(train_indices)
            loss = model.loss_func(output, train_values)

            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss.append(loss.data.item())
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\n".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
            model.eval()
            with torch.no_grad():
                val_metrics = corpus.get_validation_pred(model, 'test')
            if val_metrics['Mean Reciprocal Rank'] > best_test_metrics['Mean Reciprocal Rank']:
                best_test_metrics['Mean Reciprocal Rank'] = val_metrics['Mean Reciprocal Rank']
            if val_metrics['Mean Rank'] < best_test_metrics['Mean Rank']:
                best_test_metrics['Mean Rank'] = val_metrics['Mean Rank']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            model.format_metrics(val_metrics, 'test')]))
    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():
            best_test_metrics = corpus.get_validation_pred(model, 'test')

    print(' '.join(['Val set results:',
                    model.format_metrics(best_val_metrics, 'val')]))
    print(' '.join(['Test set results:',
                    model.format_metrics(best_test_metrics, 'test')]))
    if args.save:
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')
        print('Saved model!')


if __name__ == '__main__':
    # pre_train_decoder(args)
    train_decoder(args)
