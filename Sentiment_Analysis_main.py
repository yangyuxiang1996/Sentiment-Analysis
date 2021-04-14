from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from Sentiment_Analysis_DataProcess import prepare_data,build_word2vec,Data_set
from sklearn.metrics import confusion_matrix,f1_score,recall_score
import os
from Sentiment_model import LSTMModel,LSTM_attention
from Sentiment_Analysis_Config import Config
from Sentiment_Analysis_eval import val_accuary
import logging
import numpy as np
from lr_schedular import StepLR
from pytorchtools import EarlyStopping
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S'
                    )



def train(train_dataloader, model, device, epoches, lr):
    logging.info("start training...")
    model.train()
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
#     optimizer = optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=0.1)
    criterion = nn.CrossEntropyLoss()
#     scheduler = StepLR(optimizer, step_size=10, gamma=0.2)  # 学习率调整
    
    if os.path.exists(os.path.join(Config.model_state_dict_path, "best_val_result.txt")):
        with open(os.path.join(Config.model_state_dict_path, "best_val_result.txt")) as f:
            best_acc = float(list(f.readlines())[0].split("=")[-1])
            logging.info("the best val_acc of last train is: %s " % best_acc)
    else:
        best_acc = -np.inf
        
    early_stopping = EarlyStopping(patience=Config.patience, verbose=True)
    
    train_avg_losses = []
    val_avg_losses = []
    for epoch in tqdm(range(epoches)):  # 一个epoch可以认为是一次训练循环
        batch_losses = []
        correct = 0
        total = 0
        logging.info("epoch: %s, batches: %s" % (epoch, len(train_dataloader)))

#         train_dataloader = tqdm(train_dataloader)
#         train_dataloader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epoches, 'lr:', scheduler.get_last_lr()[0]))
        for i, data_ in (enumerate(train_dataloader)):
            optimizer.zero_grad()
            input_, target = data_[0], data_[1]
            input_ = input_.type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            input_ = input_.to(device)
            target = target.to(device)
            output = model(input_)
            # 经过模型对象就产生了输出
            target = target.squeeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
            _, predicted = torch.max(output, 1)
            #print(predicted.shape)
            # 此处的size()类似numpy的shape: np.shape(train_images)[0]
            total += target.size(0)
            #print(target.shape)
            correct += (predicted == target).sum().item()
            F1 = f1_score(target.cpu(), predicted.cpu())
            Recall = recall_score(target.cpu(), predicted.cpu())
            #CM=confusion_matrix(target.cpu(),predicted.cpu())
            
            
            if (i%10==0):
                postfix = {'epoch: {}, batch: {}, train_loss: {:.5f}, acc: {:.3f}%, F1: {:.3f}%, Recall: {:.3f}%'.format(epoch, i, \
                                                                np.mean(batch_losses), 100 * correct / total, 100*F1, 100 * Recall)}
                logging.info(postfix)

        mean_val_loss, val_acc, val_f1, val_recall = val_accuary(model, val_dataloader, device, criterion)
        logging.info('epoch: {}, val_loss: {:.5f}, acc: {:.3f}%, F1：{:.3f}%, Recall: {:.3f}%'.format(epoch, mean_val_loss,\
                                                                                             100*val_acc,100*val_f1,100*val_recall))
        if val_acc > best_acc:
            best_acc = val_acc
            if os.path.exists(Config.model_state_dict_path) == False: 
                os.mkdir(Config.model_state_dict_path)
            torch.save(model, os.path.join(Config.model_state_dict_path, 'sen_model_best.pkl'))
            with open(os.path.join(Config.model_state_dict_path, "best_val_result.txt"), "w") as f:
                f.write("best val_acc=%s\n"%val_acc)
                f.write("best val_f1=%s\n"%val_f1)
                f.write("best val_recall=%s\n"%val_recall)
                    
        train_avg_losses.append(np.mean(train_avg_losses))
        val_avg_losses.append(mean_val_loss)
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(mean_val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    model.load_state_dict(torch.load('checkpoint.pt'))
    return  model, train_avg_losses, val_avg_losses


if __name__ == '__main__':
    splist=[]
    word2id={}
    with open(Config.word2id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()#去掉\n \t 等
            splist.append(sp)
        word2id=dict(splist)#转成字典

    for key in word2id:# 将字典的值，从str转成int
        word2id[key]=int(word2id[key])


    id2word={}#得到id2word
    for key,val in word2id.items():
        id2word[val]=key

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    train_array,train_lable,val_array,val_lable,test_array,test_lable=prepare_data(word2id,
                                                             train_path=Config.train_path,
                                                             val_path=Config.val_path,
                                                             test_path=Config.test_path,seq_lenth=Config.max_sen_len)

    logging.info("training examples: {}".format(len(train_array)))
    logging.info("training label=1: {}".format(np.sum(train_lable == 1)))
    logging.info("training label=0: {}".format(np.sum(train_lable == 0)))
    train_loader = Data_set(train_array, train_lable)
    train_dataloader = DataLoader(train_loader,
                                 batch_size=Config.batch_size,
                                 shuffle=True,
                                 num_workers=0)#用了workers反而变慢了

    val_loader = Data_set(val_array, val_lable)
    val_dataloader = DataLoader(val_loader,
                                 batch_size=Config.batch_size,
                                 shuffle=True,
                                 num_workers=0)

    test_loader = Data_set(test_array, test_lable)
    test_dataloader = DataLoader(test_loader,
                                 batch_size=Config.batch_size,
                                 shuffle=True,
                                 num_workers=0)
    w2vec=build_word2vec(Config.pre_word2vec_path,word2id,None)#生成word2vec
    w2vec=torch.from_numpy(w2vec)
    w2vec=w2vec.float()#CUDA接受float32，不接受float64
    if Config.attention:
        model = LSTM_attention(Config.vocab_size, Config.embedding_dim, w2vec, Config.update_w2v,
                               Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class, Config.bidirectional)
    else:
        model = LSTMModel(Config.vocab_size, Config.embedding_dim, w2vec, Config.update_w2v,
                          Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class, Config.bidirectional)

    #训练
    model, train_avg_losses, val_avg_losses = train(train_dataloader, model=model, device=device, epoches=Config.n_epoch, lr=Config.lr)


    #保存模型
    if os.path.exists(Config.model_state_dict_path) == False:
        os.mkdir(Config.model_state_dict_path)
    torch.save(model, os.path.join(Config.model_state_dict_path, "sen_model.pkl"))
