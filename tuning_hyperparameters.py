#############This code is just used as a demonstration.##################
#############Distribution without permission may result in legal risk################

import os
from io import BytesIO
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import nni
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, Callback, ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import numpy as np
from time import sleep
import pandas_ta as ta
# %matplotlib inline
from tqsdk2 import TqApi, TargetPosTask, TqSim, TqBacktest, TqAccount, TqAuth, TqKq
from tqsdk2.ta import BOLL, PUBU, ENV, ATR, MACD, ATR, CCI, RSI
from datetime import date
import time
from datetime import datetime
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# from saved_time_list import saved_time_list # 读取存储的time list
from saved_time_list import saved_time_list
from saved_time_list_price import saved_time_list_price
from contract_min_max import contract_min_max
# from model_list import alexnet, alexnet1d

def to_unix(t):
    timearray = time.strptime(t, '%Y-%m-%d %H:%M:%S')
    tt = int(time.mktime(timearray))
    return tt

def timestamp(t):
    format = '%Y-%m-%d %H:%M:%S'
    t = int(t / 1e9)
    t = time.localtime(t)
    dt = time.strftime(format, t)
    return dt

def find_target(symbol, timestr, period):
    
    # if period == 10:
    #     history = 40
    # elif period > 10:
    #     history = 20
    history = 40
    df = api.get_kline_serial(symbol, period * 60, 5000)
    df = df.dropna()
    df.loc[:, "time"] = [timestamp(t) for t in df["datetime"]]
    df.loc[:, "macd"] = MACD(df, *ma_para)["bar"]
    df.loc[:, "cci"] = CCI(df, 14) # 14 days CCI indicator
    df.loc[:, "rsi"] = RSI(df, 7) # 7 days RSI
    
    print(f"fartest datetime acquired for {period}min is at {df['time'].iloc[0]}")
    target_index = df.loc[df["time"] == timestr].index[0]
    target_values = df[["macd", "close", "cci", "rsi"]].loc[target_index - history + 1:target_index]

    return target_values
    
def prepare_data(symbol, time_list, label, exam=False):
    temp_pd = pd.DataFrame()
    for item in time_list:
        try:
            target_values = find_target(symbol, item[0], item[1])
            temp_pd[f"macd{item[1]}"], temp_pd[f"close{item[1]}"] = list(target_values.loc[:, "macd"]), list(target_values.loc[:, "close"]) 
        except:
            print(f"{item[1]} is problematic")
    
    # add cci10 and rsi10 at the end of dataframe
    for item in time_list:
        if int(item[1]) == 10:
            target_values = find_target(symbol, item[0], item[1])
            temp_pd[f"cci{item[1]}"], temp_pd[f"rsi{item[1]}"] = list(target_values.loc[:, "cci"] / 200.), list(target_values.loc[:, "rsi"] / 100.) # scale cci by 200, rsi by 100
            break
    
    # another way to add cci10 and rsi10 to the end of datafram
    # cci_temp = temp_pd.pop("cci10")
    # rsi_temp = temp_pd.pop("rsi10")
    # temp_pd.insert(8, "cci10", cci_temp)
    # temp_pd.insert(9, "rsi10", rsi_temp)

    # for examination
    if exam:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', constrained_layout=True)
        ax1.bar(range(40), temp_pd["macd10"], width=0.5)
        ax1.set_title(f'macd 10min at {time_list[0][0]}, mark: {label}')
        ax2.bar(range(40), temp_pd["macd15"], width=0.5)
        ax2.set_title(f'macd 15min at {time_list[1][0]}, mark: {label}')
        ax3.bar(range(40), temp_pd["macd60"], width=0.5)
        ax3.set_title(f'macd 60 at {time_list[2][0]}, mark: {label}')
        ax4.bar(range(40), temp_pd["macd1440"], width=0.5)
        ax4.set_title(f'macd day at {time_list[3][0]}, mark: {label}')
        plt.show()
    # end examination

    return temp_pd, torch.unsqueeze(torch.from_numpy(temp_pd.values), 0), torch.tensor([label])

def init_dataset(saved_time_list, exam=0):
    data_tot = torch.zeros(1, 40, 10) # initialize data_tot
    label_tot = torch.tensor([0.]) # initialize label_tot
    for item in saved_time_list:
        _, data, label = prepare_data(item[0], item[1], item[2], exam)
        data_tot = torch.cat((data_tot, data),dim=0)
        label_tot = torch.cat((label_tot, label))
    return data_tot[1:], label_tot[1:]  # exclude initial zero data        

# find the min and max close value for every symbol in data_tot, and use them to scale data_tot
def scale_data(time_list, data_tot, contract_min_max=contract_min_max):
    local_data = deepcopy(data_tot)

    for ii, item in enumerate(time_list):
        for contract in contract_min_max:
            if contract[0] in item[0]:
                local_data[ii, :, 1:8:2] = (local_data[ii, :, 1:8:2] - contract[1]) / (contract[2] - contract[1])
                break
        else:
            print(f"\n{item[0]} is not in contract_list! Scale it manually or add contract to contract_min_max.py.\n")
    return contract_min_max, local_data

def fig_to_mat(data, size=(320,320), lw=7.0):
    fig, ax = plt.subplots(1, 1)
    ax.plot(data, linewidth=lw)
    plt.axis('off')
    plt.margins(0, 0)
    # plt.show()
    buffer_ = BytesIO() # request cache from memory instead of disk
    plt.savefig(buffer_, format='png', bbox_inches='tight')
    buffer_.seek(0)
    pildata = Image.open(buffer_)
    pildata = pildata.resize(size) # resize image
    pildata = pildata.convert('1') # black and white mode
    pildata = ImageOps.invert(pildata) # invert True and False
    img_data = np.asarray(pildata, dtype=np.float32) # tranform True False to 1 0
    plt.close()
    buffer_.close() # release caches
    return pildata, torch.from_numpy(img_data) # return image and matrix

def read_data(source, date):

    # date = datetime.strftime(datetime.now(), "%Y-%m-%d")
    if source == 'tqsdk': # not available in jupyter
        global api
        api = TqApi(TqKq(), auth=TqAuth("", ""))
        print(f"\nlength of saved_time_list is: {len(saved_time_list)}\n")
        data_tot, label_tot = init_dataset(saved_time_list, exam=0)
        load_data = TensorDataset(data_tot.to(torch.float32), label_tot)
        torch.save(load_data, f"./data/macd_datasets_{date}.pt") # save data to file
        api.close()
    else:
        load_data = torch.load(f"./data/macd_datasets_{date}.pt")
        print(f"\nshape of loaded data is: {load_data.tensors[0].shape}\n")
    return load_data

def prepare_img_datasets(load_data, data_dim=2, need_symbol_list=False):
    target_dataset = load_data.tensors[0].transpose(1,2)

    if data_dim == 2:
        print("\ntransforming to 2D data")
        ## 转为二维图片
        img_size = (300, 300)
        linewidth = 9
        shape_of_datasets = target_dataset.shape
        img_datasets = torch.zeros(shape_of_datasets[0], shape_of_datasets[1], img_size[0], img_size[1])
        for i in range(img_datasets.shape[0]):
            for j in range(img_datasets.shape[1]):
                _, img_datasets[i, j] = fig_to_mat(target_dataset[i, j], size=img_size, lw=linewidth)

        print(f"\nshape of datasets is: {img_datasets.shape} \n")
        plt.imshow(img_datasets[0,1]) # show a picture
        return img_datasets
    else:
        print("\nUse 1d dataset")
        symbol_list, img_datasets = scale_data(saved_time_list, target_dataset.transpose(1,2))
        img_datasets = img_datasets.transpose(1,2)
        print(f"\nshape of datasets is: {img_datasets.shape} \n")
        if need_symbol_list:
            return symbol_list, img_datasets
        else:
            return img_datasets


def prepare_dataloader(img_datasets, target_label, bs=32, label_type='binary', test=True, one_hot_label=False):
    # build dataloader
    scaled_label = target_label / 10.0
    binary_label = torch.tensor((scaled_label > 0), dtype=torch.float32) # 只判断macd信号对错不考虑幅值    
    
    if label_type == "scaled":
        print("\nUse scaled label \n")
        datasets_trans = TensorDataset(img_datasets, torch.unsqueeze(scaled_label, dim=1)) # 使用scaled label
    elif label_type == "binary":
        print("\nUse binary label \n")
        if one_hot_label:
            print("Enable one-hot encoding")
            datasets_trans = TensorDataset(img_datasets, F.one_hot(torch.tensor([int(i) for i in binary_label])).to(torch.float32)) # 使用binary label并进行one-hot编码
        else:
            print("Disable one-hot encoding")
            datasets_trans = TensorDataset(img_datasets, torch.unsqueeze(binary_label, dim=1)) # 使用binary label不进行one-hot编码

    if test: # add test loader
        print("\nenable test loader")
        train_data_size = int(len(img_datasets) * 0.7) # 70%
        val_data_size = int((len(img_datasets) - train_data_size) / 2) # 15%
        test_data_size = len(img_datasets) - train_data_size - val_data_size # 15% 

        train_dataset, val_dataset, test_dataset = random_split(datasets_trans, [train_data_size, val_data_size, test_data_size])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size = bs,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=len(val_dataset),
            shuffle=False
        )    
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=len(test_dataset),
            shuffle=False
        )
        return train_loader, val_loader, test_loader
    else: # use only train and val loader
        print("\nno test loader")
        train_data_size = int(len(img_datasets) * 0.8) # 80%
        val_data_size = int((len(img_datasets) - train_data_size)) # 20%

        train_dataset, val_dataset = random_split(datasets_trans, [train_data_size, val_data_size])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size = bs,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=len(val_dataset),
            shuffle=False
        )    
        return train_loader, val_loader

## build model
class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25, gamma=2.0,use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(),1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label= label[mask]
        focal_weight = self.alpha *(label- pred).abs().pow(self.gamma) * (label> 0.0).float() + (1 - self.alpha) * pred.abs().pow(self.gamma) * (label<= 0.0).float()
        loss = F.binary_cross_entropy(pred, label, reduction='none') * focal_weight
        return loss.sum()/pos_num

class alexnet1d(nn.Module):
    def __init__(self, input_channels=8, out_dim=1, first_kernel=7):
        super(alexnet1d, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=first_kernel, stride=2, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(2048), nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(out_dim),
        )
    def forward(self, x):
        y = self.model(x)
        return y

if __name__ == '__main__':

    pl.seed_everything(42) # secure random seed
    load_file_date = "2023-04-28-price"
    load_data = read_data("file", date=load_file_date)
    if 'price' in load_file_date:
        print("Predict price\n")
        saved_time_list = saved_time_list_price[:len(load_data.tensors[1])] # 限制saved_time_list的长度，以免添加time list时对NNI中的程序造成影响
    else:
        saved_time_list = saved_time_list[:len(load_data.tensors[1])]
    print(saved_time_list[26][2])
    print('\n')
    nni_params = {
        "channels": [0,2,4,6,8,9],
        "batch_size": 64,
        "threshold": 0.8,
        "patient": 6,
        "first_kernel": 7,
    }

    optimized_params = nni.get_next_parameter()
    nni_params.update(optimized_params)

    trial_num = nni.get_sequence_id()
    exp_id = nni.get_experiment_id()
    print(nni_params, '\n', f"Exp: {exp_id}: no.{trial_num}\n")
    ## macd and atr parameters
    ma_para = [13,34,9]
    atr_para = [13]

    data_dim = 1 # use 1d data and model
    need_symbol_list = False
    if data_dim == 1 and need_symbol_list:
        symbol_list, img_datasets = prepare_img_datasets(load_data, data_dim=data_dim, need_symbol_list=True)
    else:
        img_datasets = prepare_img_datasets(load_data, data_dim=data_dim)
    img_datasets = F.normalize(img_datasets,p=2,dim=2) # normalize data
    batch_size = int(nni_params["batch_size"])
    # label_type = nni_params["label_type"] # binary or scaled, default binary 
    label_type = "binary"
    # one_hot_label = True if nni_params["one_hot_label"] else False
    one_hot_label = True
    # train_loader, test_loader = prepare_dataloader(img_datasets, load_data.tensors[1], batch_size, label_type, test=False, one_hot_label=one_hot_label)
    channels = nni_params["channels"] #  channels of data that will be used
    first_kernel = int(nni_params["first_kernel"])
    ## compare models
    max_trial = 200 if data_dim == 1 else 50
    trial_acc_list = []
    best_model = [0, 0]
    for trial in range(max_trial):
        train_loader, test_loader = prepare_dataloader(img_datasets, load_data.tensors[1], batch_size, label_type, test=False, one_hot_label=one_hot_label) # use different dataloader for every trial
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        used_net = "alexnet1d"
        out_dim = 2 if label_type == "binary" and one_hot_label else 1
        net = alexnet1d(len(channels), out_dim, first_kernel).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss() if one_hot_label else nn.MSELoss()
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = FocalLoss()
        scheduler = lrs.StepLR(optimizer, step_size=5, gamma=0.85)
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        patient = int(nni_params["patient"]) # 训练达到一定精度之后，经过多少epoch停止训练
        tolerance = 0
        threshold = nni_params["threshold"]
        max_epochs = 200
        for epoch in range(max_epochs):
            net.train()
            for batch, (data, label) in enumerate(train_loader):
                data = data[:,channels].to(device)
                label = label.to(device)

                logits = net(data)
                loss = criterion(logits, label)
                train_loss.append(loss.item() / len(data))

                if label_type == 'binary':
                    if one_hot_label: 
                        acc = logits.max(1)[1].eq(label.max(1)[1]).sum().item() / len(label)
                    else:
                        acc = ((logits.flatten() > 0.5 ).flatten() == label.flatten()).sum().item() / len(label)
                else:
                    count = 0
                    for item in (torch.cat((logits, label), 1) > 0):
                        if item[0] == item[1]:
                            count += 1
                    acc = count / len(label)
                train_acc.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if batch % 2 == 0:
                    # print(f"Train Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.6f}, {batch * len(data)}/{len(train_loader.dataset)}, avg loss: {loss.item() / len(data):.6f}, train_acc: {train_acc[-1]:.2f}")
            scheduler.step() # update learning rate after an epoch

            with torch.no_grad():
                net.eval()
                for data, label in test_loader:
                    data = data[:, channels].to(device)
                    label = label.to(device)
                    
                    logits = net(data)
                    loss = criterion(logits, label)

                    if label_type == 'binary':
                        if one_hot_label: 
                            acc = logits.max(1)[1].eq(label.max(1)[1]).sum().item() / len(label)
                        else:
                            acc = ((logits.flatten() > 0.5 ).flatten() == label.flatten()).sum().item() / len(label)
                    else:
                        count = 0
                        for item in (torch.cat((logits, label), 1) > 0):
                            if item[0] == item[1]:
                                count += 1
                        acc = count / len(label)
                    test_acc.append(acc)
                    # print(((net(data).flatten() > 0.5 ) == label.flatten()).sum() / len(data))
                    test_loss.append(loss.item() / len(data))
                # print(f"{used_net}, trial.{trial}:\nEpoch {epoch}, avg test_loss: {loss.item() / len(data):.6f}, test_acc: {acc:.2f}")
            if train_acc[-1] >= threshold: # early stop
                tolerance += 1
                # print(f"\ntrain data has been fully fitted. Stop training process in 5 epochs. tolerance = {tolerance}")
                if tolerance >= patient:
                    print(f"\nTraining stop at epoch: {epoch} with tolerance: {tolerance}\nlast train_acc: {train_acc[-1]:.3f}")
                    break
            
        # torch.save(net.state_dict(), f"e:\\ml_data\\model_checkpoints\\{used_net}_{today}_{len(saved_time_list)}_{int(test_acc[-1] * 100)}_{channels}_{label_type}.pth")
        
        # show information about the dataloader
        with torch.no_grad():
            net = net.to('cpu')
            net.eval()
            for data, label in test_loader:
                if one_hot_label:
                    result = torch.stack((net(data[:, channels]).max(1)[1], label.max(1)[1]), 1)
                else:
                    result = torch.cat((torch.tensor(net(data[:, channels]) > 0.5, dtype=torch.float32), label),1)
                pos_corr = 0
                neg_corr = 0
                for res in result:
                    if res[0] == res[1] and res[1] == 1.:
                        pos_corr += 1
                    elif res[0] == res[1] and res[1] == 0.:
                        neg_corr += 1
                print(f"\nIn {len(result)} test instance:\ncorrect signal: {sum([item[1] == 1 for item in result])}\nwrong signal: {sum([item[1] == 0 for item in result])}\nright ratio about correct signal: {round(pos_corr/sum([item[1] == 1 for item in result]).item(), 3)}\nright ratio about wrong signal: {round(neg_corr/sum([item[1] == 0 for item in result]).item(),3)}\ntotal correct: {sum([item[0] == item[1] for item in result])/len(result):.2f}")

        nni.report_intermediate_result(round(test_acc[-1], 5))
        trial_acc_list.append(test_acc[-1])
        if test_acc[-1] > best_model[0]: # find best results
            best_model[0] = test_acc[-1]
            best_model[1] = net.state_dict()
        print(f"\nCurrent best model has acc: {best_model[0]:.3f}")
        # torch.cuda.empty_cache() # clear gpu cache at the end of every trial
    # report the avg acc of the best 15 trials
    
    if int(trial_num) >= 200 and int(trial_num) < 300: # save 100 best models after 200 trial
        if 'price' in load_file_date:
            torch.save({
                "channels": channels,
                "batch_size": batch_size,
                "threshold": threshold,
                "patient": patient,
                "first_kernel": first_kernel,
                "best_model": best_model[1],
                "acc": round(best_model[0], 6),
            }, f"/home/fxw/Documents/ml_data/model_checkpoints/model_nni_{exp_id}_price_{int(trial_num)-200}.pth")
        else:
            torch.save({
                "channels": channels,
                "batch_size": batch_size,
                "threshold": threshold,
                "patient": patient,
                "first_kernel": first_kernel,
                "best_model": best_model[1],
                "acc": round(best_model[0], 6),
            }, f"/home/fxw/Documents/ml_data/model_checkpoints/model_nni_{exp_id}_{int(trial_num)-200}.pth")


    nni.report_final_result(round(sum(sorted(trial_acc_list, reverse=True)[:15])/15, 6))