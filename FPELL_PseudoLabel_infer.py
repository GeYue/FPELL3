#coding=UTF-8

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding

os.environ['TOKENIZERS_PARALLELISM']='false'
GPU = os.environ['CUDA_VISIBLE_DEVICES']
os.environ['TOKENIZERS_PARALLELISM']='true'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BASE_CFG:    
    model = ""
    path = ""
    base = ""
    config_path = ""
    
    gradient_checkpointing=False
    batch_size=8
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=4
    trn_fold=list(range(n_fold))
    num_workers=4
    weight = 1.0

class CFG1(BASE_CFG):
    model = "microsoft/deberta-v3-base"
    path = "kaggle/043/rumor03/"
    base = "model/huggingface-bert/deberta-v3-base/" 
    config_path = base + "config.json"
    tokenizer = AutoTokenizer.from_pretrained(base)
    batch_size=24
    max_len = 768

    
    
class CFG2(BASE_CFG):
    model = "microsoft/deberta-v3-large"
    path = "kaggle/DebertaV3Large-FixedV3-Re1-NoLrDec/"
    base = "model/huggingface-bert/deberta-v3-large/"
    config_path = base + "config.json"
    tokenizer = AutoTokenizer.from_pretrained(base)
    batch_size=16
    max_len = 1000
  
    
class CFG3(BASE_CFG):
    model = "microsoft/deberta-v2-xlarge"
    path = "kaggle/DebertaV2XLarge-FixedReV3-LrDecWithout/"
    base = "model/huggingface-bert/deberta-v2-xlarge/"
    config_path = base + "config.json"    
    tokenizer = AutoTokenizer.from_pretrained(base)
    batch_size=8
    max_len = 1000


class CFG4(BASE_CFG):
    model = "roberta-large"
    path = "kaggle/model_test/Roberta-Large/"
    base = "model/huggingface-bert/roberta-large/"
    config_path = base + "config.json"    
    tokenizer = AutoTokenizer.from_pretrained(base)
    batch_size=24
    max_len = 512
    
class CFG5(BASE_CFG):
    model = "microsoft/deberta-v2-xxlarge"
    path = "kaggle/DebertaV2-XXLarge-FixedV3Re2-NoLrDec/"
    base = "model/huggingface-bert/deberta-v2-xxlarge/"
    config_path = base + "config.json"    
    tokenizer = AutoTokenizer.from_pretrained(base)
    batch_size=12
    max_len = 1000
    
class CFG6(BASE_CFG):
    model = "microsoft/deberta-v2-xlarge-mnli"
    path = "kaggle/DebertaV2XLarge-MNLI-FixedV3-Re1-noLrDec/"
    base = "model/huggingface-bert/deberta-v2-xlarge/"
    config_path = base + "config.json"    
    tokenizer = AutoTokenizer.from_pretrained(base)
    batch_size=12
    max_len = 1000

#CFG_list = [CFG1, CFG2, CFG3]
CFG_list = [CFG1]

# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

def get_logger(filename='inference'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

# ====================================================
# oof
# ====================================================
# for CFG in CFG_list:
#     oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
#     labels = oof_df[CFG.target_cols].values
#     preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
#     score, scores = get_score(labels, preds)
#     LOGGER.info(f'Model: {CFG.model} Score: {score:<.4f}  Scores: {scores}')


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.max_len,
        #pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    
class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings
        

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = AutoConfig.from_pretrained(config_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

import glob
for _idx, CFG in enumerate(CFG_list):
    test = pd.read_csv('./data/test.csv')
    submission = pd.read_csv('./data/sample_submission.csv')
    # sort by length to speed up inference
    test['tokenize_length'] = [len(CFG.tokenizer(text)['input_ids']) for text in test['full_text'].values]
    test = test.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
                             #num_workers=0, pin_memory=True, drop_last=False)
    predictions = []
    files = sorted(glob.glob(f"{CFG.path}*gpu{GPU}*best*.pth"))
    #for fold in CFG.trn_fold:
    for file in files:
        model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
        print (f"Loading {file} ... ...")
        state = torch.load(file, #CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                           map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        #predictions.append(prediction)

        test[CFG.target_cols] = prediction
        submission = submission.drop(columns=CFG.target_cols).merge(test[['text_id'] + CFG.target_cols], on='text_id', how='left')
        print(submission.head())
        submission[['text_id'] + CFG.target_cols].to_csv(f'submission_{os.path.basename(file).split(".")[0]}.csv', index=False)

        del model, state, prediction; gc.collect()
        torch.cuda.empty_cache()


