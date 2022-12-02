#coding=UTF-8

import numpy as np 
import pandas as pd 
import os, gc, re, warnings
warnings.filterwarnings("ignore")

dftr = pd.read_csv("./data/train.csv")
dftr["src"]="train"
dfte = pd.read_csv("./data/test.csv___orginal")
dfte["src"]="test"
print('Train shape:',dftr.shape,'Test shape:',dfte.shape,'Test columns:',dfte.columns)
#df = pd.concat([dftr,dfte],ignore_index=True)
print(dftr.head())

target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
import sys
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
FOLDS = 25
skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
for i,(train_index, val_index) in enumerate(skf.split(dftr,dftr[target_cols])):
    dftr.loc[val_index,'FOLD'] = i
print('Train samples per fold:')
dftr.FOLD.value_counts()

from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

tokenizer = None
MAX_LEN = 1000
BATCH_SIZE = 48
class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]
        tokens = tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens

ds_tr = EmbedDataset(dftr)
embed_dataloader_tr = torch.utils.data.DataLoader(ds_tr,\
                        num_workers=32, \
                        batch_size=BATCH_SIZE,\
                        shuffle=False)
ds_te = EmbedDataset(dfte)
embed_dataloader_te = torch.utils.data.DataLoader(ds_te,\
                        num_workers=32, \
                        batch_size=BATCH_SIZE,\
                        shuffle=False)


def get_embeddings(MODEL_NM='', MAX=640, BATCH=4, verbose=True):
    global tokenizer, MAX_LEN #BATCH_SIZE
    DEVICE="cuda"
    model = AutoModel.from_pretrained( MODEL_NM )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NM, use_fast=False)
    MAX_LEN = MAX
    #BATCH_SIZE = BATCH
    
    model = model.to(DEVICE)
    model.eval()
    all_train_text_feats = []
    for batch in tqdm(embed_dataloader_tr,total=len(embed_dataloader_tr)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        all_train_text_feats.extend(sentence_embeddings)
    all_train_text_feats = np.array(all_train_text_feats)
    if verbose:
        print('Train embeddings shape',all_train_text_feats.shape)
        
    te_text_feats = []
    for batch in tqdm(embed_dataloader_te,total=len(embed_dataloader_te)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        te_text_feats.extend(sentence_embeddings)
    te_text_feats = np.array(te_text_feats)
    if verbose:
        print('Test embeddings shape',te_text_feats.shape)
        
    return all_train_text_feats, te_text_feats


MODEL_NM = 'model/huggingface-bert/roberta-large'
all_train_text_feats3, te_text_feats3 = get_embeddings(MODEL_NM, MAX=512)

MODEL_NM = 'model/huggingface-bert/deberta-v3-large'
all_train_text_feats2, te_text_feats2 = get_embeddings(MODEL_NM, MAX=MAX_LEN)

MODEL_NM = 'model/huggingface-bert/deberta-v3-base'
all_train_text_feats, te_text_feats = get_embeddings(MODEL_NM, MAX=MAX_LEN)

all_train_text_feats = np.concatenate([all_train_text_feats,all_train_text_feats2,
                                       #all_train_text_feats3,all_train_text_feats4,all_train_text_feats5,
                                       all_train_text_feats3],axis=1)

te_text_feats = np.concatenate([te_text_feats,te_text_feats2,
                                #te_text_feats3,te_text_feats4,te_text_feats5,
                                te_text_feats3],axis=1)

del all_train_text_feats2, te_text_feats2
del all_train_text_feats3, te_text_feats3
#del all_train_text_feats4, te_text_feats4
#del all_train_text_feats5, te_text_feats5
gc.collect()

print('Our concatenated embeddings have shape', all_train_text_feats.shape )

from cuml.svm import SVR
import cuml
print('RAPIDS version',cuml.__version__)

from sklearn.metrics import mean_squared_error

preds = []
scores = []
def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(target_cols)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)

#for fold in tqdm(range(FOLDS),total=FOLDS):
for fold in range(FOLDS):
    print('#'*25)
    print('### Fold',fold+1)
    print('#'*25)
    
    dftr_ = dftr[dftr["FOLD"]!=fold]
    dfev_ = dftr[dftr["FOLD"]==fold]
    
    tr_text_feats = all_train_text_feats[list(dftr_.index),:]
    ev_text_feats = all_train_text_feats[list(dfev_.index),:]
    
    ev_preds = np.zeros((len(ev_text_feats),6))
    test_preds = np.zeros((len(te_text_feats),6))
    for i,t in enumerate(target_cols):
        print(t,', ',end='')
        clf = SVR(C=1)
        clf.fit(tr_text_feats, dftr_[t].values)
        ev_preds[:,i] = clf.predict(ev_text_feats)
        test_preds[:,i] = clf.predict(te_text_feats)
    print()
    score = comp_score(dfev_[target_cols].values,ev_preds)
    scores.append(score)
    print("Fold : {} RSME score: {}".format(fold,score))
    preds.append(test_preds)
    
print('#'*25)
print('Overall CV RSME =',np.mean(scores))


sub = dfte.copy()

sub.loc[:,target_cols] = np.average(np.array(preds),axis=0) #,weights=[1/s for s in scores]
sub_columns = pd.read_csv("./data/sample_submission.csv").columns
sub = sub[sub_columns]
sub.to_csv("submission.csv",index=None)
sub.head()


