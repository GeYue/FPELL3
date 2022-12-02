#coding=UTF-8

import numpy as np 
import pandas as pd 
import os, gc, re, warnings, glob
warnings.filterwarnings("ignore")

GPU = os.environ['CUDA_VISIBLE_DEVICES']

dftr = pd.read_csv("./data/train.csv")
dftr["src"]="train"
dfte = pd.read_csv("./data/test.csv___orginal")
#dfte = pd.read_csv("./data/train.csv")
dfte["src"]="test"
print('Train shape:',dftr.shape,'Test shape:',dfte.shape,'Test columns:',dfte.columns)
#df = pd.concat([dftr,dfte],ignore_index=True)
print(dftr.head())

target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
import sys
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
FOLDS = 4
skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1979)
for i,(train_index, val_index) in enumerate(skf.split(dftr,dftr[target_cols])):
    dftr.loc[val_index,'FOLD'] = i
print('Train samples per fold:')
print(dftr.FOLD.value_counts())


skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=2022)
for i,(train_index, val_index) in enumerate(skf.split(dftr,dftr[target_cols])):
    dftr.loc[val_index,'FOLD2'] = i
print('Train samples per fold:')
print(dftr.FOLD2.value_counts())

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
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
BATCH_SIZE = 16
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


class BASE_CFG():
    def __init__(self, path):
        #model = ""
        self.gradient_checkpointing=False
        self.BASE_PATH = path
        self.target_cols = target_cols
        #self.files = sorted(glob.glob(f"{self.BASE_PATH}/*/*gpu{GPU}*best*.pth"))

class CFG1(BASE_CFG):
    def __init__(self, gpu, fold):
        super().__init__(f'kaggle/offical/fpell-debertav3-base-reini-lrdec-pl-00')
        self.MODEL_NM = 'model/huggingface-bert/deberta-v3-base'
        self.files = sorted(glob.glob(f"{self.BASE_PATH}/*/*gpu{gpu}_fold{fold}*best*.pth"))

class CFG2(BASE_CFG):
    def __init__(self, gpu, fold):
        super().__init__(f'kaggle/offical/fpell-debertav3-large-reini-lrdec-pl-00')
        self.MODEL_NM = 'model/huggingface-bert/deberta-v3-large'
        self.files = sorted(glob.glob(f"{self.BASE_PATH}/*/*gpu{gpu}_fold{fold}*best*.pth"))

class CFG3(BASE_CFG):
    def __init__(self, gpu, fold):
        super().__init__(f'kaggle/offical/fpell-debertav2-xlarge-ver2-pl-00')
        self.MODEL_NM = 'model/huggingface-bert/deberta-v2-xlarge'
        self.files = sorted(glob.glob(f"{self.BASE_PATH}/*/*gpu{gpu}_fold{fold}*best*.pth"))

class CFG4(BASE_CFG):
    def __init__(self, gpu, fold):
        super().__init__(f'kaggle/DebertaV2-XXLarge-FixedV3Re2-NoLrDec-PL')
        self.MODEL_NM = 'model/huggingface-bert/deberta-v2-xxlarge'
        self.files = sorted(glob.glob(f"{self.BASE_PATH}/*gpu{gpu}_fold{fold}*best*.pth"))

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

LOAD_SVR_FROM_PATH = f"./kaggle/upload_cache" #None
def get_embeddings(model_cfg=None, PATH='', MAX=640, BATCH=4, verbose=True):
    global tokenizer, MAX_LEN #BATCH_SIZE
    DEVICE="cuda"
    model = CustomModel(model_cfg, config_path=model_cfg.MODEL_NM+'/config.json', pretrained=False)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.MODEL_NM, use_fast=False)
    MAX_LEN = MAX
    #BATCH_SIZE = BATCH
    
    model = model.to(DEVICE)
    model.eval()

    state = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])

    all_train_text_feats = []
    if LOAD_SVR_FROM_PATH is None:
        for batch in tqdm(embed_dataloader_tr,total=len(embed_dataloader_tr)):
            attention_mask = batch["attention_mask"]
            for k, v in batch.items():
                batch[k] = v.to(DEVICE)
            with torch.no_grad():
                model_output = model.feature(batch) #model(input_ids=input_ids,attention_mask=attention_mask)
            #sentence_embeddings = mean_pooling(model_output, attention_mask)

            # Normalize the embeddings
            sentence_embeddings = F.normalize(model_output, p=2, dim=1) #F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = sentence_embeddings.squeeze(0).detach().cpu().numpy()

            all_train_text_feats.extend(sentence_embeddings)
        all_train_text_feats = np.array(all_train_text_feats)
        np.save(f"{PATH.split('/')[-1].split('.')[0]}.npy", all_train_text_feats)
        if verbose:
            print(f'file::{PATH} / Train embeddings shape',all_train_text_feats.shape)
    else:
        all_train_text_feats = np.load(f"{LOAD_SVR_FROM_PATH}/{PATH.split('/')[-1].split('.')[0]}.npy")
        print (f"Fetched embedding from {LOAD_SVR_FROM_PATH}/{PATH.split('/')[-1].split('.')[0]}.npy, shape=={all_train_text_feats.shape}.")      

    te_text_feats = []
    for batch in tqdm(embed_dataloader_te,total=len(embed_dataloader_te)):
        attention_mask = batch["attention_mask"]
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
        with torch.no_grad():
            model_output = model.feature(batch) #model(input_ids=input_ids,attention_mask=attention_mask)
        #sentence_embeddings = mean_pooling(model_output, attention_mask)
        
        # Normalize the embeddings
        sentence_embeddings = F.normalize(model_output, p=2, dim=1) #F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.squeeze(0).detach().cpu().numpy()
        
        te_text_feats.extend(sentence_embeddings)
    te_text_feats = np.array(te_text_feats)
    if verbose:
        print('Test embeddings shape',te_text_feats.shape)
        
    return all_train_text_feats, te_text_feats

from cuml.svm import SVR
import cuml
print('RAPIDS version',cuml.__version__)

from sklearn.metrics import mean_squared_error

def run_SVR_on_features(CFG_list, file_index):

    ALL_Train_Text_Feats = []
    ALL_Te_Text_Feats = []
    for cfg in CFG_list:
        for file in cfg.files:
            print (f"file---->{file}")
            train_text_feats, te_text_feats = get_embeddings(model_cfg=cfg, PATH=f"{file}", MAX=1000)

            if len(ALL_Train_Text_Feats) > 0:
                ALL_Train_Text_Feats = np.concatenate([ALL_Train_Text_Feats, train_text_feats], axis=1)
                ALL_Te_Text_Feats = np.concatenate([ALL_Te_Text_Feats, te_text_feats], axis=1)
            else:
                ALL_Train_Text_Feats = train_text_feats
                ALL_Te_Text_Feats = te_text_feats

            del train_text_feats, te_text_feats
            gc.collect()

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
        
        tr_text_feats = ALL_Train_Text_Feats[list(dftr_.index),:]
        ev_text_feats = ALL_Train_Text_Feats[list(dfev_.index),:]

        ev_preds = np.zeros((len(ev_text_feats),6))
        test_preds = np.zeros((len(ALL_Te_Text_Feats),6))

        for i,t in enumerate(target_cols):
            print(t,', ',end='')
            clf = SVR(C=1)
            clf.fit(tr_text_feats, dftr_[t].values)
            ev_preds[:,i] = clf.predict(ev_text_feats)
            test_preds[:,i] = clf.predict(ALL_Te_Text_Feats)
        print()
        score = comp_score(dfev_[target_cols].values, ev_preds)
        scores.append(score)
        print("Fold : {} RSME score: {}".format(fold,score))
        preds.append(test_preds)
        
    print('#'*25)
    print('Overall CV RSME =',np.mean(scores))

    del ALL_Train_Text_Feats, ALL_Te_Text_Feats
    gc.collect()

    sub = dfte.copy()

    #sub.loc[:,target_cols] = np.average(np.array(preds.tolist()),axis=0) #,weights=[1/s for s in scores]
    sub.loc[:,target_cols] = np.average(np.array(preds), axis=0) #,weights=[1/s for s in scores]

    sub_columns = pd.read_csv("./data/sample_submission.csv").columns
    sub = sub[sub_columns]

    sub.to_csv(f"submission_{file_index}.csv",index=None)
    print (sub.head())


CFG_List1 = [CFG4(gpu=0, fold=0), CFG3(gpu=0, fold=0), CFG2(gpu=0, fold=0), CFG1(gpu=0, fold=0)]
weight_1 = 1
run_SVR_on_features(CFG_List1, file_index=1)

CFG_List2 = [CFG4(gpu=0, fold=1), CFG3(gpu=0, fold=1), CFG2(gpu=0, fold=1), CFG1(gpu=0, fold=1)]
weight_2 = 1
run_SVR_on_features(CFG_List2, file_index=2)

cfg_comm = CFG1(gpu=0, fold=0)
sub1 = pd.read_csv(f'submission_1.csv')[cfg_comm.target_cols] * weight_1
sub2 = pd.read_csv(f'submission_2.csv')[cfg_comm.target_cols] * weight_2

ens = (sub1 + sub2)/(weight_1 + weight_2)

submission = pd.read_csv("./data/sample_submission.csv___orginal") #('../input/feedback-prize-english-language-learning/sample_submission.csv')

submission[cfg_comm.target_cols] = ens
print(submission.head())
submission.to_csv('submission.csv', index=False)

#sub.to_csv("submission.csv",index=None)
print (f"in the End!")
print (submission.head())

