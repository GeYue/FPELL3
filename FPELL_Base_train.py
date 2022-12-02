#coding=UTF-8

import ast, re, gc, copy, sys, random, os, math, time
from itertools import chain
import scipy as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from tqdm.notebook import tqdm
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizerFast, RobertaModel, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import lr_scheduler as torch_lrs

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from colorama import Fore, Back, Style
red_  = Fore.RED
green_ = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

os.environ['TORCH_HOME'] = "/home/xyb/Lab/TorchModels"
GPU = os.environ['CUDA_VISIBLE_DEVICES']
os.environ['TOKENIZERS_PARALLELISM']='true'

BASE_URL = "./data"
OUTPUT_DIR = './'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ====================================================
# CFG
# ====================================================
class CFG:
    wandb=False
    competition='FB3'
    wandb_kernel='FPELL-Train'
    debug=False
    apex=True
    print_freq=40
    num_workers=52
    model='model/huggingface-bert/deberta-v2-xlarge' #"microsoft/deberta-v3-large" #"microsoft/deberta-v3-base" 
    gradient_checkpointing=True #True   ##for XLNet, should be False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5#5#4
    encoder_lr=2e-5 #2e-5 for others 2e-6 for DebertaV2 XL/XXL
    decoder_lr=2e-5 #2e-5 for others 2e-6 for DebertaV2 XL/XXL
    min_lr=1e-7 #1e-6/7 for others 1e-8 for DebertaV2 XL/XXL
    eps=1e-7
    betas=(0.9, 0.999)
    batch_size=24 #64 #24 #12
    max_len=768 #768 #1000 #512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=100 #1000

    fgm = True
    unscale = False
    reinit_n_layers=2
    layerwise_lr_decay=False
    layerwise_lr = 5e-5 #5e-5
    layerwise_weight_decay = 0.01
    layerwise_adam_epsilon = 1e-7
    lr_decay_cofficient=0.9756 # 0.9(12 Layers) 0.9508(24 Layers) 0.9756(48 Layers)

    pooling = 'mean' # mean, max, min, attention, attention_superior, weightedlayer, mixed, absurd,
    layer_start = 9 #9 

    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    train=True

if GPU == '0':
	CFG.seed = 1979
else:
	CFG.seed = 2022


### logger setting
import logging
logging.basicConfig(level=logging.INFO,
                    filename=f'output.log.gpu{GPU}',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s - %(levelname)s -:: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"logger started. 💭FPELL_Training on GPU{GPU}, KFold={CFG.n_fold} 🔴🟡🟢 {sys.argv} Seed={CFG.seed}")
logger.info(f"CAUTION::: max_len=={CFG.max_len}, apex={CFG.apex},\
	reLayer={CFG.reinit_n_layers}, layerwise_option={CFG.layerwise_lr_decay},\
	lr_decay_cofficient={CFG.lr_decay_cofficient}!!!!")

print(f"{red_}CAUTION{sr_}::: max_len=={red_}{CFG.max_len}{sr_}, apex={red_}{CFG.apex}{sr_},\
	reLayer={red_}{CFG.reinit_n_layers}{sr_}, layerwise_option={red_}{CFG.layerwise_lr_decay}{sr_},\
	lr_decay_cofficient={red_}{CFG.lr_decay_cofficient}{sr_}!!!!")

if CFG.wandb:
    import wandb
    try:
        wandb.login(key="67871c2e8f97fa74b52c18bcfccbee7fee0361d2")
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    def readCfgDict(f):
        return dict((name, f[name]) for name in f.keys() if not name.startswith('__'))
    
    run = wandb.init(project=CFG.wandb_kernel, 
                    #name="York_PPPM",
                    config=class2dict(CFG), #readCfgDict(CFG), #class2dict(CFG),
                    #group="DeBERTa-V3L",
                    #job_type="train",
                    )


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

# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv(f'{BASE_URL}/train.csv')
test = pd.read_csv(f'{BASE_URL}/test.csv')
submission = pd.read_csv(f'{BASE_URL}/sample_submission.csv')

# print(f"train.shape: {train.shape}")
# print(train.head())
# print(f"test.shape: {test.shape}")
# print(test.head())
# print(f"submission.shape: {submission.shape}")
# print(submission.head())

# ====================================================
# CV split
# ====================================================
Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
print(train.groupby('fold').size())


# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model, use_fast=True)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

# ====================================================
# Define max_len
# ====================================================
# lengths = []
# tk0 = tqdm(train['full_text'].fillna("").values, total=len(train))
# for text in tk0:
#     length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
#     lengths.append(length)
# CFG.max_len = max(lengths) + 2 # cls & sep (& sep ::should not include this last 'sep')
logger.info(f"max_len: {CFG.max_len}")


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        padding='max_length',
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


# ====================================================
# Model
# ====================================================

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon = 1., emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class AttentionPooling_Superior(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling_Superior, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        #self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t).float())
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht).float())

    def forward(self, all_hidden_states):
        # hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
        #                              for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        # hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)

        out = all_hidden_states[1:self.num_hidden_layers+1, :, 0, :]
        out = torch.stack([out[:,batch,:] for batch in range(out.shape[1])])

        out = self.attention(out) #(hidden_states)
        #out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = torch.nn.functional.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states, attention_mask):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_average.size()).float()
        sum_embeddings = torch.sum(weighted_average * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
        """


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
    
class MixedPooling(nn.Module):
    def __init__(self, config, layers_start, layer_weights):
        super(MixedPooling, self).__init__()
        self.layers_start = layers_start
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )
        self.meanpool = MeanPooling()
        
    def forward(self, all_hidden_states, attention_mask):
        stack_meanpool = torch.stack([self.meanpool(hidden_s, attention_mask) for hidden_s in all_hidden_states[self.layers_start:]], 
            axis=2)
        stack_meanpool = stack_meanpool.permute(2, 0, 1)
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).expand(stack_meanpool.size())
        weighted_average = (weight_factor*stack_meanpool).sum(dim=0) / self.layer_weights.sum()

        return weighted_average


class ScaleLayer(nn.Module):
   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale + 1.0

class AbsurdPooling(nn.Module):
    def __init__(self, config, layers_start):
        super(AbsurdPooling, self).__init__()
        self.layers_start = layers_start
        self.meanpool = MeanPooling()
        
        self.post_layer1 = nn.Linear(config.num_hidden_layers - layers_start + 1, 1)
        self.post_layer2 = nn.Linear(config.hidden_size, 6)
        self.scale_layer = ScaleLayer(init_value=4.0)

    def forward(self, all_hidden_states, attention_mask):
        stack_meanpool = torch.stack([self.meanpool(hidden_s, attention_mask) for hidden_s in all_hidden_states[self.layers_start:]], 
            axis=2)
        outputs = torch.softmax(stack_meanpool, axis=0)

        outputs = self.post_layer1(outputs)
        outputs = torch.squeeze(outputs, axis=-1)
        
        outputs = self.post_layer2(outputs)
        outputs = torch.sigmoid(outputs)
        outputs = self.scale_layer(outputs)

        return outputs


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
            logger.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.cfg.pooling == 'mean':
            self.pool = MeanPooling()
        # elif self.cfg.pooling == 'max':
        #     self.pool = MaxPooling()
        # elif self.pooling == 'min':
        #     self.pool = MinPooling()
        elif self.cfg.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif self.cfg.pooling == 'attention_sup':
            self.pool = AttentionPooling_Superior(self.config.num_hidden_layers, self.config.hidden_size, 6)
        elif self.cfg.pooling == 'weightedlayer':
            weights = [2**-i for i in range(0, self.config.num_hidden_layers-self.cfg.layer_start+1)]
            weights.reverse()
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, 
                layer_start = CFG.layer_start, 
                layer_weights = nn.Parameter(torch.tensor(weights))
                )        
        elif self.cfg.pooling == 'mixed':
            weights = [2**-i for i in range(0, self.config.num_hidden_layers-self.cfg.layer_start+1)]
            weights.reverse()
            self.pool = MixedPooling(self.config, self.cfg.layer_start, layer_weights=nn.Parameter(torch.tensor(weights)))
        elif self.cfg.pooling == "absurd":
            self.pool = AbsurdPooling(self.config, self.cfg.layer_start)

        #weighted_layer_started = math.floor(self.config.num_hidden_layers * 3 / 4) - 1
        #self.pool2 = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=weighted_layer_started, layer_weights=None)
        self.fc = nn.Linear(self.config.hidden_size, 6)

        ## ... code referenced from https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e#6196
        #self.reinit_n_layers = reinit_n_layers
        #if reinit_n_layers > 0: self._do_reinit()

        ## ... then modified it by myself to match this competition.
        if CFG.reinit_n_layers > 0: 
            self._do_reinit()
     
        self._init_weights(self.fc)

    def _do_reinit(self):
        # Re-init pooler.
        if "roberta" in self.config.model_type:
            self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            self.model.pooler.dense.bias.data.zero_()
            for param in self.model.pooler.parameters():
                param.requires_grad = True

        """
            ## Commented the below code, because it may be a mistake. 
            ## For Deberta models, the encoder.rel_embeddings/Laynorm may be the key parameters worked with all hidden layers.
            ## So re-initialized these layers may harm the whole model's performance.
        """
        # elif "deberta" in self.config.model_type: ## for Deberta V3
        #     self._init_weights(self.model.encoder.rel_embeddings)
        #     self._init_weights(self.model.encoder.LayerNorm)
        #     if hasattr(self.model.encoder.conv, "conv"): ## for Deberta V2
        #         self.model.encoder.conv.apply(self._init_weights)
        
        # Re-init last n layers.
        for n in range(CFG.reinit_n_layers):
            self.model.encoder.layer[-(n+1)].apply(self._init_weights)

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
        all_hidden_states = torch.stack(outputs['hidden_states'])

        if self.cfg.pooling == 'weightedlayer':
            feature = self.pool(all_hidden_states, inputs['attention_mask'])
            #feature = feature[:, 0]
        elif self.cfg.pooling == 'attention_sup':
            feature = self.pool(all_hidden_states)
        elif self.cfg.pooling == 'mixed':
            feature = self.pool(all_hidden_states, inputs['attention_mask'])
        elif self.cfg.pooling == 'absurd':
            feature = self.pool(all_hidden_states, inputs['attention_mask'])
        else:
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

        #weighted_pooling_embeddings = self.pool2(torch.stack(outputs['hidden_states']))
        ###weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        #weighted_pooling_embeddings = weighted_pooling_embeddings.mean(1)
        #return weighted_pooling_embeddings

    def forward(self, inputs):
        feature = self.feature(inputs)
        if self.cfg.pooling == 'attention_sup' or self.cfg.pooling == 'absurd':
            return feature
        else:
            output = self.fc(feature)
            return output


# ====================================================
# Loss
# ====================================================
class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    grad_trend = AverageMeter()
    start = end = time.time()
    global_step = 0

    if CFG.fgm:
        fgm = FGM(model)

    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        if CFG.unscale:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        grad_trend.update(grad_norm, 1)

        if CFG.fgm:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled = CFG.apex):
                y_preds = model(inputs)
                loss_adv = criterion(y_preds, labels)
                loss_adv.backward()
            fgm.restore()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_last_lr()[0]))
        if CFG.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] grad_norm": grad_trend.avg,
                       f"[fold{fold}] lr": scheduler.get_last_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    
    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+f'GPU{GPU}_config.pth')
    model.to(device)

    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
        #print(f'{idx}: {name}')

    layer_names.reverse()
    #print(layer_names[0:5])

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        named_parameters = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        if CFG.layerwise_lr_decay:
            opt_parameters = []
            init_lr = encoder_lr
            head_lr = encoder_lr
            lr = init_lr

            # === Final FC layer ======================================================   
            params_0 = [p for n,p in named_parameters if "fc" in n and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if "fc" in n and not any(nd in n for nd in no_decay)]
    
            head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
            opt_parameters.append(head_params)
        
            head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay}    
            opt_parameters.append(head_params)

            # === Pooler or Regessor (for Roberta) / Other top layers (for Deberta) ======================================================   
            if "roberta" in model.config.model_type:
                params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) and any(nd in n for nd in no_decay)]
                params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) and not any(nd in n for nd in no_decay)]

                head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
                opt_parameters.append(head_params)

                head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay}    
                opt_parameters.append(head_params)
            # else:
                # params_0 = [p for n,p in named_parameters if ("encoder" in n and "layer" not in n) and any(nd in n for nd in no_decay)]
                # params_1 = [p for n,p in named_parameters if ("encoder" in n and "layer" not in n) and not any(nd in n for nd in no_decay)]

                # head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
                # opt_parameters.append(head_params)

                # head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay}    
                # opt_parameters.append(head_params)

            #=== Model Hidden layers ==========================================================
            for layer in range(model.config.num_hidden_layers-1,-1,-1):        
                params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)]
                params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n and not any(nd in n for nd in no_decay)]
        
                layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
                opt_parameters.append(layer_params)   
                            
                layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
                opt_parameters.append(layer_params)       
        
                lr *= CFG.lr_decay_cofficient

            # === Model Embeddings layer ==========================================================
            params_0 = [p for n,p in named_parameters if ("embeddings" in n and "encoder" not in n) and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if ("embeddings" in n and "encoder" not in n) and not any(nd in n for nd in no_decay)]
    
            embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
            opt_parameters.append(embed_params)
        
            embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay} 
            opt_parameters.append(embed_params)
        else:
            opt_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],
                'lr': decoder_lr, 'weight_decay': 0.0}
            ]
        return opt_parameters

    def get_optimizer_grouped_parameters(model, layerwise_lr, layerwise_weight_decay, lr_decay_cofficient):
        
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                         "weight_decay": 0.0,
                                         "lr": layerwise_lr,
                                        },]
        # initialize lrs for every layer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = layerwise_lr
        for layer in layers:
            optimizer_grouped_parameters += [{"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                                              "weight_decay": layerwise_weight_decay,
                                              "lr": lr,
                                             },
                                             {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                                              "weight_decay": 0.0,
                                              "lr": lr,
                                             },]
            lr *= lr_decay_cofficient
        return optimizer_grouped_parameters

    if CFG.layerwise_lr_decay:
        grouped_optimizer_params = get_optimizer_grouped_parameters(model, 
                                                                    CFG.layerwise_lr, 
                                                                    CFG.layerwise_weight_decay, 
                                                                    CFG.lr_decay_cofficient)
        optimizer = AdamW(grouped_optimizer_params,
                          lr = CFG.layerwise_lr,
                          eps = CFG.layerwise_adam_epsilon,
                          betas = CFG.betas)
    else:
        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=CFG.encoder_lr, 
                                                    decoder_lr=CFG.decoder_lr,
                                                    weight_decay=CFG.weight_decay)
        optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
        
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
    
    best_score = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})
        
        if best_score > score:
            best_score = score
            logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_gpu{GPU}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_gpu{GPU}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

if __name__ == '__main__':
    
    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        logger.info(f'Score: {score:<.4f}  Scores: {scores}')
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        logger.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+f'gpu{GPU}_oof_df.pkl')
        
    if CFG.wandb:
        wandb.finish()

