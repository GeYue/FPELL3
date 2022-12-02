#coding=UTF-8

import pandas as pd
df_fb2021 = pd.read_csv('./feedback-prize-2021/train.csv', dtype={'discourse_id':int})
df_fb2021['textlen'] = df_fb2021.discourse_text.str.len()
print(df_fb2021)


import re
def join_texts(v):
    return re.sub(' +', ' ', ' '.join(v.discourse_text).replace("\n", ' ').replace("\a", ' ') )

df_fb2021_agg = df_fb2021.groupby('id').apply(lambda v: join_texts(v)).to_frame('text')
print(df_fb2021_agg)
df_fb2021_agg.sample().text.item()


df_fb3 = pd.read_csv('../data/train.csv')
df_fb3.sample().full_text.item()

common_ids = set(df_fb2021_agg.index) & set(df_fb3.text_id)
print(len(common_ids))
print(len(common_ids) / len(df_fb3))

df_fb3.query('text_id == "009F4E9310CB"').full_text.item()
df_fb2021_agg.loc['009F4E9310CB'].item()

# import numpy as np
# all_fb3 = ' '.join(df_fb3.full_text)
# print(np.mean([r.discourse_text in all_fb3 for i, r in df_fb2021.sample(1000).iterrows()]), ' fb2021 strings are in fb3 dataset')
# all_fb2021 = ' '.join(df_fb2021_agg.text)
# print(np.mean([r.full_text in all_fb2021 for i, r in df_fb3.sample(1000).iterrows()]), ' fb3 strings are in fb2021 dataset')


ft = df_fb2021_agg.index.isin(df_fb3.text_id)
ref2021_agg = df_fb2021_agg[~ft]

#!cp ../input/feedback-prize-english-language-learning/train.csv .
ref2021_agg.reset_index().rename(columns={'id': 'text_id', 'text': 'full_text'}).to_csv('test.csv', index=False)
df_sample = ref2021_agg.reset_index().rename(columns={'id': 'text_id'})[['text_id']]
df_sample[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']] = 3.0
df_sample.to_csv('sample_submission.csv', index=False)


