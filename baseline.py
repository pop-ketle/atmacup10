import yaml
import nltk
import colorsys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import texthero as hero
import lightgbm as lgbm
from boruta import BorutaPy
import seaborn as sns
import matplotlib.pyplot as plt
from fasttext import load_model
from geopy.geocoders import Nominatim
from gensim.models import word2vec, KeyedVectors
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')

nltk.download('stopwords')
# 英語とオランダ語を stopword として指定
custom_stopwords = nltk.corpus.stopwords.words('dutch') + nltk.corpus.stopwords.words('english')

plt.rcParams['font.family'] = 'IPAexGothic'

N_SPLITS    = 5
RANDOM_SEED = 313


def cleansing_hero_only_text(input_df, text_col):
    ## get only text (remove html tags, punctuation & digits)
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.stem,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

# text の基本的な情報をgetする関数
def basic_text_features_transformer(input_df, column, cleansing_hero=None, name=''):
    input_df[column] = input_df[column].astype(str)
    _df = pd.DataFrame()
    _df[column + name + '_num_chars']             = input_df[column].apply(len)
    _df[column + name + '_num_exclamation_marks'] = input_df[column].apply(lambda x: x.count('!'))
    _df[column + name + '_num_question_marks']    = input_df[column].apply(lambda x: x.count('?'))
    _df[column + name + '_num_punctuation']       = input_df[column].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    _df[column + name + '_num_symbols']           = input_df[column].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    _df[column + name + '_num_words']             = input_df[column].apply(lambda x: len(x.split()))
    _df[column + name + '_num_unique_words']      = input_df[column].apply(lambda x: len(set(w for w in x.split())))
    _df[column + name + '_words_vs_unique']       = _df[column + name + '_num_unique_words'] / _df[column + name + '_num_words']
    _df[column + name + '_words_vs_chars']        = _df[column + name + '_num_words'] / _df[column + name + '_num_chars']
    return _df

# カウントベースの text vector をgetする関数 
def text_vectorizer(input_df, 
                    text_columns,
                    cleansing_hero=None,
                    vectorizer=CountVectorizer(),
                    transformer=TruncatedSVD(n_components=128),
                    name='count_svd'):
    
    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].astype(str).fillna('missing')
    features = []
    for c in text_columns:
        if cleansing_hero is not None:
            output_df[c] = cleansing_hero(output_df, c)

        sentence = vectorizer.fit_transform(output_df[c])
        feature = transformer.fit_transform(sentence)
        num_p = feature.shape[1]
        feature = pd.DataFrame(feature, columns=[name+str(num_p) + f'_{i:03}' for i in range(num_p)])
        features.append(feature)
    output_df = pd.concat(features, axis=1)
    return output_df

def make_prediction(train, y, fold, model, model_name=None, logarithmic=False):
    oof_pred = np.zeros_like(y, dtype=np.float)
    scores, models = [], []
    for i, (train_idx, valid_idx) in enumerate(fold):
        x_train, x_valid = train.iloc[train_idx], train.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if  logarithmic:
            y_train, y_valid = np.log1p(y_train), np.log1p(y_valid)

        # 学習
        if model_name=='lgbm':
            train_data, valid_data = lgbm.Dataset(x_train, y_train), lgbm.Dataset(x_valid, y_valid)
            model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=50,
                    verbose=False,
                )

        elif model_name=='catboost':
            train_data, valid_data = Pool(x_train, y_train), Pool(x_valid, y_valid)
            model.fit(train_data, 
                eval_set=valid_data,
                early_stopping_rounds=50,
                verbose=False,
                use_best_model=True)
        
        else:
            model.fit(x_train, y_train)
        
        pred_valid = model.predict(x_valid)
        score = mean_squared_error(y_valid, pred_valid, squared=False)
        print(f'Fold:{i} {model_name} RMSLE: {score}')

        oof_pred[valid_idx] = pred_valid
        models.append(model)
        scores.append(score)
        
    return oof_pred, models, scores


train = pd.read_csv('./features/train.csv')
test  = pd.read_csv('./features/test.csv')

color_df   = pd.read_csv('./features/color.csv')
palette_df = pd.read_csv('./features/palette.csv')

historical_person_df = pd.read_csv('./features/historical_person.csv')
object_collection_df = pd.read_csv('./features/object_collection.csv')

technique_df = pd.read_csv('./features/technique.csv')

material_df = pd.read_csv('./features/material.csv')

maker_info_df                 = pd.read_csv('./features/maker.csv')
principal_maker_df            = pd.read_csv('./features/principal_maker.csv')
principal_maker_occupation_df = pd.read_csv('./features/principal_maker_occupation.csv')

with open('./features/materials_converter.yml') as f:
    materials_dict = yaml.safe_load(f.read())

# materials_dictの対応関係を逆にする
new_dict = dict()
for key, value in materials_dict.items():
    for v in value:
        new_dict[v] = key
materials_dict = new_dict

production_place   = pd.read_csv('./features/production_place.csv').rename(columns={'name': 'place_name'})
production_country = pd.read_csv('./features/production_country.csv') # 作成した特徴量

train_test = pd.concat([train, test], ignore_index=True)
print(train.shape, test.shape, train_test.shape) # (12026, 19) (12008, 18) (24034, 19)


# text features
transformer = TruncatedSVD(n_components=128, random_state=RANDOM_SEED)
for c in ['title', 'description', 'long_title']:
    for name in ['en','nl','ex']:
        _df = pd.read_csv(f'./features/texts_lang/{c}_{name}_feature.csv')
        feature = transformer.fit_transform(_df.drop(['Unnamed: 0','object_id'], axis=1))

        num_p = feature.shape[1]
        feature = pd.DataFrame(feature, columns=[f'{c}_{name}{num_p}_{i:03}' for i in range(num_p)])
        _df = pd.concat([_df['object_id'], feature], axis=1)
        
        train_test = pd.merge(train_test, _df, on='object_id', how='left')
print(train_test.shape)


###############################

# for c in train.columns:
#     print(c, len(set(train[c])))
# object_id 12026
# art_series_id 11169
# title 10240
# description 6984
# long_title 11058
# principal_maker 2221
# principal_or_first_maker 2250
# sub_title 8573
# copyright_holder 33
# more_title 10472
# acquisition_method 9
# acquisition_date 713
# acquisition_credit_line 430
# dating_presenting_date 2793
# dating_sorting_date 493
# dating_period 10
# dating_year_early 493
# dating_year_late 520
# likes 629

# print(train.isnull().sum())
# object_id                       0
# art_series_id                   0
# title                           0
# description                  3520
# long_title                      0
# principal_maker                 0
# principal_or_first_maker        1
# sub_title                      34
# copyright_holder            11313
# more_title                    135
# acquisition_method            196
# acquisition_date             1017
# acquisition_credit_line      8501
# dating_presenting_date         10
# dating_sorting_date            10
# dating_period                   0
# dating_year_early              10
# dating_year_late               38
# likes                           0
# dtype: int64

# print(test.isnull().sum())
# object_id                       0
# art_series_id                   0
# title                           0
# description                  3685
# long_title                      0
# principal_maker                 0
# principal_or_first_maker        0
# sub_title                      30
# copyright_holder            11287
# more_title                    164
# acquisition_method            175
# acquisition_date              998
# acquisition_credit_line      8470
# dating_presenting_date          8
# dating_sorting_date             8
# dating_period                   0
# dating_year_early               8
# dating_year_late               31
# dtype: int64


###############################

# 色の情報を特徴量へ
_df = palette_df.groupby('object_id').size()

_df = palette_df.groupby('object_id')['ratio'].agg(['min','max']).add_prefix('color_ratio_')
train_test = pd.merge(train_test, _df, on='object_id', how='left')

# rgbごとに'min','max','mean','std','var','max-min'をとる
for c in ['color_r','color_g','color_b']:
    _df = palette_df.groupby('object_id')[c].agg(['min','max','mean','std','var']).add_prefix(f'{c}_')
    _df[f'{c}_max-min'] = _df[f'{c}_max'] - _df[f'{c}_min']
    train_test = pd.merge(train_test, _df, on='object_id', how='left')

# HSVに変換して、HSVごとに'min','max','mean','std','var','max-min'をとる
palette_df['HSV'] = list(map(lambda r,g,b: list(colorsys.rgb_to_hsv(r,g,b)),
                        palette_df['color_r'], palette_df['color_g'], palette_df['color_b']))
palette_df['color_h'] = palette_df['HSV'].map(lambda x: x[0])
palette_df['color_s'] = palette_df['HSV'].map(lambda x: x[1])
palette_df['color_v'] = palette_df['HSV'].map(lambda x: x[2])
palette_df = palette_df.drop(['HSV'], axis=1)

for c in ['color_h','color_s','color_v']:
    _df = palette_df.groupby('object_id')[c].agg(['min','max','mean','std','var']).add_prefix(f'{c}_')
    _df[f'{c}_max-min'] = _df[f'{c}_max'] - _df[f'{c}_min']
    train_test = pd.merge(train_test, _df, on='object_id', how='left')

# ratio最大のものを取得
max_palette = palette_df.groupby('object_id')['ratio'].max().reset_index()
max_palette = pd.merge(max_palette, palette_df, on=['object_id','ratio'], how='left').rename(
    columns={'ratio':'max_ratio', 'color_r':'max_palette_r', 'color_g':'max_palette_g','color_b':'max_palette_b',
            'color_h':'max_palette_h', 'color_s':'max_palette_s','color_v':'max_palette_v'})
max_palette = max_palette.loc[max_palette['object_id'].drop_duplicates().index.tolist()].reset_index()  # 同じidでmax ratioが同じものは削除
max_palette = max_palette.drop(['index'], axis=1)
train_test = pd.merge(train_test, max_palette, on='object_id', how='left')

# 平均のrgb、hsvを取得
mean_palette = palette_df.copy()
for c in ['color_r','color_g','color_b','color_h','color_s','color_v']:
    mean_palette[c] = palette_df['ratio'] * palette_df[c]
mean_palette = mean_palette.groupby('object_id').sum().reset_index().rename(
    columns={'color_r':'mean_palette_r', 'color_g':'mean_palette_g','color_b':'mean_palette_b',
            'color_h':'mean_palette_h', 'color_s':'mean_palette_s','color_v':'mean_palette_v'})
train_test = pd.merge(train_test, mean_palette, on='object_id', how='left')


# 作家情報を追加
maker_info_df = maker_info_df.add_prefix('principal_maker_').rename(columns={'principal_maker_name': 'principal_maker'})
train_test = pd.merge(train_test, maker_info_df, on='principal_maker', how='left')

principal_maker_df = pd.merge(principal_maker_df, principal_maker_occupation_df, on='id', how='left')
principal_maker_df = principal_maker_df.rename(columns={'name': 'work'})
principal_maker_df['productionPlaces'] = principal_maker_df['productionPlaces'].map({'Liège': 'Liege'})

for c in ['qualification', 'roles', 'productionPlaces', 'work']:
    # principal_maker_df[c] = principal_maker_df[c].apply(lambda x: x.replace(' ', '_'))
    _df = pd.crosstab(principal_maker_df['object_id'], principal_maker_df[c]).add_prefix(f'{c}=')
    train_test = pd.merge(train_test, _df, on='object_id', how='left')

# 作品がどのような形式であるか
# クロス集計表にデータを成型してマージ
cross_object_type = pd.crosstab(object_collection_df['object_id'], object_collection_df['name']).add_prefix('object_type=')
train_test = pd.merge(train_test, cross_object_type, on='object_id', how='left')

# 作品にhistorical_personが写っているかどうか、その人数
_df = historical_person_df['object_id'].value_counts().reset_index().rename(columns={'index': 'object_id', 'object_id': 'n_historical_person'})
train_test = pd.merge(train_test, _df, on='object_id', how='left')
train_test['n_historical_person'] = train_test['n_historical_person'].fillna(0)

train_test['exist_historical_person'] = np.where(train_test['n_historical_person']>=1, 1, 0)

# # NOTE: lightgbm.basic.LightGBMError: Do not support special JSON characters in feature name.
# # クロス集計表にデータを成型してマージ
# vc = historical_person_df['name'].value_counts()

# # 出現回数30以上に絞る
# use_names = vc[vc > 30].index

# # isin で 30 回以上でてくるようなレコードに絞り込んでから corsstab を行なう
# idx = historical_person_df['name'].isin(use_names)
# _use_df = historical_person_df[idx].reset_index(drop=True)
# # NOTE: 正規表現の方がいいだろうけど
# _use_df['name'] = _use_df['name'].str.strip('(')
# _use_df['name'] = _use_df['name'].str.strip(')')

# cross_historical_person = pd.crosstab(_use_df['object_id'], _use_df['name']).add_prefix('historical_person=')
# train_test = pd.merge(train_test, cross_historical_person, on='object_id', how='left')

# NOTE: 若干CV落ちたのでパス
# # どのような技法で描かれたか
# # クロス集計表にデータを成型してマージ
# cross_technique = pd.crosstab(technique_df['object_id'], technique_df['name']).add_prefix('technique=')
# train_test = pd.merge(train_test, cross_technique, on='object_id', how='left')

# 'object_id'の出現回数を特徴量へ
place_counts = production_place['object_id'].value_counts().reset_index().rename(columns={'index': 'object_id', 'object_id': 'n_place'})
train_test = pd.merge(train_test, place_counts, on='object_id', how='left')

# クロス集計表にデータを成型してマージ
cross_place   = pd.crosstab(production_place['object_id'], production_place['place_name'])
cross_country = pd.crosstab(production_country['object_id'], production_country['country_name'])

# train_test = pd.merge(train_test, cross_place, on='object_id', how='left') # ?がきついっぽいのでマージなしで
train_test = pd.merge(train_test, cross_country, on='object_id', how='left')

# # acquisition_methodに応じてフラグ NOTE: label encodingしてるからいらないんじゃ...
# dfs = []
# dfs.append(pd.DataFrame(np.where(train_test['acquisition_method']=='transfer', 1, 0), columns=['acquisition_method_is_transfer']))
# dfs.append(pd.DataFrame(np.where(train_test['acquisition_method']=='unknowwn', 1, 0), columns=['acquisition_method_is_unknowwn']))
# dfs.append(pd.DataFrame(np.where(train_test['acquisition_method']=='bequest', 1, 0), columns=['acquisition_methodis_bequest']))
# dfs.append(pd.DataFrame(np.where(train_test['acquisition_method']=='loan', 1, 0), columns=['acquisition_methodis_loan']))
# dfs.append(pd.DataFrame(np.where(train_test['acquisition_method']=='nationalization', 1, 0), columns=['acquisition_methodis_nationalization']))
# _df = pd.concat(dfs, axis=1)
# train_test = pd.concat([train_test, _df], axis=1)


# material.csv, technique.csv, collection.csvの連なりを文章として見立ててWord2Vec
mat_col = pd.concat([material_df, object_collection_df], axis=0).reset_index(drop=True)
mat_tec = pd.concat([material_df, technique_df], axis=0).reset_index(drop=True)
col_tec = pd.concat([object_collection_df, technique_df], axis=0).reset_index(drop=True)
mat_col_tec = pd.concat([material_df, object_collection_df, technique_df], axis=0).reset_index(drop=True)

maker_df = train_test[['object_id','principal_maker']]
maker_df = maker_df.rename(columns={'principal_maker': 'name'})

maker_mat    = pd.concat([maker_df, material_df], axis=0).reset_index(drop=True)
maker_col    = pd.concat([maker_df, object_collection_df], axis=0).reset_index(drop=True)
maker_tec    = pd.concat([maker_df, technique_df], axis=0).reset_index(drop=True)
maker_person = pd.concat([maker_df, historical_person_df], axis=0).reset_index(drop=True)

maker_mat_col = pd.concat([maker_df, mat_col], axis=0).reset_index(drop=True)
maker_mat_tec = pd.concat([maker_df, mat_tec], axis=0).reset_index(drop=True)
maker_col_tec = pd.concat([maker_df, col_tec], axis=0).reset_index(drop=True)
maker_mat_col_tec = pd.concat([maker_df, mat_col_tec], axis=0).reset_index(drop=True)

person_mat    = pd.concat([historical_person_df, material_df], axis=0).reset_index(drop=True)
person_col    = pd.concat([historical_person_df, object_collection_df], axis=0).reset_index(drop=True)
person_tec    = pd.concat([historical_person_df, technique_df], axis=0).reset_index(drop=True)

person_mat_col = pd.concat([historical_person_df, mat_col], axis=0).reset_index(drop=True)
person_mat_tec = pd.concat([historical_person_df, mat_tec], axis=0).reset_index(drop=True)
person_col_tec = pd.concat([historical_person_df, col_tec], axis=0).reset_index(drop=True)
person_mat_col_tec  = pd.concat([historical_person_df, mat_col_tec], axis=0).reset_index(drop=True)

maker_person_mat = pd.concat([maker_df, person_mat], axis=0).reset_index(drop=True)
maker_person_col = pd.concat([maker_df, person_col], axis=0).reset_index(drop=True)
maker_person_tec = pd.concat([maker_df, person_tec], axis=0).reset_index(drop=True)

maker_person_mat_col_tec = pd.concat([maker_person, mat_col_tec], axis=0).reset_index(drop=True)

# 単語ベクトル表現の次元数 NOTE: 元の語彙数をベースに適当に決めました
model_size = {
    'material': 20,
    'technique': 8,
    'collection': 3,
    'maker': 50,
    'person': 50,
    'material_collection': 20,
    'material_technique': 20,
    'collection_technique': 10,
    'material_collection_technique': 25,
    'maker_materianl': 50,
    'makert_collection': 50,
    'maker_technique': 50,
    'maker_person': 75,
    'person_materianl': 50, 
    'person_collection': 50, 
    'person_technique': 50,
    'maker_mat_col': 50,
    'maker_mat_tec': 50,
    'maker_col_tec': 50,
    'maker_mat_col_tec': 50,
    'person_mat_col': 50,
    'person_mat_tec': 50,
    'person_col_tec': 50,
    'person_mat_col_tec': 75,
    'maker_person_mat': 100,
    'maker_person_col': 100,
    'maker_person_tec': 100,
    'maker_person_mat_col_tec': 125,
}
n_iter = 100

w2v_dfs = []
for df, df_name in zip(
        [
            material_df, object_collection_df, technique_df,
            maker_df, historical_person_df,
            mat_col, mat_tec, col_tec,
            maker_mat ,maker_col, maker_tec, maker_person,
            person_mat, person_col, person_tec,
            mat_col_tec,
            maker_mat_col, maker_mat_tec, maker_col_tec, maker_mat_col_tec,
            person_mat_col, person_mat_tec, person_col_tec, person_mat_col_tec,
            maker_person_mat, maker_person_col, maker_person_tec,
            maker_person_mat_col_tec,
        ], [
            'material', 'collection', 'technique',
            'maker', 'person',
            'material_collection', 'material_technique', 'collection_technique',
            'maker_materianl', 'makert_collection', 'maker_technique', 'maker_person',
            'person_materianl', 'person_collection', 'person_technique',
            'material_collection_technique',
            'maker_mat_col', 'maker_mat_tec', 'maker_col_tec', 'maker_mat_col_tec',
            'person_mat_col', 'person_mat_tec', 'person_col_tec', 'person_mat_col_tec',
            'maker_person_mat', 'maker_person_col', 'maker_person_tec',
            'maker_person_mat_col_tec',
        ]):

    # dfs = []
    # df_group = df.groupby('object_id')['name'].apply(str).reset_index()

    # _df = basic_text_features_transformer(df_group, 'name', cleansing_hero=cleansing_hero_only_text, name=df_name)
    # dfs.append(_df)

    # _df = text_vectorizer(df_group,
    #                             ['name'],
    #                             cleansing_hero=cleansing_hero_only_text,
    #                             vectorizer=CountVectorizer(),
    #                             transformer=TruncatedSVD(n_components=64, random_state=RANDOM_SEED),
    #                             name=f'{df_name}_countvec_sdv'
    #                             )
    # dfs.append(_df)

    # _df = text_vectorizer(df_group,
    #                             ['name'],
    #                             cleansing_hero=cleansing_hero_only_text,
    #                             vectorizer=TfidfVectorizer(),
    #                             transformer=TruncatedSVD(n_components=64, random_state=RANDOM_SEED),
    #                             name=f'{df_name}_tfidf_sdv'
    #                             )
    # dfs.append(_df)

    # output_df = pd.concat(dfs, axis=1)
    # train_test = pd.concat([train_test, output_df], axis=1)

    
    df_group = df.groupby('object_id')['name'].apply(list).reset_index()

    # Word2Vecの学習
    w2v_model = word2vec.Word2Vec(df_group['name'].values.tolist(),
                                  size=model_size[df_name],
                                  min_count=1,
                                  window=1,
                                  iter=n_iter)

    # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
    sentence_vectors = df_group['name'].apply(
        lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
    sentence_vectors = np.vstack([x for x in sentence_vectors])
    sentence_vector_df = pd.DataFrame(sentence_vectors,
                                      columns=[f'{df_name}_w2v_{i}'
                                               for i in range(model_size[df_name])])
    sentence_vector_df.index = df_group['object_id']
    w2v_dfs.append(sentence_vector_df)

for w2v_df in w2v_dfs:
    train_test = pd.merge(train_test, w2v_df, on='object_id', how='left')


# materials をまとめて数を減らした上で結合
_df = material_df['name'].apply(lambda x: materials_dict[x])
_df    = pd.concat([material_df['object_id'], _df], axis=1).rename(columns={'name': 'material'})
material_df    = pd.concat([material_df, _df['material']], axis=1)
cross_material = pd.crosstab(material_df['object_id'], material_df['material']).add_prefix('material=')
train_test = pd.merge(train_test, cross_material, on='object_id', how='left')

# NOTE: これをどう活かせばいいのかよくわからない
# cross_material = pd.crosstab(material_df['object_id'], material_df['name'])
# freq_material_df = apriori(cross_material, min_support=.005, use_colnames=True).sort_values('support', ascending=False)
# print(freq_material_df)
# association_rules(freq_material_df, metric='lift').sort_values('lift', ascending=False)
# sns.clustermap(cross_material.corr(), cmap='Blues')
# plt.show()
# exit()

# 年代でビニングしてみる(時期、世紀)
_df = pd.DataFrame({
    'period': pd.cut(train_test['dating_sorting_date'], [1250, 1400, 1600, 1900, 2100], labels=False),
    'century': pd.cut(train_test['dating_sorting_date'], [1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100], labels=False)
})
train_test = pd.concat([train_test, _df], axis=1)

# 収集に際して資金提供などを行った情報があるかどうか
train_test['exist_acquisition_credit_line'] = np.where(train_test['acquisition_credit_line'].isnull()==False, 1, 0)

# 主な製作者にたいしてagg
group = train_test.groupby('principal_maker')

agg_df = pd.concat([
    group.size().rename('n_principal_maker'), # 著者が何回出てくるか
    group['sub_title'].nunique().rename('nunique_sub_title'), # 著者ごとに何種類の sub_title を持っているか
    group['dating_sorting_date'].agg(['min', 'max', 'mean']).add_prefix('dating_sorting_date_grpby_principal_maker_'), # 著者ごとに描いた年度の最小・最大・平均
    # group['likes'].agg(['min', 'max', 'mean']).add_prefix('likes_grpby_principal_maker_')
], axis=1)
train_test = pd.merge(train_test, agg_df, on='principal_maker', how='left')

# # 作者ごとに、それまでの年度の絵で獲得した合計のいいね数
# sorted_group = train.sort_values('dating_year_late').groupby('principal_maker')['likes'].cumsum()
# print(sorted_group)
# exit()

for c in ['principal_maker','principal_or_first_maker','title','description','sub_title','long_title','more_title']:
    print(c)
    dfs = []

    _df = basic_text_features_transformer(train_test, c, cleansing_hero=cleansing_hero_only_text, name='')
    dfs.append(_df)

    _df = text_vectorizer(train_test,
                                [c],
                                cleansing_hero=cleansing_hero_only_text,
                                vectorizer=CountVectorizer(),
                                transformer=TruncatedSVD(n_components=64, random_state=RANDOM_SEED),
                                name=f'{c}_countvec_sdv'
                                )
    dfs.append(_df)

    _df = text_vectorizer(train_test,
                                [c],
                                cleansing_hero=cleansing_hero_only_text,
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=64, random_state=RANDOM_SEED),
                                name=f'{c}_tfidf_sdv'
                                )
    dfs.append(_df)

    output_df = pd.concat(dfs, axis=1)
    train_test = pd.concat([train_test, output_df], axis=1)

    # Word2Vecの学習
    w2v_model = word2vec.Word2Vec(cleansing_hero_only_text(train_test, c), size=64, min_count=1, window=1, iter=100)

    # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
    sentence_vectors = cleansing_hero_only_text(train_test, c).apply(
        lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
    sentence_vectors = np.vstack([x for x in sentence_vectors])
    sentence_vector_df = pd.DataFrame(sentence_vectors, columns=[f'{c}_w2v_{i}' for i in range(64)])
    sentence_vector_df.index = train_test['object_id']
    
    train_test = pd.merge(train_test, sentence_vector_df, on='object_id', how='left')

# sub_titleから作品のサイズを抽出して単位をmmに統一する
for axis in ['h', 'w', 't', 'd']:
    column_name = f'size_{axis}'
    size_info = train_test['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis)) # 正規表現を使ってサイズを抽出
    size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
    size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float) # dtypeがobjectになってるのでfloatに直す
    size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
    train_test[column_name] = size_info[column_name] # trainにくっつける


for c in ['size_h', 'size_w', 'size_t', 'size_d']:
    train_test[c] = train_test[c].fillna(1.0).astype('float64')
train_test['size_h*w']     = train_test['size_h'] * train_test['size_w']
train_test['size_h*w*t']   = train_test['size_h'] * train_test['size_w'] * train_test['size_t']
train_test['size_h*w*t*d'] = train_test['size_h'] * train_test['size_w'] * train_test['size_t'] * train_test['size_d']

# # NOTE: 変なもの( h 166mm × w 78/54mm )が混じってて例外処理がいるけどとりあえず延期で
# print(train_test.iloc[18194][['sub_title', 'size_h', 'size_w', 'size_t', 'size_d']])
# exit()

# 言語判定特徴
model = load_model('./bin/lid.176.bin')

for c in ['title', 'description', 'long_title']:
    train_test[f'{c}_lang'] = train_test[c].fillna('').map(
        lambda x: model.predict(x.replace('\n', ''))[0][0].split('_')[-1])


# # geopyによる地名 -> 国名の変換 重めなので作ってセーブしておく
# def place2country(address):
#     geolocator = Nominatim(user_agent='sample', timeout=200)
#     loc = geolocator.geocode(address, language='en')
#     coordinates = (loc.latitude, loc.longitude)
#     location = geolocator.reverse(coordinates, language='en')
#     country = location.raw['address']['country']
#     return country

# country_dict = {}
# for place in tqdm(set(production_place['name'])):
#     try:
#         country_dict[place] = place2country(place)
#     except:
#         # 国名を取得できない場合はnan
#         country_dict[place] = np.nan

# production_place['country_name'] = production_place['name'].map(country_dict)

# out_df = pd.DataFrame({
#     'object_id': production_place['object_id'],
#     'country_name': production_place['country_name']
# })
# out_df.to_csv('./features/production_country.csv', index=False)

# print(production_place)


# 各種エンコーディング
def count_encoding(df, target_col):
    _df = pd.DataFrame(df[target_col].value_counts().reset_index()).rename(columns={'index': target_col, target_col: f'CE_{target_col}'})
    return pd.merge(df, _df, on=target_col, how='left')

def label_encoding(df, target_col):
    le = preprocessing.LabelEncoder()
    df[f'LE_{target_col}'] = le.fit_transform(df[target_col])
    return df

def onehot_encoding(df, target_col):
    _df = pd.get_dummies(df[target_col], dummy_na=False).add_prefix(f'OH_{target_col}=')
    return pd.concat([df, _df], axis=1)

def target_encoding(train, test, target_col, y_col):
    # 学習データ全体でカテゴリにおけるyの平均を計算
    data_tmp    = pd.DataFrame({'target': train[target_col], 'y': train[y_col]})
    target_mean = data_tmp.groupby('target')['y'].mean()
    # テストデータのカテゴリを追加
    test[f'TE_{target_col}'] = test[target_col].map(target_mean)

    # 変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train.shape[0])
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
    for train_idx, test_idx in skf.split(train, train[y_col]):
        # 学習データに対して、各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[train_idx].groupby('target')['y'].mean()
        # バリデーションデータについて、変換後の値を一時配列に格納
        tmp[test_idx] = train[target_col].iloc[test_idx].map(target_mean)
    # 返還後のデータで元の変数を置換
    train[f'TE_{target_col}'] = tmp


# cat_cols = [
#     'principal_maker',
#     'principal_or_first_maker',
#     'copyright_holder',
#     'acquisition_date',
#     'acquisition_method',
#     'acquisition_credit_line',
#     'title',
#     'sub_title',
#     'more_title',
#     'long_title',
#     'description',
#     'title_lang',
#     'description_lang',
#     'long_title_lang',
#     'dating_presenting_date',
#     'period',
#     'century',
#     'principal_maker_nationality',
#     'principal_maker_date_of_death',
#     'principal_maker_date_of_birth',
# ]
cat_cols = train_test.select_dtypes(include=object).columns.tolist()

for c in cat_cols:
    train_test = count_encoding(train_test, c)
    train_test = label_encoding(train_test, c)
    # train_test = onehot_encoding(train_test, c)
# print(train_test)

# trainとtestに分割
train, test = train_test[:len(train)], train_test[len(train):]

for c in cat_cols:
    target_encoding(train, test, c, 'likes')


obj_col = train_test.select_dtypes(include=object).columns.tolist()
print(obj_col)
for c in obj_col:
    train = train.drop(c, axis=1)
    test  = test.drop(c, axis=1)

y = train['likes']
train = train.drop('likes', axis=1)
test  = test.drop('likes', axis=1)
print(train.shape, y.shape, test.shape)

lgbm_params = {
    'objective': 'rmse', # 目的関数. これの意味で最小となるようなパラメータを探します. 
    'learning_rate': 0.1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
    'reg_lambda': 1., # L2 Reguralization 
    'reg_alpha': .1, # こちらは L1
    'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
    # bagging の頻度と割合
    'subsample_freq': 3,
    'subsample': .9,
    'importance_type': 'gain', # 特徴重要度計算のロジック(後述)
    'random_state': RANDOM_SEED,
}

# # boruta
# X = train.copy()
# X = X.fillna(-999)
# model = lgbm.LGBMRegressor(**lgbm_params)
# feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=RANDOM_SEED)
# feat_selector.fit(X.values, y.values)

# # 選択された特徴量を確認
# selected = feat_selector.support_
# print('選択された特徴量の数: %d' % np.sum(selected))

# print(selected)
# print(train.columns[selected])

# train = train[train.columns[selected]]
# test  = test[test.columns[selected]]
# print(train.shape, y.shape, test.shape)


skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)

lgbm_oof_pred, lgbm_models, lgbm_scores = make_prediction(train, y,
                                                        skf.split(train, y),
                                                        lgbm.LGBMRegressor(**lgbm_params),
                                                        model_name='lgbm',
                                                        logarithmic=True)

# scores = lgbm_scores
# score  = sum(scores) / len(scores)
# print(scores)
# print(f'cv: {score}')

cat_oof_pred, cat_models, cat_scores = make_prediction(train, y,
                                                        skf.split(train, y),
                                                        CatBoostRegressor(),
                                                        model_name='catboost',
                                                        logarithmic=True)

scores = lgbm_scores + cat_scores
score  = sum(scores) / len(scores)
print(scores)
print(f'cv: {score}')


X2 = np.stack([lgbm_oof_pred, cat_oof_pred])
X2 = X2.T
X2 = pd.DataFrame(X2, columns=[i for i in range(X2.shape[-1])])
print(X2.shape)

ridge_oof_pred, ridge_models, ridge_scores = make_prediction(X2, y,
                                                        skf.split(X2, y),
                                                        Ridge(),
                                                        model_name='ridge',
                                                        logarithmic=True)
score  = sum(ridge_scores) / len(ridge_scores)
print(ridge_scores)
print(f'cv: {score}')



# pred = np.array([model.predict(test) for model in lgbm_models])
# pred = np.mean(pred, axis=0)
# pred = np.expm1(pred)
# pred = np.where(pred < 0, 0, pred)

test2 = []
lgbm_pred = np.array([model.predict(test) for model in lgbm_models])
cat_pred  = np.array([model.predict(test) for model in cat_models])
test2.append(np.mean(lgbm_pred, axis=0))
test2.append(np.mean(cat_pred, axis=0))
test2 = np.array(test2)
test2 = test2.T
print(f'test2.shape={test2.shape}')

pred = np.array([model.predict(test2) for model in ridge_models])
pred = np.mean(pred, axis=0)
pred = np.expm1(pred)
pred = np.where(pred < 0, 0, pred)


sub_df = pd.DataFrame({
    'likes': pred
    })
sub_df.to_csv(f'./submissions/cv:{score}.csv', index=False)



###############################

# feature importanceの可視化
feature_importance_df = pd.DataFrame()
for i, model in enumerate(lgbm_models):
    _df = pd.DataFrame()
    _df['feature_importance'] = model.feature_importances_
    # _df['column'] = train.drop(obj_col, axis=1).columns
    _df['column'] = train.columns
    _df['fold'] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

order = feature_importance_df.groupby('column')\
    .sum()[['feature_importance']]\
    .sort_values('feature_importance', ascending=False).index[:50]

fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
ax.tick_params(axis='x', rotation=90)
ax.grid()
fig.tight_layout()
plt.savefig(f'./figs/cv:{score}_feature_importance.png')
plt.show()

# feature_importance_df = pd.DataFrame()
# for i, model in enumerate(lgbm_models):
#     _df = pd.DataFrame()
#     _df['feature_importance'] = model.feature_importances_
#     _df['column'] = train.columns
#     _df['fold'] = i + 1
#     feature_importance_df = pd.concat([feature_importance_df, _df], 
#                                         axis=0, ignore_index=True)

# order = feature_importance_df.groupby('column')\
#     .sum()[['feature_importance']]\
#     .sort_values('feature_importance', ascending=False).index[:50]

# fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
# sns.boxenplot(data=feature_importance_df, 
#                 x='feature_importance', 
#                 y='column', 
#                 order=order, 
#                 ax=ax, 
#                 palette='viridis', 
#                 orient='h')
# ax.tick_params(axis='x', rotation=90)
# ax.set_title('Importance')
# ax.grid()
# fig.tight_layout()
# plt.savefig(f'./figs/cv:{score}_feature_importance.png')
# plt.show()


# 予測値の可視化
fig, ax = plt.subplots(figsize=(8, 8))
sns.histplot(np.log1p(pred), label='Test Predict', ax=ax, color='black')
sns.histplot(lgbm_oof_pred, label='LGBM Out Of Fold', ax=ax, color='C1')
ax.legend()
ax.grid()
plt.savefig(f'./figs/cv:{score}_histogram.png')
plt.show()