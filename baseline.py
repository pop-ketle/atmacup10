import numpy as np
import pandas as pd
import lightgbm as lgbm
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostClassifier, CatBoostRegressor

from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')

plt.rcParams["font.family"] = "IPAexGothic"

N_SPLITS    = 5
RANDOM_SEED = 72

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
print(train.shape, test.shape) # (12026, 19) (12008, 18)

train_test = pd.concat([train, test], ignore_index=True)


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

# 収集に際して資金提供などを行った情報があるかどうか
train_test['exist_acquisition_credit_line'] = np.where(train_test['acquisition_credit_line'].isnull()==False, 1, 0)



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


cat_cols = ['principal_maker', 'principal_or_first_maker','copyright_holder','acquisition_method','acquisition_credit_line']
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
    'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
    'importance_type': 'gain' # 特徴重要度計算のロジック(後述)
}

skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)

lgbm_oof_pred, lgbm_models, lgbm_scores = make_prediction(train, y,
                                                        skf.split(train, y),
                                                        lgbm.LGBMRegressor(**lgbm_params),
                                                        model_name='lgbm',
                                                        logarithmic=True)

scores = lgbm_scores
score  = sum(scores) / len(scores)
print(scores)
print(f'cv: {score}')

# cat_oof_pred, cat_models, cat_scores = make_prediction(train, y,
#                                                         skf.split(train, y),
#                                                         CatBoostRegressor(),
#                                                         model_name='catboost',
#                                                         logarithmic=True)

# scores = lgbm_scores + cat_scores
# score  = sum(scores) / len(scores)
# print(scores)
# print(f'cv: {score}')


# X2 = np.stack([lgbm_oof_pred, cat_oof_pred])
# X2 = X2.T
# X2 = pd.DataFrame(X2, columns=[i for i in range(X2.shape[-1])])
# print(X2.shape)

# ridge_oof_pred, ridge_models, ridge_scores = make_prediction(X2, y,
#                                                         skf.split(X2, y),
#                                                         Ridge(),
#                                                         model_name='ridge',
#                                                         logarithmic=True)
# score  = sum(ridge_scores) / len(ridge_scores)
# print(ridge_scores)
# print(f'cv: {score}')



pred = np.array([model.predict(test) for model in lgbm_models])
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
    if i%2==0: continue
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

# 予測値の可視化
fig, ax = plt.subplots(figsize=(8, 8))
sns.distplot(np.log1p(pred), label='Test Predict')
sns.distplot(lgbm_oof_pred, label='LGBM Out Of Fold')
# sns.distplot(cat_oof_pred, label='CAT Out Of Fold')
ax.legend()
ax.grid()
plt.savefig(f'./figs/cv:{score}_histogram.png')
plt.show()