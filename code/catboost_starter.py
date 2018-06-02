import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
POS_CASH = pd.read_csv('../input/POS_CASH_balance.csv')
credit_card = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
previous_app = pd.read_csv('../input/previous_application.csv')
subm = pd.read_csv("../input/sample_submission.csv")


print("Converting...")
le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = \
    le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = \
    POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR')\
        .nunique()[['NAME_CONTRACT_STATUS']]\
        .rename(columns={'NAME_CONTRACT_STATUS': 'NUNIQUE_STATUS_POS_CASH'})
nunique_status.reset_index(inplace=True)
POS_CASH = POS_CASH.merge(nunique_status, how='left', on='SK_ID_CURR')
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

credit_card['NAME_CONTRACT_STATUS'] = \
    le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = \
    credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR')\
        .nunique()[['NAME_CONTRACT_STATUS']]\
        .rename(columns={'NAME_CONTRACT_STATUS': 'NUNIQUE_STATUS_credit_card'})
nunique_status.reset_index(inplace=True)
credit_card = credit_card.merge(nunique_status, how='left', on='SK_ID_CURR')
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

bureau_cat_features = [f for f in bureau.columns if bureau[f].dtype == 'object']
for f in bureau_cat_features:
    bureau[f] = le.fit_transform(bureau[f].astype(str))
    nunique = bureau[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()[[f]]\
        .rename(columns={f: 'NUNIQUE_'+f+'_bureau'})
    nunique.reset_index(inplace=True)
    bureau = bureau.merge(nunique, how='left', on='SK_ID_CURR')
    bureau.drop([f], axis=1, inplace=True)
bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

previous_app_cat_features = [f for f in previous_app.columns if previous_app[f].dtype == 'object']
for f in previous_app_cat_features:
    previous_app[f] = le.fit_transform(previous_app[f].astype(str))
    nunique = previous_app[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()[[f]]\
        .rename(columns={f: 'NUNIQUE_'+f+'_previous_app'})
    nunique.reset_index(inplace=True)
    previous_app = previous_app.merge(nunique, how='left', on='SK_ID_CURR')
    previous_app.drop([f], axis=1, inplace=True)
previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

print("Merging...")
data_train = application_train.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(),
                                                             how='left', on='SK_ID_CURR')
data_test = application_test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(),
                                                           how='left', on='SK_ID_CURR')

data_train = data_train.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(),
                                                         how='left', on='SK_ID_CURR')
data_test = data_test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(),
                                                       how='left', on='SK_ID_CURR')
                                                       
data_train = data_train.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(),
                                                    how='left', on='SK_ID_CURR')
data_test = data_test.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(),
                                                  how='left', on='SK_ID_CURR')
                                                  
data_train = data_train.merge(previous_app.groupby('SK_ID_CURR').mean().reset_index(),
                                                          how='left', on='SK_ID_CURR')
data_test = data_test.merge(previous_app.groupby('SK_ID_CURR').mean().reset_index(),
                                                        how='left', on='SK_ID_CURR')
   
target_train = data_train['TARGET']
data_train.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
data_test.drop(['SK_ID_CURR'], axis=1, inplace=True)

cat_features = [f for f in data_train.columns if data_train[f].dtype == 'object']
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
cat_features_inds = column_index(data_train, cat_features)    
print("Cat features are: %s" % [f for f in cat_features])
print(cat_features_inds)

for col in cat_features:
    data_train[col] = le.fit_transform(data_train[col].astype(str))
    data_test[col] = le.fit_transform(data_test[col].astype(str))
    
data_train.fillna(-1, inplace=True)
data_test.fillna(-1, inplace=True)
cols = data_train.columns

X_train, X_valid, y_train, y_valid = train_test_split(data_train, target_train,
                                                      test_size=0.1, random_state=17)
print(X_train.shape)
print(X_valid.shape)

                                     
print("\nCatBoost...")                                     
cb_model = CatBoostClassifier(iterations=1000,
                              learning_rate=0.1,
                              depth=7,
                              l2_leaf_reg=40,
                              bootstrap_type='Bernoulli',
                              subsample=0.7,
                              scale_pos_weight=5,
                              eval_metric='AUC',
                              metric_period=50,
                              od_type='Iter',
                              od_wait=45,
                              random_seed=17,
                              allow_writing_files=False)

cb_model.fit(X_train, y_train,
             eval_set=(X_valid, y_valid),
             cat_features=cat_features_inds,
             use_best_model=True,
             verbose=True)
             
fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': cols})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
plt.savefig('catboost_feature_importance.png')             

print('AUC:', roc_auc_score(y_valid, cb_model.predict_proba(X_valid)[:,1]))
y_preds = cb_model.predict_proba(data_test)[:,1]
subm['TARGET'] = y_preds
subm.to_csv('submission.csv', index=False)