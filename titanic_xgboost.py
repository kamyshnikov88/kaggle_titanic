import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import StratifiedKFold
import xgboost as xg
from skopt import BayesSearchCV


def seed_init_fn(x):
    np.random.seed(x)


seed = 0
seed_init_fn(seed)

train_x = pd.read_csv('titanic_train_x.csv')
train_y = pd.read_csv('titanic_train_y.csv')
test = pd.read_csv('titanic_test.csv')
passenger_id = test[['PassengerId']]
test_to_model = test.drop('PassengerId', axis=1)

data_size = len(train_x)
validation_split = .3
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

model = xg.XGBClassifier(use_label_encoder=False, random_state=seed)
model.fit(train_x.iloc[train_indices], train_y.iloc[train_indices], eval_metric='logloss')
val_prediction = model.predict(train_x.iloc[val_indices])
print(sk.metrics.accuracy_score(train_y.iloc[val_indices], val_prediction))

params = {}
params['learning_rate'] = (0, 100.0)
params['alpha'] = (0, 100.0)
params['gamma'] = (0, 100.0)
params['lambda'] = (0, 100.0)
params['subsample'] = (1e-9, 0.99999)
params['min_child_weight'] = (0, 50.0)
params['max_delta_step'] = (0, 10.0)
params['n_estimators'] = (1, 5000)
params['max_depth'] = (1, 50)
params['eval_metric'] = ['auc']

clf = xg.XGBClassifier(use_label_encoder=False, random_state=seed, n_jobs=-1)
cv = StratifiedKFold(n_splits=10)
search = BayesSearchCV(clf, search_spaces=params, n_jobs=-1, cv=cv)
search.fit(train_x, np.resize(train_y, (len(train_y), )))
print('Best score: ', search.best_score_)
print('Best params: ', search.best_params_)

clf = search.best_estimator_
model_preds = clf.predict(test_to_model)
df_model_preds = pd.DataFrame(model_preds)

test_predictions = df_model_preds.replace([True, False], [1, 0])
test_result = passenger_id.join(test_predictions)
test_result.columns = ['PassengerId', 'Survived']
test_result.to_csv('titanic_xg_result.csv', index=False)

# Best validation accuracy: 0.78
# submission score 0.76555