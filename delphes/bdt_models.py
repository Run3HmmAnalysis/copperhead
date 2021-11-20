# import xgboost as xgb
from xgboost import XGBClassifier

test_bdt = XGBClassifier(
    max_depth=7,  # for 2018
    # max_depth=6,previous value
    n_estimators=100,
    # n_estimators=100,
    # objective='multi:softmax',
    objective="binary:logistic",
    num_class=1,
    # learning_rate=0.001,#for 2018
    # learning_rate=0.0034,#previous value
    # reg_alpha=0.680159426755822,
    # colsample_bytree=0.47892268305051233,
    min_child_weight=20,
    # subsample=0.5606,
    # reg_lambda=16.6,
    # gamma=24.505,
    # n_jobs=5,
    tree_method="hist",
)
