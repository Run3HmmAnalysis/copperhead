def train_dnn(df, cat_name, df_info, features):
    print()
    print(cat_name)
    print(df["class"].value_counts())
    print()
    ignore_features = df.loc[:, df.min(axis=0) == -999.0].columns
    features_clean = []
    for c in df.columns:
        if (c in features) and (c not in ignore_features):
            features_clean.append(c)
    print(features_clean)
    return
