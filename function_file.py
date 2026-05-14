import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create a copy to keep your original data safe
def standardize_columns(X_train):
    df_encoded = X_train.copy()

    # 2. Map ordinal features (where order matters)
    work_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    df_encoded['work_interfere'] = df_encoded['work_interfere'].fillna('Never').map(work_map)

    # 3. Map binary/ternary features
    # We give "Don't know" a 0.5 because it's a middle-ground response
    support_map = {'Yes': 1, 'No': 0, "Don't know": 0.5, 'Not sure': 0.5}
    df_encoded['family_history'] = df_encoded['family_history'].map({'Yes': 1, 'No': 0})
    df_encoded['benefits'] = df_encoded['benefits'].map(support_map)

    # 'phys_health_consequence' - Is there a physical health stigma?
    # Higher score = more fear of consequences
    consequence_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
    df_encoded['phys_health_consequence'] = df_encoded['phys_health_consequence'].map(consequence_map)

    # Higher score = more comfort/support
    comfort_map = {'No': 0, 'Some of them': 1, 'Yes': 2}
    df_encoded['coworkers'] = df_encoded['coworkers'].map(comfort_map)

    return df_encoded

def one_hot_encoding(X_train, encoder):
    color_encoded = encoder.transform(X_train[["work_interfere","family_history","benefits","phys_health_consequence","coworkers"]])
    df = pd.DataFrame(color_encoded, columns=encoder.get_feature_names_out(["work_interfere","family_history","benefits","phys_health_consequence","coworkers"]))
    X_train = X_train.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([X_train, df], axis=1).reset_index(drop=True)
    df = df.drop(columns=["work_interfere","family_history","benefits","phys_health_consequence","coworkers"])
    return df

def normalize_data(df, normalizer):
    X_train_norm = normalizer.transform(df)
    X_train_norm  = pd.DataFrame(X_train_norm,columns=df.columns)
    return X_train_norm