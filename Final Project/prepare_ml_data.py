import pandas as pd
from sklearn.model_selection import train_test_split

def add_delta_features(df):
    df = df.copy()
    df["delta_lon"] = df["lon_last"] - df["lon_1st"]
    df["delta_lat"] = df["lat_last"] - df["lat_1st"]
    return df

def origin_call_flg(x):
    return 0 if pd.isna(x["ORIGIN_CALL"]) else 1

def origin_stand_flg(x):
    return 0 if pd.isna(x["ORIGIN_STAND"]) else 1

def miss_flg(x):
    return 0 if str(x["MISSING_DATA"]).lower() == "false" else 1

def add_flag_features(df):
    df = df.copy()
    df["ORIGIN_CALL"] = df.apply(origin_call_flg, axis=1)
    df["ORIGIN_STAND"] = df.apply(origin_stand_flg, axis=1)
    df["MISSING_DATA"] = df.apply(miss_flg, axis=1)
    return df

def prepare_ml_data(train_path, test_path, sample_size=136000, test_split=0.3, random_state=1):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = add_delta_features(train)
    test = add_delta_features(test)

    train = add_flag_features(train)
    test = add_flag_features(test)

    ml_train = train.sample(sample_size, random_state=random_state) if sample_size < len(train) else train.copy()
    ml_test = test.copy()

    feature_cols = [
        "call_type_a", "call_type_b", "call_type_c",
        "ORIGIN_CALL", "ORIGIN_STAND", "MISSING_DATA",
        "lon_1st", "lat_1st", "delta_lon", "delta_lat"
    ]
    target_cols = ["lon_last", "lat_last"]

    X = ml_train[feature_cols]
    y = ml_train[target_cols]
    X_Test = ml_test[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
    return X_train, X_test, y_train, y_test, X_Test
