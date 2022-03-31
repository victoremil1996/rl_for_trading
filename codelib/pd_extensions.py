from pandas.core.base import PandasObject


def create_features(data, cols=["price"]):
    data["ma_5"] = data[cols].rolling(window=5).mean()
    data["ma_50"] = data[cols].rolling(window=50).mean()
    return data


def extend_pandas():
    PandasObject.create_features = create_features
