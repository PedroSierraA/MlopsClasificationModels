from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

def build_preprocessor(X):
    num_col = X.select_dtypes(include=['int64', 'float64']).columns.to_list()
    cat_col = X.select_dtypes(include=['object']).columns.to_list()

    num_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    cat_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_col),
            ('cat', cat_transformer, cat_col)
        ]
    )
    return preprocessor
