import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import build_preprocessor
import pandas as pd

arboles = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_split=2, min_samples_leaf=1)
knn = KNeighborsClassifier(n_neighbors=3, p=2)

X_train = pd.read_csv('../data/processed/Xtrain.csv')
Y_train = pd.read_csv('../data/processed/Ytrain.csv')

def train_and_save_model(X, y, model_path, model):

    preprocessor = build_preprocessor(X)

    # Pipeline completo
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Entrenar
    model_pipeline.fit(X, y)

    # Exportar
    joblib.dump(model_pipeline, model_path)
    print(f"Modelo guardado en {model_path}")

train_and_save_model(X_train, Y_train, '../models/decision_tree_model.pkl', arboles)
train_and_save_model(X_train, Y_train, '../models/knn_model.pkl', knn)