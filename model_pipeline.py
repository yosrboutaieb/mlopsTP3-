import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def prepare_data(path, target="Exited", test_size=0.2, random_state=42):
    """Charge le CSV, prétraite et retourne X_train, X_test, y_train, y_test."""
    df = pd.read_csv(path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df = df.dropna()
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne et retourne le modèle."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Affiche et retourne les performances."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc


def save_model(model, path="model.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé."""
    return joblib.load(path)
