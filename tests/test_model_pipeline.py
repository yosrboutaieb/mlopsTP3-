import numpy as np
import pandas as pd

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def make_fake_csv(path, n=60):
    """Crée un petit CSV factice au format Churn_Modelling."""
    df = pd.DataFrame(
        {
            "RowNumber": range(n),
            "CustomerId": range(1000, 1000 + n),
            "Surname": [f"Name{i}" for i in range(n)],
            "CreditScore": np.random.randint(400, 850, n),
            "Geography": ["France", "Spain", "Germany"] * (n // 3),
            "Gender": ["Male", "Female"] * (n // 2),
            "Age": np.random.randint(18, 80, n),
            "Tenure": np.random.randint(0, 10, n),
            "Balance": np.random.rand(n) * 100000,
            "NumOfProducts": np.random.randint(1, 4, n),
            "HasCrCard": np.random.randint(0, 2, n),
            "IsActiveMember": np.random.randint(0, 2, n),
            "EstimatedSalary": np.random.rand(n) * 100000,
            "Exited": np.random.randint(0, 2, n),
        }
    )
    df.to_csv(path, index=False)


def test_train_and_predict():
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    model = train_model(X, y)
    assert len(model.predict(X)) == 50


def test_save_and_load(tmp_path):
    X = np.random.rand(30, 4)
    y = np.random.randint(0, 2, 30)
    model = train_model(X, y)
    path = str(tmp_path / "m.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert (loaded.predict(X) == model.predict(X)).all()


def test_prepare_data_shapes(tmp_path):
    csv = tmp_path / "fake.csv"
    make_fake_csv(csv)
    X_train, X_test, y_train, y_test = prepare_data(str(csv))
    assert len(X_train) + len(X_test) == 60
    assert len(X_train) == len(y_train)
    assert X_train.shape[1] == 11


def test_evaluate_model_returns_valid_accuracy():
    X = np.random.rand(40, 5)
    y = np.random.randint(0, 2, 40)
    model = train_model(X, y)
    acc = evaluate_model(model, X, y)
    assert 0.0 <= acc <= 1.0
