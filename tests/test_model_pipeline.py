import numpy as np
from model_pipeline import train_model, save_model, load_model


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
