import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)

    if args.prepare:
        print("Données prêtes :", X_train.shape, X_test.shape)

    if args.train:
        model = train_model(X_train, y_train)
        save_model(model)

    if args.evaluate:
        model = load_model()
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
