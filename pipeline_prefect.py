import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd, check=True):
    """Run a shell command and print its output."""
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.returncode


# ---------- Code tasks ----------
@task(name="install_dependencies")
def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format_code")
def format_code():
    run([sys.executable, "-m", "black", *FILES])


@task(name="code_quality")
def code_quality():
    run([sys.executable, "-m", "flake8", "--max-line-length=100", *FILES], check=False)


@task(name="code_security")
def code_security():
    run([sys.executable, "-m", "bandit", "-q", *FILES], check=False)


@task(name="unit_tests")
def unit_tests():
    run([sys.executable, "-m", "pytest", "tests", "-q"])


# ---------- ML tasks ----------
@task(name="prepare_data")
def prepare_task():
    return prepare_data(DATA_PATH)


@task(name="train_model")
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="save_model")
def save_task(model):
    save_model(model)


@task(name="load_model")
def load_task():
    return load_model()


@task(name="evaluate_model")
def evaluate_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


# ---------- Flows ----------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, X_test, y_test)


@flow(name="all")
def all_flow():
    code_flow()
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    evaluate_task(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        default="all",
        choices=["all", "train", "entrainement", "evaluate", "code", "install"],
    )
    args = parser.parse_args()
    flows = {
        "all": all_flow,
        "train": train_flow,
        "entrainement": train_flow,
        "evaluate": evaluate_flow,
        "code": code_flow,
        "install": install_flow,
    }
    flows[args.flow]()
