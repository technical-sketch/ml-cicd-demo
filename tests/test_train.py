
import math

from src.train import load_data, train_and_evaluate


def test_model_predictions_shape():
    X_train, X_test, y_train, y_test = load_data()

    model, accuracy, f1 = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape, (
        "Prediction shape must match test label shape"
    )


def test_metrics_are_finite():
    X_train, X_test, y_train, y_test = load_data()

    _, accuracy, f1 = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    assert math.isfinite(accuracy), (
        "Accuracy must be a finite number"
    )
    assert math.isfinite(f1), (
        "F1-score must be a finite number"
    )


def test_metrics_in_valid_range():
    X_train, X_test, y_train, y_test = load_data()

    _, accuracy, f1 = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0
