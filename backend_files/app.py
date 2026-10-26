# Import necessary libraries
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

extraalearn_conversion_api = Flask("ExtraaLearn Lead Converison Predictor")
extraalearn_conversion_api.logger.setLevel(logging.INFO)

MODEL_PATH = Path(__file__).resolve().parent / "extraalearn_model.joblib"
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "age",
    "current_occupation",
    "first_interaction",
    "profile_completed",
    "website_visits",
    "time_spent_on_website",
    "page_views_per_visit",
    "last_activity",
    "print_media_type1",
    "print_media_type2",
    "digital_media",
    "educational_channels",
    "referral",
]

TARGET_COLUMN = "status"


def normalize_input_frame(frame):
    """Normalize categorical values without inventing feature values."""

    frame = frame.copy()

    categorical_columns = [
        "current_occupation",
        "first_interaction",
        "profile_completed",
        "last_activity",
    ]

    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype("string")
                .str.strip()
            )

    return frame


def validate_features(frame):
    """Validate input features and preserve training column order."""

    if frame.empty:
        return None, "Input contains no rows."

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        return None, f"Missing required feature columns: {missing}"

    frame = frame.loc[:, FEATURE_COLUMNS].copy()
    frame = normalize_input_frame(frame)

    return frame, None


def get_conversion_label(prediction):
    """Convert the model's predicted class into a readable label."""

    # Adjust these mappings to match your training target encoding.
    if prediction in (1, True):
        return "Yes"

    if prediction in (0, False):
        return "No"

    return str(prediction)


def get_conversion_probability(model, frame):
    """Return the probability of conversion when supported."""

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)

    # Adjust the positive-class definition if your target uses
    # labels other than 1/0 or Yes/No.
    positive_labels = {"1", "true", "yes", "converted"}

    positive_index = next(
        (
            index
            for index, label in enumerate(classes)
            if str(label).strip().lower() in positive_labels
        ),
        None,
    )

    if positive_index is None:
        return None

    return probabilities[:, positive_index]


@extraalearn_conversion_api.post("/v1/extraalearn")
def predict_extraalearn_conversion():
    lead_data = request.get_json(silent=True)

    if not isinstance(lead_data, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    frame, error = validate_features(
        pd.DataFrame([lead_data])
    )

    if error:
        return jsonify({"error": error}), 400

    try:
        prediction = model.predict(frame)[0]

        result = {
            "conversion_prediction": get_conversion_label(
                prediction
            )
        }

        probabilities = get_conversion_probability(model, frame)

        if probabilities is not None:
            result["conversion_probability"] = round(
                float(probabilities[0]), 4
            )

        return jsonify(result), 200

    except Exception:
        extraalearn_conversion_api.logger.exception(
            "ExtraaLearn prediction failed"
        )

        return jsonify({
            "error": (
                "Prediction failed. "
                "Check backend container logs for details."
            )
        }), 500
