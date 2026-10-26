import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")
REQUEST_TIMEOUT = 60

st.set_page_config(
    page_title="ExtraaLearn Lead Conversion Prediction",
    layout="centered",
)

st.title("ExtraaLearn Lead Conversion Prediction")
st.caption("Predict whether a lead will convert into a paid customer.")

# Feature options
OCCUPATIONS = ["Professional", "Unemployed", "Student"]
FIRST_INTERACTIONS = ["Website", "Mobile App"]
PROFILE_COMPLETED = ["Low", "Medium", "High"]
LAST_ACTIVITIES = [
    "Email Activity",
    "Phone Activity",
    "Website Activity",
]

MEDIA_OPTIONS = ["No", "Yes"]

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

# --------------------------------------------------
# Online prediction
# --------------------------------------------------

st.subheader("Online prediction")

with st.form("online_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

        current_occupation = st.selectbox(
            "Current Occupation",
            OCCUPATIONS,
        )

        first_interaction = st.selectbox(
            "First Interaction",
            FIRST_INTERACTIONS,
        )

        profile_completed = st.selectbox(
            "Profile Completed",
            PROFILE_COMPLETED,
        )

        website_visits = st.number_input(
            "Website Visits",
            min_value=0,
            value=5,
            step=1,
        )

        time_spent_on_website = st.number_input(
            "Time Spent on Website (seconds)",
            min_value=0,
            value=300,
            step=10,
        )

        page_views_per_visit = st.number_input(
            "Average Page Views per Visit",
            min_value=0.0,
            value=3.0,
            step=0.1,
        )

    with col2:
        last_activity = st.selectbox(
            "Last Activity",
            LAST_ACTIVITIES,
        )

        print_media_type1 = st.selectbox(
            "Seen Newspaper Advertisement?",
            MEDIA_OPTIONS,
        )

        print_media_type2 = st.selectbox(
            "Seen Magazine Advertisement?",
            MEDIA_OPTIONS,
        )

        digital_media = st.selectbox(
            "Seen Digital Advertisement?",
            MEDIA_OPTIONS,
        )

        educational_channels = st.selectbox(
            "Heard Through Educational Channels?",
            MEDIA_OPTIONS,
        )

        referral = st.selectbox(
            "Heard Through Referral?",
            MEDIA_OPTIONS,
        )

    predict_clicked = st.form_submit_button(
        "Predict Conversion",
        type="primary",
    )

if predict_clicked:

    input_payload = {
        "age": int(age),
        "current_occupation": current_occupation,
        "first_interaction": first_interaction,
        "profile_completed": profile_completed,
        "website_visits": int(website_visits),
        "time_spent_on_website": int(time_spent_on_website),
        "page_views_per_visit": float(page_views_per_visit),
        "last_activity": last_activity,
        "print_media_type1": 1 if print_media_type1 == "Yes" else 0,
        "print_media_type2": 1 if print_media_type2 == "Yes" else 0,
        "digital_media": 1 if digital_media == "Yes" else 0,
        "educational_channels": (
            1 if educational_channels == "Yes" else 0
        ),
        "referral": 1 if referral == "Yes" else 0,
    }

    st.dataframe(
        pd.DataFrame([input_payload]),
        use_container_width=True,
    )

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/extraalearn",
            json=input_payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.ok:
            result = response.json()

            prediction = result["conversion_prediction"]

            if str(prediction).strip().lower() in (
                "yes", "1", "true", "converted"
            ):
                st.success("Prediction: Lead is likely to convert.")

            elif str(prediction).strip().lower() in (
                "no", "0", "false", "not converted"
            ):
                st.warning("Prediction: Lead is unlikely to convert.")

            else:
                st.info(f"Conversion Prediction: {prediction}")

            if "conversion_probability" in result:
                probability = float(
                    result["conversion_probability"]
                )

                st.metric(
                    "Conversion Probability",
                    f"{probability:.2%}",
                )

        else:
            st.error(
                f"Backend returned HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )

    except requests.RequestException as exc:
        st.error(
            f"Could not reach the backend at {BACKEND_URL}. "
            f"Details: {exc}"
        )

    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unexpected API response: {exc}")
