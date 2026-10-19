import streamlit as st
import pandas as pd
from joblib import load
import random
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    mean_absolute_error,
    mean_squared_error
)
from math import sqrt
import matplotlib.pyplot as plt
import pandasql as ps


results = None


model = load("knn_model.joblib")
scaler = load("scaler.joblib")
test_dataset = pd.read_csv("test_data.csv")
selected_features = [
    "clicks",
    "median_click_interval_ms",
    "click_interval_std_ms",
    "mouse_path_curvature",
    "mouse_distance_px",
    "idle_ratio",
    "scroll_entropy",
    "digraph_latency_ms",
    "avg_dwell_time",
    "pages_visited",
    "referrer_type_ad",
    "referrer_type_search",
    "button_hover_ms"
]

X_test = test_dataset[selected_features]
Y_test = test_dataset["bot"]

X_test_scaled = scaler.transform(X_test)
Y_prediction = model.predict(X_test_scaled)

# Evaluation metrics
accuracy = accuracy_score(Y_test, Y_prediction)
precision = precision_score(Y_test, Y_prediction, zero_division=0)

mae = mean_absolute_error(Y_test, Y_prediction)
mse = mean_squared_error(Y_test, Y_prediction)
rmse = sqrt(mse)

# Confusion matrix: 0 = Human, 1 = Bot
tn, fp, fn, tp = confusion_matrix(
    Y_test, Y_prediction, labels=[0, 1]
).ravel()

# Percentages of all test records
total = len(Y_test)

tn_pct = tn / total * 100
fp_pct = fp / total * 100
fn_pct = fn / total * 100
tp_pct = tp / total * 100


#UI Design

plt.rcParams.update({
    "figure.facecolor": "#0b1120",
    "axes.facecolor": "#151f32",
    "axes.labelcolor": "#cbd5e1",
    "axes.titlecolor": "#f1f5f9",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "text.color": "#f1f5f9",
    "legend.facecolor": "#151f32",
    "legend.edgecolor": "#26354d",
    "legend.labelcolor": "#cbd5e1",
    "axes.edgecolor": "#26354d",
})


st.title("🛡️ Bot VS HUMAN")
st.markdown(
    "Monitor session behavior, identify automated traffic, "
    "and explore model predictions."
)

st.caption("MACHINE LEARNING  /  SESSION ANALYSIS")
st.divider()


st.subheader("Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.1%}")

with col2:
    st.metric("Precision", f"{precision:.1%}")

with col3:
    st.metric("Total Sessions", f"{total:,}")

with col4:
    st.metric("Bots Detected", f"{tp + fp:,}")

st.divider()

chart_choice = st.radio(
    "Choose what to display:",
    [
        "Prediction Outcomes",
        "Evaluation Metrics",
        "Actual vs Predicted"
    ],
    horizontal=True
)

if chart_choice == "Prediction Outcomes":
    labels = ["TP", "TN", "FP", "FN"]
    values = [tp, tn, fp, fn]

    fig, ax = plt.subplots()

    wedges, _, _ = ax.pie(
        values,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.legend(
        wedges,
        labels,
        title="Outcomes",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    ax.set_title("Prediction Outcome Distribution")
    ax.axis("equal")

    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()

    labels = ["TP", "TN", "FP", "FN"]
    values = [tp, tn, fp, fn]

    bars = ax.bar(
        ["TP", "TN", "FP", "FN"],
        [tp, tn, fp, fn],
        color=["#22c55e", "#38bdf8", "#f97316", "#ef4444"]
    )
    ax.set_title("Confusion Matrix Results")
    ax.set_xlabel("Prediction Category")
    ax.set_ylabel("Number of Sessions")

    st.pyplot(fig)
    plt.close(fig)

elif chart_choice == "Evaluation Metrics":
    metric_names = ["Accuracy", "Precision", "MAE", "MSE", "RMSE"]
    metric_values = [accuracy, precision, mae, mse, rmse]
    metrics = {
    "Accuracy": accuracy_score(Y_test, Y_prediction),
    "Precision": precision_score(Y_test, Y_prediction, zero_division=0),
    "MAE": mean_absolute_error(Y_test, Y_prediction),
    "MSE": mean_squared_error(Y_test, Y_prediction),
    "RMSE": sqrt(mean_squared_error(Y_test, Y_prediction))
    }
    col1, col2, col3 = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):
        with [col1, col2, col3][i % 3]:
            st.metric(
                label=name,
                value=f"{value:.4f}"
            )
    fig, ax = plt.subplots()
    ax.bar(metric_names, metric_values)
    ax.set_title("Model Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)

    st.pyplot(fig)
    plt.close(fig)

elif chart_choice == "Actual vs Predicted":
    results = pd.DataFrame({
        "Actual": Y_test.tolist(),
        "Predicted": list(Y_prediction)
    })

    results["Correct"] = (
        results["Actual"] == results["Predicted"]
    )

    st.dataframe(results, hide_index=True)

    counts = results["Correct"].value_counts().reindex(
        [True, False], fill_value=0
    )

    fig, ax = plt.subplots()
    ax.bar(
        ["Correct", "Incorrect"],
        [counts[True], counts[False]]
    )

    ax.set_title("Correct vs Incorrect Predictions")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)
    plt.close(fig)



st.subheader("Bot Prediction")

st.write("Enter values below for prediction")

if st.button("🎲 Randomize Inputs"):
    for feature in selected_features:
        if feature in [
            "avg_dwell_time",
            "dwell_field1_ms",
            "dwell_field2_ms",
            "dwell_field3_ms",
            "referrer_type",
            "referrer_type_ad",
            "referrer_type_direct",
            "referrer_type_social",
            "referrer_type_search"
        ]:
            continue

        if feature in ["mouse_path_curvature", "idle_ratio"]:
            st.session_state[feature] = round(random.uniform(0, 1), 2)
        elif feature in ["timezone_mismatch", "webdriver_flag"]:
            st.session_state[feature] = random.randint(0, 1)
        else:
            st.session_state[feature] = round(random.uniform(1, 1000), 2)

    # Random dwell times
    st.session_state["dwell_field1_ms"] = random.randint(50, 2000)
    st.session_state["dwell_field2_ms"] = random.randint(50, 2000)
    st.session_state["dwell_field3_ms"] = random.randint(50, 2000)

    # Random referrer type
    st.session_state["referrer_type_choice"] = random.choice(
        ["direct", "ad", "search"]
    )


with st.form("bot_prediction_form"):
    user_input = {}
    dwell_values = {}

    col1, col2 = st.columns(2)

    zero_to_one_features = ["mouse_path_curvature", "idle_ratio"]
    binary_features = ["timezone_mismatch", "webdriver_flag"]

    excluded_features = [
        "avg_dwell_time", "dwell_field1_ms", "dwell_field2_ms",
        "dwell_field3_ms", "referrer_type", "referrer_type_ad",
        "referrer_type_direct", "referrer_type_social",
        "referrer_type_search"
    ]

    input_index = 0

    for feature in selected_features:
        if feature in excluded_features:
            continue

        with [col1, col2][input_index % 2]:
            if feature in zero_to_one_features:
                user_input[feature] = st.slider(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    key=feature
                )
            elif feature in binary_features:
                user_input[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=[0, 1],
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                    key=feature
                )
            else:
                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=None,
                    placeholder="Enter a value",
                    step=1.0,
                    format="%.2f",
                    key=feature
                )

        input_index += 1

    st.write("### Dwell Times")
    dwell_col1, dwell_col2 = st.columns(2)

    with dwell_col1:
        dwell_values["dwell_field1_ms"] = st.number_input(
            "Dwell Field 1 (ms)",
            min_value=0.0,
            value=None,
            placeholder="Enter a value",
            key="dwell_field1_ms"
        )

        dwell_values["dwell_field3_ms"] = st.number_input(
            "Dwell Field 3 (ms)",
            min_value=0.0,
            value=None,
            placeholder="Enter a value",
            key="dwell_field3_ms"
        )

    with dwell_col2:
        dwell_values["dwell_field2_ms"] = st.number_input(
            "Dwell Field 2 (ms)",
            min_value=0.0,
            value=None,
            placeholder="Enter a value",
            key="dwell_field2_ms"
        )

    referrer_type = st.selectbox(
        "Referrer Type",
        options=["direct", "ad", "search"],
        key="referrer_type_choice"
    )

    submitted = st.form_submit_button("Predict")


if submitted:
    missing_fields = [
        feature for feature, value in user_input.items()
        if value is None
    ]

    missing_fields.extend(
        feature for feature, value in dwell_values.items()
        if value is None
    )

    if missing_fields:
        st.warning("Please fill in all numerical fields before predicting.")
    else:
        user_input["avg_dwell_time"] = sum(dwell_values.values()) / 3

        user_input["referrer_type_ad"] = int(referrer_type == "ad")
        user_input["referrer_type_direct"] = int(referrer_type == "direct")
        user_input["referrer_type_social"] = 0
        user_input["referrer_type_search"] = int(referrer_type == "search")

        input_df = pd.DataFrame(
            [user_input],
            columns=selected_features
        )

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("Prediction: BOT")
        else:
            st.success("Prediction: HUMAN")

        st.write("### Input Values")
        st.dataframe(input_df)


st.subheader("Upload Session Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    key="session_csv"
)

if uploaded_file is not None:

    uploaded_data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Sessions")
    st.dataframe(uploaded_data, use_container_width=True)

    # Calculate average dwell time if needed
    dwell_fields = [
        "dwell_field1_ms",
        "dwell_field2_ms",
        "dwell_field3_ms"
    ]

    if (
        "avg_dwell_time" not in uploaded_data.columns
        and all(f in uploaded_data.columns for f in dwell_fields)
    ):
        uploaded_data["avg_dwell_time"] = (
            uploaded_data[dwell_fields].mean(axis=1)
        )

    # Check whether all model features are available
    missing_features = [
        f for f in selected_features
        if f not in uploaded_data.columns
    ]

    if missing_features:
        st.error(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    else:
        prediction_results = None
        if st.button("Predict All Sessions"):
            input_data = uploaded_data[selected_features].copy()

            # Predict using the existing model and scaler
            input_scaled = scaler.transform(input_data)
            predictions = model.predict(input_scaled)

            # Save results so they survive reruns
            prediction_results = uploaded_data.copy()
            prediction_results["Prediction"] = [
                "BOT" if p == 1 else "HUMAN"
                for p in predictions
            ]

            st.session_state["uploaded_results"] = prediction_results

        # Display saved results outside the button condition
        results = st.session_state.get("uploaded_results")

        if results is not None:
            st.write("### Prediction Results")
            st.dataframe(results, use_container_width=True)

            st.write("### Prediction Summary")
            st.write(results["Prediction"].value_counts())

            st.download_button(
                "Download Results as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="bot_predictions.csv",
                mime="text/csv"
            )

            # ---------------- FILTERS ----------------
            st.subheader("Filter Sessions")
            filtered_data = results.copy()
            with st.expander("Choose filters", expanded=True):

                # Filter by prediction
                selected_classes = st.multiselect(
                    "Prediction",
                    ["BOT", "HUMAN"],
                    default=["BOT", "HUMAN"]
                )

                filtered_data = filtered_data[
                    filtered_data["Prediction"].isin(selected_classes)
                ]

                # Filter by referrer type
                referrer_columns = {
                    "ad": "referrer_type_ad",
                    "direct": "referrer_type_direct",
                    "social": "referrer_type_social",
                    "search": "referrer_type_search"
                }

                available_referrers = [
                    name for name, column in referrer_columns.items()
                    if column in filtered_data.columns
                ]

                if available_referrers:
                    selected_referrers = st.multiselect(
                        "Referrer Type",
                        available_referrers,
                        default=available_referrers
                    )

                    selected_columns = [
                        referrer_columns[name]
                        for name in selected_referrers
                    ]

                    filtered_data = filtered_data[
                        filtered_data[selected_columns].eq(1).any(axis=1)
                    ]

                # Filter by numerical feature using a range slider
                numerical_features = [
                    feature for feature in selected_features
                    if feature in filtered_data.columns
                    and feature not in referrer_columns.values()
                    and pd.api.types.is_numeric_dtype(filtered_data[feature])
                ]

                if numerical_features and not filtered_data.empty:
                    selected_feature = st.selectbox(
                        "Numerical Feature",
                        numerical_features
                    )

                    minimum = float(filtered_data[selected_feature].min())
                    maximum = float(filtered_data[selected_feature].max())

                    if minimum < maximum:
                        selected_range = st.slider(
                            f"Range for {selected_feature}",
                            min_value=minimum,
                            max_value=maximum,
                            value=(minimum, maximum)
                        )

                        filtered_data = filtered_data[
                            filtered_data[selected_feature].between(
                                selected_range[0],
                                selected_range[1]
                            )
                        ]

                # Display filtered results
                st.write(
                    f"Showing {len(filtered_data)} of {len(results)} sessions"
                )

                st.dataframe(filtered_data, use_container_width=True)

        #Find the relations
        if st.session_state.get("uploaded_results") is not None:
            dataset = st.session_state["uploaded_results"].copy()
            selected_query = st.selectbox(
                "Select a SQL query",
                [
                    "Show all bot sessions",
                    "Show all non-bot sessions",
                    "Compare session duration between bots and humans",
                    "Compare clicks",
                    "Compare mouse path curvature",
                    "Compare idle ratio",
                    "Compare scroll entropy",
                    "Compare digraph latency",
                    "Compare timezone mismatch",
                    "Compare webdriver flag",
                    "Compare time hovering over the submit button",
                    "Compare copy actions",
                    "Compare  browser width in pixels"
                ]
            )

            if selected_query == "Show all bot sessions":
                sql = """
                    SELECT *
                    FROM dataset
                    WHERE Prediction = 'BOT'
                """

            elif selected_query == "Show all non-bot sessions":
                sql = """
                    SELECT *
                    FROM dataset
                    WHERE Prediction = 'HUMAN' 
                """
            elif selected_query == "Compare session duration between bots and humans":
                sql = """
                    SELECT Prediction, AVG(session_duration_ms) as Average_session_duration_in_ms
                    FROM dataset 
                    GROUP BY Prediction
                """

            elif selected_query == "Compare clicks":
                sql = """
                    SELECT Prediction, AVG(clicks) AS Average_amount_of_clicks
                    FROM dataset 
                    GROUP BY Prediction
                """

            elif selected_query == "Compare mouse path curvature":
                sql = """
                    SELECT Prediction, AVG(mouse_path_curvature) AS Average_mouse_path_curvature
                    FROM dataset 
                    GROUP BY Prediction
                """
            elif selected_query == "Compare idle ratio":
                sql = """
                    SELECT Prediction, AVG(idle_ratio) AS Average_idle_ratio
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare scroll entropy":
                sql = """
                    SELECT Prediction, AVG(scroll_entropy) AS Average_scroll_entropy
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare digraph latency":
                sql = """
                    SELECT Prediction, AVG(digraph_latency_ms) AS Average_digraph_latency
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare timezone mismatch":
                sql = """
                    SELECT Prediction, AVG(timezone_mismatch) AS Average_timezone_mismatch
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare webdriver flag":
                sql = """
                    SELECT Prediction, AVG(webdriver_flag) AS Average_amount_of_times_the_browser_exposes_the_automation_flag
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare time hovering over the submit button":  
                sql = """
                    SELECT Prediction, AVG(button_hover_ms) AS Average_amount_of_time_hovering
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare copy actions":
                sql = """
                    SELECT Prediction, AVG(copy_events)  AS Copy_actions
                    FROM dataset 
                    GROUP BY Prediction 
                """
            elif selected_query == "Compare  browser width in pixels":
                sql = """
                    SELECT Prediction, AVG(viewport_width) AS Average_viewport_width
                    FROM dataset 
                    GROUP BY Prediction 
                """

            result = ps.sqldf(sql, {"dataset": dataset})

            st.subheader("Query Results")
            st.dataframe(result)
        