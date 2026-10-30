
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "student-mat.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
PREDICTIONS_PATH = BASE_DIR / "test_predictions.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"

FEATURES = [
    "age",
    "studytime",
    "failures",
    "schoolsup",
    "internet",
    "absences",
]

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. LOAD PROJECT FILES
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def require_files():
    required_files = [
        DATA_PATH,
        MODEL_PATH,
        METRICS_PATH,
        PREDICTIONS_PATH,
        IMPORTANCE_PATH,
    ]

    missing = [path.name for path in required_files if not path.exists()]

    if missing:
        st.error(
            "Some project files are missing: "
            + ", ".join(missing)
        )
        st.info(
            "Make sure student-mat.csv is present, then run "
            "`python train_model.py` to generate the model files."
        )
        st.stop()


require_files()

df = load_dataset()
model = load_model()
metrics = load_json(METRICS_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)
importance = pd.read_csv(IMPORTANCE_PATH)

# Prepare evaluation metrics globally so every page can use them.
metrics_df = pd.DataFrame(metrics).T
metrics_df.index.name = "Model"
metrics_df = metrics_df.reset_index()


# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎓 Student Performance ML")
st.sidebar.write("Machine Learning Mini Project")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Predict Performance",
        "Model Evaluation",
        "Dataset Analysis",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Models: Linear Regression and Random Forest"
)


# --------------------------------------------------
# 4. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🎓 Student Performance Prediction")

    st.write(
        "Explore student grades and understand how study habits "
        "and other factors relate to academic performance."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(df))
    col2.metric(
        "Average Final Grade",
        f"{df['G3'].mean():.2f}/20",
    )
    col3.metric(
        "Highest Final Grade",
        f"{df['G3'].max()}/20",
    )
    col4.metric(
        "Students Scoring 10+",
        f"{(df['G3'] >= 10).sum()}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Distribution of Final Grades")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            data=df,
            x="G3",
            bins=11,
            kde=True,
            ax=ax,
        )

        ax.set_xlabel("Final Grade (0–20)")
        ax.set_ylabel("Number of Students")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Study Time vs Final Grade")

        study_labels = {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours",
        }

        plot_df = df.copy()
        plot_df["Study Time"] = plot_df["studytime"].map(
            study_labels
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.boxplot(
            data=plot_df,
            x="Study Time",
            y="G3",
            ax=ax,
        )

        ax.set_xlabel("Weekly Study Time")
        ax.set_ylabel("Final Grade (0–20)")
        ax.tick_params(axis="x", rotation=15)

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Key Observations")

    average_grade = df["G3"].mean()
    pass_rate = (df["G3"] >= 10).mean() * 100

    st.write(
        f"- The average final grade in this dataset is "
        f"{average_grade:.2f} out of 20."
    )

    st.write(
        f"- {pass_rate:.1f}% of students scored at least 10/20."
    )

    st.caption(
        "These are descriptive statistics for this dataset, "
        "not proof that any single factor causes higher grades."
    )


# --------------------------------------------------
# 5. PREDICT STUDENT PERFORMANCE
# --------------------------------------------------

elif page == "Predict Performance":

    st.title("🔮 Predict Student Performance")

    st.write(
        "Enter the student's details to estimate their final grade. "
        "This is an estimate from the trained Random Forest model, "
        "not a guaranteed result."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.slider(
                "Student Age",
                min_value=15,
                max_value=22,
                value=17,
            )

            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - Less than 2 hours",
                    2: "2 - 2 to 5 hours",
                    3: "3 - 5 to 10 hours",
                    4: "4 - More than 10 hours",
                }[x],
            )

            failures = st.selectbox(
                "Number of Past Class Failures",
                options=[0, 1, 2, 3],
            )

        with col2:

            absences = st.number_input(
                "Number of School Absences",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )

            schoolsup = st.selectbox(
                "Extra Educational Support",
                options=["no", "yes"],
            )

            internet = st.selectbox(
                "Internet Access at Home",
                options=["yes", "no"],
            )

        submitted = st.form_submit_button(
            "Predict Final Grade",
            type="primary",
        )

    if submitted:

        input_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "studytime": studytime,
                    "failures": failures,
                    "schoolsup": 1 if schoolsup == "yes" else 0,
                    "internet": 1 if internet == "yes" else 0,
                    "absences": absences,
                }
            ],
            columns=FEATURES,
        )

        predicted_grade = float(model.predict(input_data)[0])

        # Keep the displayed estimate within the grade scale.
        predicted_grade = max(
            0.0,
            min(20.0, predicted_grade),
        )

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Estimated Final Grade",
            f"{predicted_grade:.2f}/20",
        )

        if predicted_grade >= 10:
            result_col2.success(
                "Estimated grade is in the passing range."
            )
        else:
            result_col2.warning(
                "Estimated grade is below the 10/20 threshold."
            )

        st.progress(predicted_grade / 20)

        st.caption(
            "The estimate is based on patterns in the training data. "
            "Actual grades may differ, and the model is not suitable "
            "for making high-stakes decisions about students."
        )


# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write(
        "Compare Linear Regression and Random Forest using the "
        "same held-out test dataset."
    )

    # metrics_df is already defined above for all pages.

    st.subheader("Performance Metrics")

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Lower MAE and RMSE generally indicate smaller prediction "
        "errors. Higher R² generally indicates a better fit. "
        "R² can be negative when predictions perform poorly."
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Actual vs Predicted Grades")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            predictions["Actual Grade"],
            predictions["Linear Regression"],
            label="Linear Regression",
            alpha=0.7,
        )

        ax.scatter(
            predictions["Actual Grade"],
            predictions["Random Forest"],
            label="Random Forest",
            alpha=0.7,
        )

        ax.plot([0, 20], [0, 20], linestyle="--")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel("Actual Grade")
        ax.set_ylabel("Predicted Grade")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Mean Absolute Error")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(
            metrics_df["Model"],
            metrics_df["MAE"],
        )

        ax.set_ylabel("MAE (lower is better)")
        ax.tick_params(axis="x", rotation=15)

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Random Forest Feature Importance")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax,
    )

    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Input Feature")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Feature importance describes how the trained Random Forest "
        "uses features. It does not establish causation."
    )


# --------------------------------------------------
# 7. DATASET ANALYSIS
# --------------------------------------------------

elif page == "Dataset Analysis":

    st.title("📁 Dataset Analysis")

    st.write(
        "Inspect the original student dataset used for model training."
    )

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Dataset Preview")

    number_of_rows = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=min(50, len(df)),
        value=min(10, len(df)),
    )

    st.dataframe(
        df.head(number_of_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Column Data Types")

    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
    })

    st.dataframe(
        dtypes_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
    })

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Numerical Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True,
    )

    st.subheader("Final Grade Distribution")

    grade_counts = (
        df["G3"]
        .value_counts()
        .sort_index()
        .rename_axis("Final Grade")
        .reset_index(name="Number of Students")
    )

    st.dataframe(
        grade_counts,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_dataset.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Model Evaluation Results",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="model_evaluation_results.csv",
        mime="text/csv",
    )

    st.success(
        "Dataset analysis and CSV download options are ready."
    )
