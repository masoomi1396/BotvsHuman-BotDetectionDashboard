# 🤖 Bot vs Human — Bot Detection Dashboard

A web application built with **Streamlit** that uses a machine learning model to classify website sessions as either human or bot activity. The application provides an interactive dashboard for making predictions, evaluating model performance, exploring session data, and querying results using SQL.

## 🌐 Live Application

**Try the application here:** [https://botvshuman-botdetectiondashboard.streamlit.app/](YOUR_STREAMLIT_APP_URL)

*The link will be updated once the application is deployed.*

## 📌 About the Project

This project was developed as part of a **university group assignment at the University of Twente**.

The main goal is to explore how machine learning can help distinguish human users from bots on an online ticket shop website. The application works with recorded website sessions and uses session features, such as clicking behaviour, mouse movements, scrolling, and typing patterns, to make predictions.

The project combines machine learning, data analysis, visualisation, and SQL queries in one interactive application.

## ✨ Features

- **Bot and human classification:** Predict whether a website session is likely to belong to a human or a bot.
- **CSV upload:** Upload a CSV file containing session data to make predictions for multiple sessions.
- **Pre-trained machine learning model:** Use a saved K-Nearest Neighbors (KNN) model and scaler to generate predictions.
- **Model evaluation:** Display performance metrics by comparing predictions with labelled test data.
- **Data visualisation:** Explore session data and prediction results through graphs.
- **SQL queries:** Filter and explore prediction results using SQL queries on pandas DataFrames.
- **Interactive dashboard:** Use the Streamlit interface to explore the data and results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing and manipulation |
| Scikit-learn | Machine learning and evaluation metrics |
| Joblib | Loading the saved model and scaler |
| Matplotlib | Data visualisation |
| pandasql | SQL queries on pandas DataFrames |
| Jupyter Notebook | Model development and experimentation |

## 📂 Project Structure

```text
bot-detection/
├── main.py
├── Project_M1.ipynb
├── data-7.csv
├── knn_model.joblib
├── scaler.joblib
├── test_data.csv
├── bot_prediction_upload.csv
├── requirements.txt
├── README.md
└── docs/
    ├── Bot_vs_Human_Requirements.pdf
    └── Ui sketch.pdf
```

### Main Files

- `main.py` — The main Streamlit application.
- `Project_M1.ipynb` — The Jupyter Notebook used for model development and experimentation.
- `data-7.csv` — The dataset used during model development.
- `knn_model.joblib` — The saved KNN classification model.
- `scaler.joblib` — The saved scaler used to prepare input features.
- `test_data.csv` — Labelled test data used to evaluate model performance.
- `bot_prediction_upload.csv` — An example CSV file that can be uploaded to test the application's prediction functionality.
- `requirements.txt` — The Python packages required to run the application.
- `docs/Bot_vs_Human_Requirements.pdf` — The project requirements document.
- `docs/Ui sketch.pdf` — The initial user interface design sketch.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd bot-detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run main.py
```

Streamlit will provide a local URL where you can open the application in your browser.

## 📊 How to Use the Application

1. Start the application using Streamlit.
2. Open the application in your browser.
3. Upload a CSV file containing website session data, if using the batch prediction functionality.
4. Allow the application to process the input data and generate predictions.
5. Explore the prediction results, visualisations, and available evaluation metrics.
6. Use the SQL section to filter and investigate session data and prediction results.

An example CSV file, `bot_prediction_upload.csv`, is included in the repository to help users test the upload and prediction functionality.

## 🧠 Machine Learning Model

The application uses a **K-Nearest Neighbors (KNN)** model to classify website sessions.

The model was developed and evaluated in `Project_M1.ipynb`. The trained model and scaler are saved separately and loaded by the Streamlit application.

The model uses behavioural features such as:

- Number of clicks
- Median click interval
- Variation in click intervals
- Mouse path curvature
- Mouse movement distance
- Idle ratio
- Scroll entropy
- Typing latency
- Average dwell time
- Number of pages visited
- Referrer information
- Button hover time

These features represent different aspects of how a user interacts with a website.

The model produces one of two classifications:

- **HUMAN** — The session is classified as human activity.
- **BOT** — The session is classified as bot activity.

The classification is a prediction made by the model and should not be treated as a guarantee that a session has been correctly identified.

## 📈 Model Evaluation

The application uses labelled test data to evaluate the model's predictions.

Depending on the selected evaluation functionality, the metrics include:

- **Accuracy:** The proportion of predictions that are correct.
- **Precision:** The proportion of sessions predicted as a particular class that actually belong to that class.
- **Confusion matrix:** A summary of correct and incorrect classifications for each class.
- **Mean Absolute Error (MAE):** The average absolute difference between numerical values.
- **Mean Squared Error (MSE):** The average squared difference between numerical values.

MAE and MSE are regression-oriented metrics. When evaluating a binary classifier, accuracy, precision, recall, F1-score, and the confusion matrix are generally more directly interpretable.

Evaluation results depend on the test dataset and the labels it contains.

## 🗄️ SQL and Data Exploration

The application includes SQL-based queries for exploring session data and prediction results.

Users can investigate patterns by filtering sessions according to their predicted classification and examining relevant behavioural features.

Examples of useful questions include:

- Which sessions were classified as bots?
- Which sessions were classified as humans?
- How do clicking behaviour and idle ratio differ between the predicted groups?
- Which sessions have particular browser or interaction characteristics?

The SQL functionality is applied to pandas DataFrames using `pandasql`.

## 📋 Input Data Requirements

The uploaded CSV file must contain the columns required by the prediction pipeline.

The model uses the following features:

```text
clicks
median_click_interval_ms
click_interval_std_ms
mouse_path_curvature
mouse_distance_px
idle_ratio
scroll_entropy
digraph_latency_ms
avg_dwell_time
pages_visited
referrer_type_ad
referrer_type_search
button_hover_ms
```

If `avg_dwell_time` is not already available, the application may calculate it from the following fields, provided they are included in the input data:

```text
dwell_field1_ms
dwell_field2_ms
dwell_field3_ms
```

For batch predictions, the input file does not need a `bot` label because the model generates the prediction. Labelled data is required when calculating evaluation metrics against known outcomes.

Make sure that the column names and feature formats match the application's expected input.

## 📦 Requirements

The required Python packages are listed in `requirements.txt`:

```text
streamlit
pandas
joblib
scikit-learn
matplotlib
pandasql
```

Python's standard library modules, such as `math` and `random`, do not need to be installed separately.

## 🎓 Academic Context

This application was developed as a group assignment for the **Technical Computer Science bachelor's programme at the University of Twente**.

The project provides practical experience with:

- Applying machine learning to website session data
- Building an interactive data application
- Processing and analysing CSV files
- Evaluating classification results
- Creating data visualisations
- Using SQL to explore datasets
- Organising and documenting a software project

The requirements document and initial interface sketch are included in the `docs/` directory.

## 👥 Project Team

**Group 14**

Developed as part of a university group assignment at the University of Twente.

## 📄 License

This project was developed for educational purposes as part of a university assignment. No specific open-source license is provided unless one is added to the repository.

---

*Bot vs Human — Exploring the use of machine learning to distinguish human users from automated website activity.*
