# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_resource
def load_models():
    base_path = os.path.join(os.path.dirname(__file__), "models")
    rf = joblib.load(os.path.join(base_path, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(base_path, "xgb_model.pkl"))
    arima = joblib.load(os.path.join(base_path, "arima_model.pkl"))
    features = joblib.load(os.path.join(base_path, "trained_features.pkl"))
    return rf, xgb, arima, features

rf_model, xgb_model, arima_model, trained_features = load_models()

def preprocess_data(data, trained_features):
    cat_cols = ['platform', 'product_name', 'category', 'sub_category',
                'region', 'day_of_week', 'promotion_type', 'customer_name', 'customer_region']
    for col in [c for c in cat_cols if c not in data.columns]:
        data[col] = "Unknown"

    label_encoder = LabelEncoder()
    for col in cat_cols:
        data[col] = label_encoder.fit_transform(data[col].astype(str))

    if 'revenue' in data.columns:
        data['revenue_lag1'] = data['revenue'].shift(1).fillna(0)
        data['rolling_mean_7'] = data['revenue'].rolling(window=7, min_periods=1).mean()
    else:
        data['revenue_lag1'] = 0
        data['rolling_mean_7'] = 0

    one_hot_cols = ['platform', 'category', 'sub_category', 'region', 'day_of_week', 'promotion_type', 'customer_region']
    data = pd.get_dummies(data, columns=one_hot_cols)

    drop_cols = [col for col in data.columns if col not in trained_features and col != 'revenue']
    data = data.drop(columns=drop_cols)

    for col in trained_features:
        if col not in data.columns:
            data[col] = 0

    return data[trained_features], data

def evaluate_model(y_true, y_pred):
    y_true = y_true.fillna(0)
    y_pred = np.nan_to_num(y_pred)
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred)
    )

def interpret_metrics(rmse, mae, r2):
    interpretation = ""
    if rmse < 1000:
        interpretation += "- RMSE is low, suggesting the model's predictions are close to actual sales.\n"
    elif rmse < 5000:
        interpretation += "- RMSE is moderate; model is somewhat accurate.\n"
    else:
        interpretation += "- RMSE is high, indicating the model struggles to predict accurately.\n"

    if mae < 500:
        interpretation += "- MAE is very low, meaning average prediction error is small.\n"
    elif mae < 3000:
        interpretation += "- MAE is acceptable but could be improved.\n"
    else:
        interpretation += "- MAE is high, implying notable prediction errors.\n"

    if r2 > 0.9:
        interpretation += "- R² is excellent. The model explains most of the variance in sales.\n"
    elif r2 > 0.6:
        interpretation += "- R² is decent. Model has moderate explanatory power.\n"
    elif r2 > 0.3:
        interpretation += "- R² is weak. Model only partially explains the variance.\n"
    else:
        interpretation += "- R² is very low. Predictions are not well correlated with actuals.\n"

    return interpretation

def plot_predictions(y_true, y_pred, title="Prediction vs Actual"):
    fig = px.line()
    fig.add_scatter(x=np.arange(len(y_true)), y=y_true, mode='lines', name='Actual')
    fig.add_scatter(x=np.arange(len(y_pred)), y=y_pred, mode='lines', name='Predicted')
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

def explain_metrics():
    st.markdown("""### 📊 What the Metrics Mean:
- **RMSE (Root Mean Square Error)**: Average squared difference from actual values. Lower is better.
- **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values.
- **R² (R-squared)**: Proportion of variance in revenue explained by the model. Closer to 1 is better.
""")

# Session state
if "platform1" not in st.session_state:
    st.session_state["platform1"] = None
if "platform2" not in st.session_state:
    st.session_state["platform2"] = None
if "prediction_data" not in st.session_state:
    st.session_state["prediction_data"] = None
if "prediction_columns" not in st.session_state:
    st.session_state["prediction_columns"] = None

st.title("Sales Forecasting Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Retrain", "EDA", "About"])

with tab1:
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if any(col not in data.columns for col in ['product_name', 'revenue']):
            st.error("Missing required columns.")
            st.stop()

        st.write("Uploaded Data Sample", data.head())
        features_data, full_data = preprocess_data(data.copy(), trained_features)

        model_options = st.sidebar.multiselect("Choose models", ['Random Forest', 'XGBoost', 'ARIMA'], default=['Random Forest'])

        if st.button("Predict Now"):
            with st.spinner("Running predictions..."):
                prediction_columns = {}

                if 'Random Forest' in model_options:
                    full_data['RF_Prediction'] = rf_model.predict(features_data)
                    prediction_columns['Random Forest'] = 'RF_Prediction'

                if 'XGBoost' in model_options:
                    full_data['XGB_Prediction'] = xgb_model.predict(features_data)
                    prediction_columns['XGBoost'] = 'XGB_Prediction'

                if 'ARIMA' in model_options and 'revenue' in full_data.columns:
                    try:
                        full_data['ARIMA_Forecast'] = arima_model.forecast(steps=len(full_data))
                        prediction_columns['ARIMA'] = 'ARIMA_Forecast'
                    except Exception as e:
                        st.error(f"ARIMA Forecast Error: {e}")
                elif 'ARIMA' in model_options:
                    st.warning("ARIMA needs a 'revenue' column.")

                st.session_state["prediction_data"] = full_data
                st.session_state["prediction_columns"] = prediction_columns
                st.success("Prediction completed!")

        if st.session_state["prediction_data"] is not None:
            full_data = st.session_state["prediction_data"]
            prediction_columns = st.session_state["prediction_columns"]

            if 'platform' in data.columns:
                platforms = sorted(data['platform'].unique())
                if st.session_state["platform1"] not in platforms:
                    st.session_state["platform1"] = platforms[0]
                if st.session_state["platform2"] not in platforms:
                    st.session_state["platform2"] = platforms[1] if len(platforms) > 1 else platforms[0]

                col1, col2 = st.columns(2)
                st.session_state["platform1"] = col1.selectbox("Select First Platform", platforms, index=platforms.index(st.session_state["platform1"]))
                st.session_state["platform2"] = col2.selectbox("Select Second Platform", platforms, index=platforms.index(st.session_state["platform2"]))

                for model in model_options:
                    pred_col = prediction_columns.get(model)
                    if not pred_col:
                        continue

                    st.subheader(f"{model} Comparison: {st.session_state['platform1']} vs {st.session_state['platform2']}")
                    for plat in [st.session_state['platform1'], st.session_state['platform2']]:
                        subset = full_data[data['platform'] == plat]
                        if subset.empty:
                            st.warning(f"No data for platform {plat}")
                            continue
                        y_true = subset['revenue']
                        y_pred = subset[pred_col]
                        rmse, mae, r2 = evaluate_model(y_true, y_pred)

                        st.markdown(f"**Metrics for {plat}**")
                        st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
                        st.markdown("**Interpretation:**")
                        st.markdown(interpret_metrics(rmse, mae, r2))
                        plot_predictions(y_true, y_pred, title=f"{model} Prediction - {plat}")

                    explain_metrics()

            st.download_button("Download Predictions", full_data.to_csv(index=False), file_name="predictions.csv")

with tab2:
    if uploaded_file:
        st.header("Retrain Random Forest")
        if st.button("Train Again"):
            if 'revenue' in data.columns:
                with st.spinner("Retraining model..."):
                    X, _ = preprocess_data(data.copy(), trained_features)
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, data['revenue'])
                    joblib.dump(model, r'C:\Users\samee\flora_edge\rf_model_retrained.pkl')
                    st.success("Model retrained and saved.")
            else:
                st.error("'Revenue' column is required to retrain.")

with tab3:
    st.header("Exploratory Data Analysis (EDA)")

    if uploaded_file:
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])

        st.subheader("Revenue Over Time")
        if 'date' in data.columns:
            time_group = data.groupby('date')['revenue'].sum().reset_index()
            fig = px.bar(time_group, x='date', y='revenue', title="Revenue Over Time", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue by Category")
        if 'category' in data.columns:
            cat_group = data.groupby('category')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(cat_group, x='revenue', y='category', orientation='h', color='revenue', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue by Region")
        if 'region' in data.columns:
            reg_group = data.groupby('region')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(reg_group, x='revenue', y='region', orientation='h', color='revenue', color_continuous_scale='Tealgrn')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Revenue by Platform")
        if 'platform' in data.columns:
            platform_rev = data.groupby('platform')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(platform_rev, x='platform', y='revenue', color='platform', title="Revenue by Platform", color_discrete_sequence=px.colors.qualitative.Dark24)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 5 Products by Revenue")
        if {'product_name', 'revenue'}.issubset(data.columns):
            top_products = data.groupby('product_name')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False).head(5)
            fig = px.bar(top_products, x='revenue', y='product_name', orientation='h', color='revenue', color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weekly Sale Pattern")
        if 'day_of_week' in data.columns:
            weekly = data.groupby('day_of_week')['revenue'].sum().reset_index()
            fig = px.line(weekly, x='day_of_week', y='revenue', markers=True, title="Weekly Revenue Pattern", color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Customer Sentiment Distribution")
        if 'sentiment_score' in data.columns:
            data['sentiment_label'] = data['sentiment_score'].apply(
                lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral'
            )

            sentiment_count = data['sentiment_label'].value_counts().reset_index()
            sentiment_count.columns = ['Sentiment', 'Count']
            fig = px.bar(sentiment_count, x='Sentiment', y='Count', color='Sentiment',
                         title="Overall Customer Sentiment Count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

            if 'platform' in data.columns:
                sentiment_platform = data.groupby(['platform', 'sentiment_label']).size().reset_index(name='count')
                fig = px.bar(sentiment_platform, x='platform', y='count', color='sentiment_label',
                             barmode='group', title="Customer Sentiment by Platform",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)

            if 'customer_region' in data.columns:
                sentiment_region = data.groupby(['customer_region', 'sentiment_label']).size().reset_index(name='count')
                fig = px.bar(sentiment_region, x='customer_region', y='count', color='sentiment_label',
                             barmode='group', title="Customer Sentiment by Customer Region",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'sentiment_score' not found in the data.")


with tab4:
    st.header("About")
    st.markdown("""
    This dashboard forecasts sales using:
    - Random Forest
    - XGBoost
    - ARIMA (Time-series)

    Developed by Sameer Ahmad using Streamlit, Scikit-learn, and Plotly.
    """)
