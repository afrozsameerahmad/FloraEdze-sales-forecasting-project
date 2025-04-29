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

# Load Models and Features
@st.cache_resource
def load_models():
    base_path = os.path.join(os.path.dirname(__file__), "models")
    rf = joblib.load(os.path.join(base_path, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(base_path, "xgb_model.pkl"))
    arima = joblib.load(os.path.join(base_path, "arima_model.pkl"))
    features = joblib.load(os.path.join(base_path, "trained_features.pkl"))
    return rf, xgb, arima, features

rf_model, xgb_model, arima_model, trained_features = load_models()

# Preprocessing
def preprocess_data(data, trained_features):
    cat_cols = ['platform', 'product_name', 'category', 'sub_category',
                'region', 'day_of_week', 'promotion_type', 'customer_name', 'customer_region']

    missing_optional = [col for col in cat_cols if col not in data.columns]
    for col in missing_optional:
        data[col] = "Unknown"

    if missing_optional:
        st.warning(f"Missing optional columns filled with 'Unknown': {', '.join(missing_optional)}")

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

    drop_cols = [col for col in data.columns if col not in trained_features and col not in ['revenue']]
    data = data.drop(columns=drop_cols)

    for col in trained_features:
        if col not in data.columns:
            data[col] = 0

    return data[trained_features], data

# Evaluation
def evaluate_model(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred)
    )

# Plotting
def plot_predictions(y_true, y_pred, title="Prediction vs Actual"):
    fig = px.line()
    fig.add_scatter(x=np.arange(len(y_true)), y=y_true, mode='lines', name='Actual')
    fig.add_scatter(x=np.arange(len(y_pred)), y=y_pred, mode='lines', name='Predicted')
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title(" Sales Forecasting Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
tab1, tab2, tab3, tab4 = st.tabs([" Predictions", "Retrain", "EDA", "About"])

with tab1:
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        required_columns = ['product_name', 'revenue']
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            st.error(f" Missing required columns: {', '.join(missing_required)}")
            st.stop()

        st.write("Uploaded Data", data.head())

        features_data, full_data = preprocess_data(data.copy(), trained_features)

        model_options = st.sidebar.multiselect("Choose models", ['Random Forest', 'XGBoost', 'ARIMA'], default=['Random Forest'])

        if st.button("Predict Now"):
            with st.spinner("Running predictions..."):
                results = {}

                if 'Random Forest' in model_options:
                    preds = rf_model.predict(features_data)
                    full_data['RF_Prediction'] = preds
                    results['Random Forest'] = preds

                if 'XGBoost' in model_options:
                    preds = xgb_model.predict(features_data)
                    full_data['XGB_Prediction'] = preds
                    results['XGBoost'] = preds

                if 'ARIMA' in model_options and 'revenue' in full_data.columns:
                    try:
                        preds = arima_model.forecast(steps=len(full_data))
                        full_data['ARIMA_Forecast'] = preds
                        results['ARIMA'] = preds
                    except Exception as e:
                        st.error(f"ARIMA Forecast Error: {e}")
                elif 'ARIMA' in model_options:
                    st.warning("ARIMA needs a 'revenue' column.")

            st.success("Prediction completed!")

            for model, pred in results.items():
                st.subheader(f"{model} Results")
                if 'revenue' in full_data.columns:
                    rmse, mae, r2 = evaluate_model(full_data['revenue'], pred)
                    st.write(f" RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
                plot_predictions(full_data['revenue'], pred, title=f"{model} - Prediction vs Actual")

            st.download_button("Download Predictions", full_data.to_csv(index=False), file_name="predictions.csv")

with tab2:
    if uploaded_file:
        st.header(" Retrain Random Forest")
        if st.button("Retrain Now"):
            if 'revenue' in data.columns:
                with st.spinner("Retraining model..."):
                    X, _ = preprocess_data(data.copy(), trained_features)
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, data['revenue'])
                    joblib.dump(model, r'C:\Users\samee\flora_edge\rf_model_retrained.pkl')
                    st.success(" Model retrained and saved.")
            else:
                st.error(" 'Revenue' column required to retrain.")

with tab3:
    st.header(" Exploratory Data Analysis (EDA)")

    if uploaded_file:
        # Convert date column
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])

        # Revenue Over Time
        st.subheader("Revenue Over Time")
        if 'date' in data.columns:
            time_group = data.groupby('date')['revenue'].sum().reset_index()
            fig = px.bar(time_group, x='date', y='revenue', title="Revenue Over Time",
                         color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)

        # Revenue by Category
        st.subheader("Revenue by Category")
        if 'category' in data.columns:
            cat_group = data.groupby('category')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(cat_group, x='revenue', y='category', orientation='h',
                         color='revenue', color_continuous_scale='Blues', title="Revenue by Category")
            st.plotly_chart(fig, use_container_width=True)

        # Revenue by Region
        st.subheader("Revenue by Region")
        if 'region' in data.columns:
            reg_group = data.groupby('region')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(reg_group, x='revenue', y='region', orientation='h',
                         color='revenue', color_continuous_scale='Tealgrn', title="Revenue by Region")
            st.plotly_chart(fig, use_container_width=True)

        # Revenue by Platform
        st.subheader("Revenue by Platform")
        if 'platform' in data.columns:
            platform_rev = data.groupby('platform')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(platform_rev, x='platform', y='revenue', color='platform',
                         title="Revenue by Platform", color_discrete_sequence=px.colors.qualitative.Dark24)
            st.plotly_chart(fig, use_container_width=True)

        # Total Units Sold by Platform
        st.subheader("Total Units Sold by Platform")
        if {'platform', 'units_sold'}.issubset(data.columns):
            platform_units = data.groupby('platform')['units_sold'].sum().reset_index().sort_values(by='units_sold', ascending=False)
            fig = px.bar(platform_units, x='platform', y='units_sold', color='platform',
                         title="Units Sold by Platform", color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

        # Top 5 Products by Revenue
        st.subheader("Top 5 Products by Revenue")
        if {'product_name', 'revenue'}.issubset(data.columns):
            top_products = data.groupby('product_name')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False).head(5)
            fig = px.bar(top_products, x='revenue', y='product_name', orientation='h',
                         color='revenue', color_continuous_scale='Oranges', title="Top 5 Products by Revenue")
            st.plotly_chart(fig, use_container_width=True)

        # Total Products by Units Sold
        st.subheader("Total Products by Units Sold")
        if {'product_name', 'units_sold'}.issubset(data.columns):
            top_units = data.groupby('product_name')['units_sold'].sum().reset_index().sort_values(by='units_sold', ascending=False).head(10)
            fig = px.bar(top_units, x='units_sold', y='product_name', orientation='h',
                         color='units_sold', color_continuous_scale='Viridis', title="Top Products by Units Sold")
            st.plotly_chart(fig, use_container_width=True)

        # Weekly Sale Pattern
        st.subheader("Weekly Sale Pattern")
        if 'day_of_week' in data.columns:
            weekly = data.groupby('day_of_week')['revenue'].sum().reset_index()
            fig = px.line(weekly, x='day_of_week', y='revenue', markers=True,
                          title="Weekly Revenue Pattern", color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig, use_container_width=True)

        # Units Sold by Category
        st.subheader("Units Sold by Category")
        if {'category', 'units_sold'}.issubset(data.columns):
            cat_units = data.groupby('category')['units_sold'].sum().reset_index().sort_values(by='units_sold', ascending=False)
            fig = px.bar(cat_units, x='units_sold', y='category', orientation='h',
                         color='units_sold', color_continuous_scale='Cividis', title="Units Sold by Category")
            st.plotly_chart(fig, use_container_width=True)

        # Revenue by Promotion Type
        st.subheader("Revenue by Promotion Type")
        if {'promotion_type', 'revenue'}.issubset(data.columns):
            promo_rev = data.groupby('promotion_type')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
            fig = px.bar(promo_rev, x='promotion_type', y='revenue', color='promotion_type',
                         title="Revenue by Promotion Type", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        # Profit Margin Distribution by Category
        st.subheader("Profit Margin by Category")
        if {'profit_margin', 'category'}.issubset(data.columns):
            fig = px.box(data, x='category', y='profit_margin', points="all",
                         color='category', title="Profit Margin Distribution per Category",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        # Revenue by Customer Rating
        st.subheader("Revenue by Customer Rating")
        if {'customer_rating', 'revenue'}.issubset(data.columns):
            data['rating_bucket'] = pd.cut(data['customer_rating'], bins=[0, 2, 3, 4, 5],
                                           labels=['0–2', '2–3', '3–4', '4–5'])
            rating_group = data.groupby('rating_bucket')['revenue'].mean().reset_index()
            fig = px.bar(rating_group, x='rating_bucket', y='revenue', color='rating_bucket',
                         title="Average Revenue by Customer Rating", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
     

with tab4:
    st.header("About")
    st.markdown("""
    This app helps forecast sales using:
    - ✅ Random Forest
    - ✅ XGBoost
    - ✅ ARIMA (Time series)

    Built by Sameer Ahmad using Streamlit, Scikit-learn, and Plotly.
    """)
