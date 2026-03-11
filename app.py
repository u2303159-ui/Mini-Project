import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# =========================
# CUSTOM CSS
# =========================
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset_clean1.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

FEATURES = ["AQI","PM2.5","PM10","NO2","SO2","CO","O3","NH3"]# FIXED WINDOW SIZE
WINDOW = 7

df = df[["city","Date"] + FEATURES]
df = df.fillna(df.mean(numeric_only=True))

# =========================
# LOAD MODELS
# =========================
scaler = joblib.load("aqi_scaler.pkl")
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

lin_model = joblib.load("models/linear_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")

# =========================
# LOAD LSTM + GRU
# =========================
lstm_model = load_model("models/lstm_model.keras", compile=False)
gru_model = load_model("models/gru_model.keras", compile=False)

# =========================
# AQI CATEGORY
# =========================
def categorize_aqi(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_color(category):

    if category == "Good":
        return "#16a34a"
    elif category == "Moderate":
        return "#facc15"
    elif category == "Unhealthy (Sensitive)":
        return "#f97316"
    else:
        return "#dc2626"

# =========================
# HEALTH RISKS
# =========================
pollutant_disease_risks = {

"PM2.5":[
"Asthma attacks",
"Bronchitis",
"Reduced lung function",
"Heart disease risk",
"Lung cancer"
],

"PM10":[
"Respiratory irritation",
"Asthma aggravation",
"Bronchitis",
"Reduced lung capacity"
],

"NO2":[
"Inflammation of airways",
"Asthma worsening",
"Respiratory infections"
],

"SO2":[
"Breathing difficulty",
"Asthma attacks",
"Bronchial irritation"
],

"CO":[
"Reduced oxygen supply",
"Headaches",
"Dizziness",
"Heart stress"
],

"O3":[
"Chest pain",
"Coughing",
"Lung inflammation",
"Reduced lung function"
]
,
"NH3":[
"Eye irritation",
"Skin irritation",
"Breathing difficulty",
"Respiratory inflammation"
]
}

pollutant_thresholds = {
"PM2.5":60,
"PM10":100,
"NO2":80,
"SO2":80,
"CO":2,
"O3":100,
"NH3":400
}

# =========================
# HEADER
# =========================
st.markdown(
'<div class="main-header">ML Driven Air Quality Forecasting & Health Risk Analysis</div>',
unsafe_allow_html=True
)

# =========================
# INPUT
# =========================
col1,col2 = st.columns([3,1])

with col1:
    cities = sorted(df["city"].unique())
    city = st.selectbox("Select City", cities)

with col2:
    predict = st.button("Predict")

# =========================
# PREDICTION
# =========================
if predict:

    city_df = df[df["city"]==city].sort_values("Date")

    if len(city_df) < WINDOW:

        st.error("Not enough data for prediction.")

    else:

        input_data = city_df.tail(WINDOW)[FEATURES].values
        input_scaled = scaler.transform(input_data)

        # =========================
        # LSTM
        # =========================
        predictions_lstm = []
        current_window = input_scaled.copy()

        for _ in range(5):

            reshaped = current_window.reshape(1, WINDOW, len(FEATURES))
            pred_scaled = lstm_model.predict(reshaped, verbose=0)[0][0]

            aqi_mean = scaler.mean_[0]
            aqi_std = scaler.scale_[0]

            pred_aqi = (pred_scaled * aqi_std) + aqi_mean
            predictions_lstm.append(pred_aqi)

            new_row = current_window[-1].copy()
            new_row[0] = pred_scaled

            current_window = np.vstack([current_window[1:], new_row])

        lstm_aqi=float(predictions_lstm[-1])
        lstm_category=categorize_aqi(lstm_aqi)

        # =========================
        # GRU
        # =========================
        predictions_gru=[]
        current_window=input_scaled.copy()

        for _ in range(5):

            reshaped=current_window.reshape(1,WINDOW,len(FEATURES))
            pred_scaled=gru_model.predict(reshaped,verbose=0)[0][0]

            aqi_mean=scaler.mean_[0]
            aqi_std=scaler.scale_[0]

            pred_aqi=(pred_scaled*aqi_std)+aqi_mean
            predictions_gru.append(pred_aqi)

            new_row=current_window[-1].copy()
            new_row[0]=pred_scaled

            current_window=np.vstack([current_window[1:],new_row])

        gru_aqi=float(predictions_gru[-1])
        gru_category=categorize_aqi(gru_aqi)

        card_color=get_color(lstm_category)

        # =========================
        # TREE MODELS
        # =========================
        latest = city_df.tail(1).copy()

        latest["Month"] = latest["Date"].dt.month
        latest["DayOfWeek"] = latest["Date"].dt.dayofweek

        latest_input = latest[
        ["city","PM2.5","PM10","NO2","NH3","CO","SO2","O3","Month","DayOfWeek"]
        ]

        # One-hot encode city
        latest_input = pd.get_dummies(latest_input, columns=["city"])

        # Add missing columns
        for col in feature_columns:
            if col not in latest_input.columns:
                latest_input[col] = 0

        # Ensure correct order
        latest_input = latest_input[feature_columns]

        rf_aqi=float(rf_model.predict(latest_input)[0])
        rf_category=categorize_aqi(rf_aqi)

        xgb_aqi=float(xgb_model.predict(latest_input)[0])
        xgb_category=categorize_aqi(xgb_aqi)

        lin_aqi = float(lin_model.predict(latest_input)[0])
        lin_category = categorize_aqi(lin_aqi)

        svm_aqi = float(svm_model.predict(latest_input)[0])
        svm_category = categorize_aqi(svm_aqi)

        # =========================
        # KPI CARDS
        # =========================
        st.markdown("---")

        c1,c2,c3,c4=st.columns(4)

        c1.markdown(
        f'<div class="card" style="background:{card_color};">Predicted AQI<br><h2>{lstm_aqi:.2f}</h2>{lstm_category}</div>',
        unsafe_allow_html=True)

        trend="Rising" if predictions_lstm[-1]>predictions_lstm[0] else "Falling"

        c2.markdown(
        f'<div class="card" style="background:{card_color};">AQI Trend<br><h2>{trend}</h2></div>',
        unsafe_allow_html=True)

        latest_pollutants=city_df.tail(1)[FEATURES].iloc[0]

        dominant_pollutant=latest_pollutants[1:].idxmax()

        c3.markdown(
        f'<div class="card" style="background:{card_color};">Dominant Pollutant<br><h2>{dominant_pollutant}</h2></div>',
        unsafe_allow_html=True)

        c4.markdown(
        f'<div class="card" style="background:{card_color};">Health Risk<br><h2>{lstm_category}</h2></div>',
        unsafe_allow_html=True)

        # =========================
        # TREND GRAPH
        # =========================
        st.markdown(
        '<div class="section-title">AQI Prediction Trend (All Models)</div>',
        unsafe_allow_html=True
        )

        days = list(range(1,6))   # Next 5 days prediction

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=days,
            y=predictions_lstm,
            mode='lines+markers',
            name="LSTM"
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=predictions_gru,
            mode='lines+markers',
            name="GRU"
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=[rf_aqi]*5,
            mode='lines+markers',
            name="Random Forest"
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=[xgb_aqi]*5,
            mode='lines+markers',
            name="XGBoost"
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=[lin_aqi]*5,
            mode='lines+markers',
            name="Linear Regression"
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=[svm_aqi]*5,
            mode='lines+markers',
            name="SVM"
        ))

        # Axis labels added here
        fig.update_layout(
            xaxis_title="Prediction Days (Next 5 Days)",
            yaxis_title="AQI Value",
            title="AQI Prediction Trend by Different Models",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # MODEL COMPARISON
        # =========================
        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

        comparison_df=pd.DataFrame({

        "Model":["LSTM","GRU","Random Forest","XGBoost","Linear Regression","SVM"],

        "Predicted AQI":[
        round(lstm_aqi,2),
        round(gru_aqi,2),
        round(rf_aqi,2),
        round(xgb_aqi,2),
        round(lin_aqi,2),
        round(svm_aqi,2)
        ],

        "Category":[
        lstm_category,
        gru_category,
        rf_category,
        xgb_category,
        lin_category,
        svm_category
        ]
        })

        st.dataframe(comparison_df,use_container_width=True)

        # =========================
        # HEALTH ANALYSIS
        # =========================
        st.markdown('<div class="section-title">Health Risk Analysis</div>', unsafe_allow_html=True)

        identified_risks=[]
        dangerous_pollutants=[]

        for pollutant,value in latest_pollutants.items():

            if pollutant in pollutant_thresholds:

                if value>pollutant_thresholds[pollutant]:

                    dangerous_pollutants.append((pollutant,value))
                    identified_risks.extend(pollutant_disease_risks[pollutant])

        colA,colB=st.columns(2)

        with colA:

            st.subheader("Pollutants Above Safe Limits")

            if dangerous_pollutants:

                pollutants_df=pd.DataFrame(
                dangerous_pollutants,
                columns=["Pollutant","Level"]
                )

                st.dataframe(pollutants_df)

            else:

                st.success("All pollutant levels are within safe limits.")

        with colB:

            st.subheader("Possible Health Effects")

            if identified_risks:

                for risk in set(identified_risks):
                    st.write("•",risk)

            else:

                st.success("No major health risks detected based on pollutant levels.")
        
        # =========================
        # HIGH POLLUTANT BAR CHART
        # =========================
        st.markdown('<div class="section-title">High Pollutant Levels</div>', unsafe_allow_html=True)

        if dangerous_pollutants:

            pollutant_names = [p[0] for p in dangerous_pollutants]
            pollutant_values = [p[1] for p in dangerous_pollutants]

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=pollutant_names,
                y=pollutant_values,
                text=[round(v,2) for v in pollutant_values],
                textposition='auto'
            ))

            fig_bar.update_layout(
                title="Pollutants Exceeding Safe Limits",
                xaxis_title="Pollutant",
                yaxis_title="Concentration Level",
                height=400
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.success("No pollutants exceed safe limits.")