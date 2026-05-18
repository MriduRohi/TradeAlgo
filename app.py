import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TradeAlgo Predictor", page_icon="📈", layout="wide")

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00FFB9;}
    .stSlider > div > div > div > div {background-color: #00FFB9;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Aesthetic TradeAlgo Predictor")
st.markdown("An interactive S&P 500 Market Predictor using Random Forest Machine Learning.")

# --- SIDEBAR INTERACTIVITY ---
st.sidebar.header("⚙️ Model Parameters")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 200, step=50)
min_samples_split = st.sidebar.slider("Min Samples Split", 10, 200, 50, step=10)
prob_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6, step=0.05)
time_period = st.sidebar.selectbox("Historical View", ["1y", "2y", "5y", "10y", "max"], index=2)

# --- DATA FETCHING & PROCESSING ---
@st.cache_data
def load_and_process_data():
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500 = sp500.loc["1990-01-01":].copy()
    
    # Target Setup
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    
    # Feature Engineering (Horizons)
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors += [ratio_column, trend_column]
        
    sp500 = sp500.dropna()
    return sp500, new_predictors

with st.spinner("Fetching Live Market Data & Engineering Features..."):
    data, predictors = load_and_process_data()

# --- MODEL TRAINING ---
@st.cache_resource
def train_model(n_est, min_split, threshold):
    model = RandomForestClassifier(n_estimators=n_est, min_samples_split=min_split, random_state=1)
    
    # Train on all but the last 100 days, test on the last 100 days
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    
    model.fit(train[predictors], train["Target"])
    
    # Get Probabilities
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    
    # Apply Threshold
    preds = (preds_proba >= threshold).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    
    return model, test, preds

with st.spinner("Training Random Forest Model..."):
    model, test_data, predictions = train_model(n_estimators, min_samples_split, prob_threshold)

# --- PREDICTION FOR TOMORROW ---
# Get the very last row of our data to predict "Tomorrow"
latest_data = data.iloc[-1:]
latest_proba = model.predict_proba(latest_data[predictors])[:, 1][0]
predicted_direction = "UP 🔼" if latest_proba >= prob_threshold else "DOWN 🔽"
prediction_color = "#00FFB9" if predicted_direction == "UP 🔼" else "#FF4B4B"

st.subheader(f"🔮 Prediction for Tomorrow's Close: :{'green' if predicted_direction == 'UP 🔼' else 'red'}[{predicted_direction}]")
st.write(f"**Model Confidence:** {latest_proba * 100:.2f}% (Threshold: {prob_threshold * 100}%)")

# --- AESTHETIC PLOTLY CHART WITH DASHED ARROWS ---
st.subheader("📊 Market Trend & Prediction Visualization")

# Filter data for visualization based on user selection
view_data = data.tail(252 * int(time_period.replace('y', '')) if time_period != 'max' else len(data))

fig = go.Figure()

# Plot Actual Close Price
fig.add_trace(go.Scatter(
    x=view_data.index, y=view_data['Close'], 
    mode='lines', name='Close Price',
    line=dict(color='#8A2BE2', width=2)
))

# Plot 60-Day Moving Average as a cool dashed line
fig.add_trace(go.Scatter(
    x=view_data.index, y=view_data['Close'].rolling(60).mean(), 
    mode='lines', name='60-Day MA',
    line=dict(color='#00FFB9', width=2, dash='dashdot')
))

# Add the cool "Dashed Arrow" pointing to the latest prediction
latest_date = view_data.index[-1]
latest_price = view_data['Close'].iloc[-1]

arrow_y = -60 if predicted_direction == "DOWN 🔽" else 60
fig.add_annotation(
    x=latest_date,
    y=latest_price,
    text=f"Predicted {predicted_direction}",
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=3,
    arrowcolor=prediction_color,
    ax=0,
    ay=arrow_y,
    arrowdash="dash", # Creates the dashed arrow aesthetic!
    font=dict(color=prediction_color, size=14, family="Courier New", weight="bold"),
    bgcolor="#0E1117",
    borderpad=4
)

# Dark theme formatting
fig.update_layout(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='#FFFFFF'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#333333'),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# --- RAW DATA EXPANDER ---
with st.expander("📂 View Raw Test Data & Predictions"):
    combined = pd.concat([test_data["Close"], test_data["Target"], predictions], axis=1)
    st.dataframe(combined.tail(20), use_container_width=True)
