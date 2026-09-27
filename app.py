
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="💧 Water Anomaly Detector", layout="wide")

@st.cache_resource
def load_model():
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    with open('feature_columns.txt', 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv('features_engineered.csv')

model, scaler, feature_cols = load_model()
data = load_data()

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Select Page:", ["📊 Dashboard", "🔮 Predict", "📈 Analysis", "ℹ️ About"])

if page == "📊 Dashboard":
    st.title("💧 Water Consumption Anomaly Classification")
    st.write("Detect unusual water usage patterns using Machine Learning")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", f"{len(data):,}")
    with col2:
        anomaly_count = (data['is_anomaly'] == 1).sum()
        st.metric("Anomalies Detected", f"{anomaly_count:,}", f"{anomaly_count/len(data)*100:.2f}%")
    with col3:
        st.metric("Buildings", data['building'].nunique())

    st.subheader("🤖 Model Performance")
    perf_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Score': ['99.99%', '100%', '99.89%', '0.9995', '0.9995']
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

    st.subheader("📍 Anomalies by Building")
    building_anomalies = data.groupby('building')['is_anomaly'].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    building_anomalies.plot(kind='bar', ax=ax, color='#ff6b6b')
    ax.set_title('Number of Anomalies by Building', fontweight='bold')
    ax.set_xlabel('Building')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "🔮 Predict":
    st.title("🔮 Predict Water Consumption Anomaly")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            water_usage = st.number_input("Water Usage (Liters)", 0.0, 1000.0, 50.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            day_of_week = st.selectbox("Day of Week", list(day_mapping.keys()))
            month = st.slider("Month", 1, 12, 6)

        with col2:
            occupancy = st.slider("Number of People", 0, 5, 3)
            is_morning = st.checkbox("Morning Peak (5-9am)?")
            is_evening = st.checkbox("Evening Peak (4-8pm)?")
            is_weekend = st.checkbox("Weekend?")

        submit = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submit:
        baseline_usage = data.groupby('hour')['water_use_l'].mean()[hour]
        rolling_mean = data['rolling_mean_7d'].mean()
        rolling_std = data['rolling_std_7d'].mean()

        input_data = {
            'hour': hour,
            'day_of_week': day_mapping[day_of_week],
            'month': month,
            'is_weekend': 1 if is_weekend else 0,
            'is_morning': 1 if is_morning else 0,
            'is_evening': 1 if is_evening else 0,
            'is_night': 1 if hour in [0, 1, 2, 3, 4] else 0,
            'occupancy_count': occupancy,
            'occupancy_ratio': occupancy / 5.0,
            'all_occupied': 1 if occupancy == 5 else 0,
            'none_occupied': 1 if occupancy == 0 else 0,
            'rolling_mean_7d': rolling_mean,
            'rolling_std_7d': rolling_std,
            'deviation_from_mean': water_usage - rolling_mean,
            'expected_usage': 2 * occupancy,
            'usage_vs_occupancy': water_usage - (2 * occupancy),
            'usage_percentile': np.percentile(data['water_use_l'], 50),
            'usage_vs_hour_baseline': water_usage - baseline_usage
        }

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 ANOMALY DETECTED! (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.success(f"✅ NORMAL (Confidence: {probability[0]*100:.2f}%)")

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(['Normal', 'Anomaly'], probability, color=['#4CAF50', '#FF5252'])
        ax.set_xlim(0, 1)
        for i, v in enumerate(probability):
            ax.text(v + 0.02, i, f'{v*100:.2f}%', va='center')
        st.pyplot(fig)

elif page == "📈 Analysis":
    st.title("📈 Data Analysis & Insights")

    tab1, tab2, tab3 = st.tabs(["Hourly Patterns", "Occupancy", "Buildings"])

    with tab1:
        st.subheader("💧 Water Usage by Hour")
        hourly_stats = data.groupby('hour')['water_use_l'].agg(['mean', 'std'])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hourly_stats.index, hourly_stats['mean'], marker='o', linewidth=2, label='Mean', color='#1f77b4')
        ax.fill_between(hourly_stats.index, hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'], alpha=0.3, label='±1 Std')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Water Usage (L)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with tab2:
        st.subheader("👥 Impact of Occupancy")
        occ_stats = data.groupby('occupancy_count')['water_use_l'].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(occ_stats.index, occ_stats.values, color='#2ecc71', edgecolor='black')
        ax.set_xlabel('Occupancy Count')
        ax.set_ylabel('Average Usage (L)')
        st.pyplot(fig)

    with tab3:
        st.subheader("🏢 Building Comparison")
        building_stats = data.groupby('building')['water_use_l'].mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        building_stats.plot(kind='bar', ax=ax, color='#9b59b6', edgecolor='black')
        ax.set_title('Average Usage by Building')
        ax.set_ylabel('Usage (L)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## 📊 Water Consumption Anomaly Classification

    A machine learning system to detect unusual water usage patterns in smart buildings.

    ### Key Features:
    - **Dataset:** 87,360 hourly measurements from 10 buildings
    - **Features:** 18+ engineered features
    - **Best Model:** Decision Tree (99.99% accuracy)
    - **Metrics:** 100% precision, 99.89% recall

    ### How It Works:
    1. Analyzes water usage patterns
    2. Detects anomalies using three rules
    3. Provides real-time predictions

    ### Technology:
    - Python, Scikit-learn, Streamlit
    - Interactive visualizations
    """)

st.markdown("---")
st.markdown("<center><small>Water Consumption Anomaly Classification | ML Project</small></center>", unsafe_allow_html=True)
