import streamlit as st

# Set page configuration (must be the very first Streamlit command)
st.set_page_config(
    page_title="Crop Yield Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --------------------------------
# Custom CSS for a dashboard feel
# --------------------------------
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .css-1d391kg { 
        background-color: #f1f5f9;
    }
    .stButton>button {
        background-color: #ff9933;
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .highlight {
        color: #047857;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Crop Yield Analysis & Prediction")
st.write("Welcome! Explore historical crop yield trends or predict future yields using our model. The app is designed with an Indian-farmer theme.")

# --------------------------------
# Sidebar Navigation
# --------------------------------
page = st.sidebar.radio("Select Page", ("Visualization & Q&A", "Prediction Model"))

# --------------------------------
# Data Loading and Preparation
# --------------------------------
@st.cache
def load_data():
    df = pd.read_csv("cleaned_crop_yield.csv")
    # Enhance the dataset if needed:
    df['Fertilizer_per_area'] = df['Fertilizer'] / df['Area']
    df['Pesticide_per_area'] = df['Pesticide'] / df['Area']
    df['Production_per_area'] = df['Production'] / df['Area']
    return df

df = load_data()

# --------------------------------
# Natural Language Query Processing Function
# --------------------------------
def process_query(query_text):
    query_lower = query_text.lower()
    crop = selected_crop
    for crop_option in crop_options:
        if crop_option != 'All' and crop_option.lower() in query_lower:
            crop = crop_option
            break
    state = selected_state
    for state_option in state_options:
        if state_option != 'All' and state_option.lower() in query_lower:
            state = state_option
            break
    season = selected_season
    for season_option in season_options:
        if season_option != 'All' and season_option.lower() in query_lower:
            season = season_option
            break
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', query_text)
    if len(years) >= 2:
        year_range_query = (int(min(years)), int(max(years)))
    elif len(years) == 1:
        year_range_query = (int(years[0]), int(years[0]))
    else:
        year_range_query = year_range
    return crop, state, season, year_range_query

# --------------------------------
# Page 1: Visualization & Q&A Interface
# --------------------------------
if page == "Visualization & Q&A":
    st.header("Visualization & Q&A Interface")
    
    # Sidebar filters for visualization
    crop_options = ['All'] + sorted(df['Crop'].unique().tolist())
    state_options = ['All'] + sorted(df['State'].unique().tolist())
    season_options = ['All'] + sorted(df['Season'].unique().tolist())
    min_year = int(df['Crop_Year'].min())
    max_year = int(df['Crop_Year'].max())
    
    st.sidebar.subheader("Filter Options")
    selected_crop = st.sidebar.selectbox("Select Crop", crop_options)
    selected_state = st.sidebar.selectbox("Select State", state_options)
    selected_season = st.sidebar.selectbox("Select Season", season_options)
    year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # Q&A text input
    st.sidebar.subheader("Natural Language Query")
    query = st.sidebar.text_input("Type your query (e.g., 'Show me yield trend for Cotton in Assam during Kharif between 2000 and 2010')")
    
    if query:
        crop_filter, state_filter, season_filter, year_range_filter = process_query(query)
        st.sidebar.markdown("**Interpreted Filters:**")
        st.sidebar.write(f"Crop: {crop_filter}")
        st.sidebar.write(f"State: {state_filter}")
        st.sidebar.write(f"Season: {season_filter}")
        st.sidebar.write(f"Year Range: {year_range_filter}")
    else:
        crop_filter, state_filter, season_filter, year_range_filter = selected_crop, selected_state, selected_season, year_range
    
    # Filter the DataFrame based on selected/query-derived filters
    filtered_df = df.copy()
    if crop_filter != 'All':
        filtered_df = filtered_df[filtered_df['Crop'] == crop_filter]
    if state_filter != 'All':
        filtered_df = filtered_df[filtered_df['State'] == state_filter]
    if season_filter != 'All':
        filtered_df = filtered_df[filtered_df['Season'] == season_filter]
    filtered_df = filtered_df[(filtered_df['Crop_Year'] >= year_range_filter[0]) & 
                              (filtered_df['Crop_Year'] <= year_range_filter[1])]
    
    st.subheader("Yield Trend Over Years")
    if filtered_df.empty:
        st.write("No data available for the selected filters.")
    else:
        yearly_yield = filtered_df.groupby('Crop_Year')['Yield'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(yearly_yield['Crop_Year'], yearly_yield['Yield'], marker='o')
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Yield")
        ax.set_title("Yield Trend Over Years")
        ax.grid(True)
        st.pyplot(fig)

# --------------------------------
# Page 2: Prediction Model
# --------------------------------
elif page == "Prediction Model":
    st.header("Crop Yield Prediction")
    st.write("Enter the details below to predict the crop yield.")

    # Form for prediction inputs
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            crop_input = st.selectbox("Crop", sorted(df['Crop'].unique().tolist()))
            year_input = st.number_input("Crop Year", min_value=int(df['Crop_Year'].min()), max_value=int(df['Crop_Year'].max()), value=int(df['Crop_Year'].median()))
            season_input = st.selectbox("Season", sorted(df['Season'].unique().tolist()))
        with col2:
            state_input = st.selectbox("State", sorted(df['State'].unique().tolist()))
            rainfall_input = st.number_input("Annual Rainfall", value=float(df['Annual_Rainfall'].mean()))
            fertilizer_input = st.number_input("Fertilizer", value=float(df['Fertilizer'].mean()))
            pesticide_input = st.number_input("Pesticide", value=float(df['Pesticide'].mean()))
        submit_button = st.form_submit_button(label="Predict Yield")
    
    @st.cache(allow_output_mutation=True)
    def train_prediction_model(data):
        features = ['Crop', 'Crop_Year', 'Season', 'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        target = 'Yield'
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        categorical_features = ['Crop', 'Season', 'State']
        numerical_features = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(X_train, y_train)
        return model

    model = train_prediction_model(df)
    
    if submit_button:
        new_sample = pd.DataFrame({
            'Crop': [crop_input],
            'Crop_Year': [year_input],
            'Season': [season_input],
            'State': [state_input],
            'Annual_Rainfall': [rainfall_input],
            'Fertilizer': [fertilizer_input],
            'Pesticide': [pesticide_input]
        })
        predicted_yield = model.predict(new_sample)
        st.success(f"Predicted Crop Yield: {predicted_yield[0]:.2f}")
