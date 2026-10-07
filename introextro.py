import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Set page configuration
st.set_page_config(
    page_title="Introvert vs. Extrovert Prediction",
    page_icon="🧠",
    layout="wide"
)

# App Title & Description
st.title("🧠 Introvert and Extrovert Prediction Project")
st.markdown("""
This application uses a machine learning model to predict whether an individual leans toward **Introversion** or **Extroversion** 
based on their behavioral traits, habits, and social tendencies.
""")

# Display Top Banner Image
st.image(
    "https://keydifferences.com/wp-content/uploads/2016/07/introvert-vs-extrovert-thumbnail.jpg",
    use_container_width=True
)

st.divider()

# Sidebar - Mode Selection
app_mode = st.sidebar.selectbox("Choose Application Page", ["Predict Personality", "Dataset Insights & EDA"])

# Helper function to generate synthetic data for model training
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time_spent_Alone': np.random.choice([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], size=n_samples),
        'Stage_fear': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.4, 0.6]),
        'Social_event_attendance': np.random.uniform(0, 10, size=n_samples).round(),
        'Going_outside': np.random.uniform(0, 10, size=n_samples).round(),
        'Drained_after_socializing': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.5, 0.5]),
        'Friends_circle_size': np.random.uniform(1, 15, size=n_samples).round(),
        'Post_frequency': np.random.uniform(0, 10, size=n_samples).round(),
    }
    
    df = pd.DataFrame(data)
    
    # Assign target label based on logical rules
    score = (
        df['Social_event_attendance'] + df['Going_outside'] + df['Friends_circle_size'] + df['Post_frequency']
        - df['Time_spent_Alone'] - (df['Stage_fear'] == 'Yes') * 3 - (df['Drained_after_socializing'] == 'Yes') * 3
    )
    df['Personality'] = np.where(score > 10, 'Extrovert', 'Introvert')
    return df

# Helper function to train model
@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Personality'])
    y = df['Personality']
    
    cat_features = ['Stage_fear', 'Drained_after_socializing']
    
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.08,
        depth=5,
        cat_features=cat_features,
        verbose=0
    )
    model.fit(X, y)
    return model

df_train = load_sample_data()
model = train_model(df_train)

# PAGE 1: PREDICTION
if app_mode == "Predict Personality":
    st.subheader("📋 Enter Behavioral Features")
    st.write("Adjust the parameters below to test the personality classifier:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_spent_alone = st.slider("Time Spent Alone (Hours/Day)", 0.0, 12.0, 3.0, step=0.5)
        stage_fear = st.selectbox("Stage Fear / Fear of Public Speaking?", ["No", "Yes"])
        social_event_attendance = st.slider("Social Event Attendance (Frequency Score 0-10)", 0.0, 10.0, 5.0, step=1.0)
        going_outside = st.slider("Frequency of Going Outside (Score 0-10)", 0.0, 10.0, 4.0, step=1.0)
        
    with col2:
        drained_after_socializing = st.selectbox("Feel Drained After Socializing?", ["No", "Yes"])
        friends_circle_size = st.number_input("Friends Circle Size (Close Friends)", min_value=0, max_value=50, value=6, step=1)
        post_frequency = st.slider("Social Media Post Frequency (Posts/Week)", 0.0, 10.0, 3.0, step=1.0)

    # Input dataframe
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_spent_alone],
        'Stage_fear': [stage_fear],
        'Social_event_attendance': [social_event_attendance],
        'Going_outside': [going_outside],
        'Drained_after_socializing': [drained_after_socializing],
        'Friends_circle_size': [float(friends_circle_size)],
        'Post_frequency': [post_frequency]
    })
    
    st.markdown("---")
    
    if st.button("🔮 Predict Personality", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        prob_dict = dict(zip(classes, probabilities))
        
        st.subheader("Prediction Result:")
        
        if prediction == "Extrovert":
            st.success(f"🎉 **Predicted Class: Extrovert** (Confidence: {prob_dict['Extrovert']*100:.1f}%)")
        else:
            st.info(f"🧘 **Predicted Class: Introvert** (Confidence: {prob_dict['Introvert']*100:.1f}%)")
            
        # Display Probability breakdown
        col_a, col_b = st.columns(2)
        col_a.metric("Extrovert Probability", f"{prob_dict.get('Extrovert', 0)*100:.1f}%")
        col_b.metric("Introvert Probability", f"{prob_dict.get('Introvert', 0)*100:.1f}%")

# PAGE 2: DATASET INSIGHTS & EDA
elif app_mode == "Dataset Insights & EDA":
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    st.write("### Dataset Preview")
    st.dataframe(df_train.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Target Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_train, x='Personality', palette='Set2', ax=ax)
        ax.set_title("Distribution of Personality Types")
        st.pyplot(fig)
        
    with col2:
        st.write("### Time Spent Alone Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_train['Time_spent_Alone'], kde=True, bins=10, color='purple', ax=ax)
        ax.set_title("Distribution of Time Spent Alone")
        st.pyplot(fig)
        
    st.write("### Feature Relationships by Personality Class")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_train, x='Personality', y='Social_event_attendance', palette='Pastel1', ax=ax)
    ax.set_title("Social Event Attendance vs. Personality")
    st.pyplot(fig)