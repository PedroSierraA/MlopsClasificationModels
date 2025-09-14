import streamlit as st
import pandas as pd
import joblib

# --- Rutas de los modelos ---
MODEL_PATHS = {
    "Modelo 1": "models/decision_tree_model.pkl",
    "Modelo 2": "models/knn_model.pkl"
}

# --- Cargar modelos en memoria ---
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)
    return models

models = load_models()

# --- Título y descripción ---
st.title("Predicción de Adicción al Celular")
st.markdown("Ingresa los datos y selecciona el modelo para predecir si existe **adicción al celular**.")

# --- Selector de modelo ---
selected_model = st.selectbox("Selecciona el Modelo:", list(MODEL_PATHS.keys()))

# --- Captura de datos en el orden correcto ---
gender = st.selectbox("Género", ["Male", "Female", "Other"])
school_grade = st.selectbox("Grado Escolar", ["6", "7", "8", "9", "10", "11", "12"])
daily_usage_hours = st.number_input("Horas de Uso Diario", min_value=0.0, max_value=24.0, value=4.0)
sleep_hours = st.number_input("Horas de Sueño", min_value=0.0, max_value=24.0, value=7.0)
academic_performance = st.number_input("Desempeño Académico (0-100)", min_value=0.0, max_value=100.0, value=80.0)
social_interactions = st.number_input("Interacciones Sociales (0-10)", min_value=0.0, max_value=10.0, value=5.0)
exercise_hours = st.number_input("Horas de Ejercicio Diario", min_value=0.0, max_value=10.0, value=1.0)
anxiety_level = st.number_input("Nivel de Ansiedad (0-10)", min_value=0.0, max_value=10.0, value=3.0)
depression_level = st.number_input("Nivel de Depresión (0-10)", min_value=0, max_value=10, value=2)
self_esteem = st.number_input("Autoestima (0-10)", min_value=0, max_value=10, value=7)
parental_control = st.number_input("Control Parental (0-10)", min_value=0.0, max_value=10.0, value=5.0)
screen_time_before_bed = st.number_input("Tiempo de Pantalla Antes de Dormir (horas)", min_value=0.0, max_value=10.0, value=1.0)
phone_checks_per_day = st.number_input("Chequeos del Teléfono por Día", min_value=0, max_value=500, value=50)
apps_used_daily = st.number_input("Apps Usadas al Día", min_value=0.0, max_value=50.0, value=5.0)
time_on_social_media = st.number_input("Tiempo en Redes Sociales (horas)", min_value=0.0, max_value=24.0, value=2.0)
time_on_gaming = st.number_input("Tiempo en Videojuegos (horas)", min_value=0.0, max_value=24.0, value=1.0)
time_on_education = st.number_input("Tiempo en Educación (horas)", min_value=0.0, max_value=24.0, value=1.0)
phone_usage_purpose = st.selectbox("Propósito Principal del Teléfono", ["Social", "Gaming", "Education", "Other"])
family_communication = st.number_input("Comunicación Familiar (0-10)", min_value=0.0, max_value=10.0, value=5.0)
weekend_usage_hours = st.number_input("Horas de Uso en Fines de Semana", min_value=0.0, max_value=48.0, value=6.0)

# --- Botón para predecir ---
if st.button("Predecir Adicción"):
    # Crear DataFrame con el orden exacto de columnas
    input_data = pd.DataFrame([[
        gender, school_grade, daily_usage_hours, sleep_hours, academic_performance,
        social_interactions, exercise_hours, anxiety_level, depression_level, self_esteem,
        parental_control, screen_time_before_bed, phone_checks_per_day, apps_used_daily,
        time_on_social_media, time_on_gaming, time_on_education, phone_usage_purpose,
        family_communication, weekend_usage_hours
    ]], columns=[
        "Gender","School_Grade","Daily_Usage_Hours","Sleep_Hours","Academic_Performance",
        "Social_Interactions","Exercise_Hours","Anxiety_Level","Depression_Level","Self_Esteem",
        "Parental_Control","Screen_Time_Before_Bed","Phone_Checks_Per_Day","Apps_Used_Daily",
        "Time_on_Social_Media","Time_on_Gaming","Time_on_Education","Phone_Usage_Purpose",
        "Family_Communication","Weekend_Usage_Hours"
    ])

    # Seleccionar modelo
    model = models[selected_model]

    # Hacer predicción
    prediction = model.predict(input_data)

    # Mostrar resultado
    st.subheader("Resultado de la Predicción:")
    st.write("Adicción al Celular" if prediction[0] == 1 else "No Adicción")
