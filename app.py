import streamlit as st
import pandas as pd
from openai import OpenAI
from streamlit_chat import message

st.title("📊 Lector de CSV + Chat con LLM")

# Inicializar cliente OpenAI con secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Subida de archivo
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.success("✅ Archivo cargado correctamente")

    # Mostrar primeras filas
    st.subheader("Primeras filas del archivo")
    st.write(df.head())

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Entrada de usuario
    user_input = st.text_input("Escribe tu pregunta sobre el dataset:")

    if user_input:
        # Guardar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Construir prompt con contexto del dataset
        prompt = f"""
        Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.
        Columnas: {list(df.columns)}.
        Primeras filas:
        {df.head(3).to_string()}

        Pregunta del usuario: {user_input}
        """

        # Llamada al modelo con nueva API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista de datos que responde sobre el dataset."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        # Guardar respuesta del asistente
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Mostrar historial de chat
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
        else:
            message(msg["content"], key=f"assistant_{i}")
