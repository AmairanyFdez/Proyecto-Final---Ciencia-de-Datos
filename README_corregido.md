** Proyecto Final – Ciencia de Datos ** 
Análisis y Predicción del Índice de Desarrollo Humano (HDI) 

Este proyecto desarrolla un análisis completo sobre el Human Development Index (HDI) utilizando datos históricos desde 1990, así como un modelo de Machine Learning para predecir el HDI en función de factores socioeconómicos clave.
Además, incluye una aplicación interactiva en Streamlit que permite explorar los datos, visualizar tendencias y probar un simulador de predicción.

Contenido del repositorio: 

Final.py – Código completo de la aplicación en Streamlit.

Notebook_Analisis_HDI.ipynb – Notebook con el análisis exploratorio, tendencias y modelo de ML.

Data/ – Carpeta con el dataset: HumanDevelopmentIndex.csv

READMEcorregido.md – Este archivo.

ReporteEjecutivo.pdf – Documento final entregable.

VideoPitch.mp4 – Presentación tipo pitch del proyecto.

🎯 1. Pregunta de negocio

El objetivo principal es responder:

¿Cómo ha evolucionado el HDI global en las últimas décadas y qué factores determinan mejor el desarrollo humano en los países?

Además, se busca identificar:

- ¿Qué países han mejorado, retrocedido o se han estancado?

- ¿Qué países tienen alto ingreso pero bajo desarrollo?

- ¿Qué variables explican mejor el HDI?


📊 2. Enfoque técnico

El análisis combina limpieza de datos, análisis exploratorio, modelado predictivo y visualización interactiva.

🔹 a) Procesamiento del dataset

Limpieza: valores atípicos → NaN

Normalización de columnas

Transformación del HDI a formato largo (1990–2021)

🔹 b) Análisis Exploratorio 

Incluye: Histogramas, Evolución del HDI global, Comparaciones GNI vs HDI, Tendencias por país, Matriz de correlación (HDI, GNIpc, LE, EYS, MYS)

Gráficos usados: Heatmap, Scatterplots, Histogramas, Líneas de tiempo, Barras comparativas

🔹 c) Modelo de Machine Learning

Se incorporó un modelo predictivo para estimar el HDI usando factores clave

Variables del modelo: GNI per cápita (2021), Esperanza de vida, Expected years of schooling, Mean years of schooling

Modelos disponibles: Regresión Lineal, Random Forest

Métricas calculadas: RMSE, MAE, R²

Comparación HDI real vs predicho


🌐 3. Aplicación en Streamlit

La aplicación incluye 4 secciones:

- 1 Inicio: Descripción general del HDI y navegación.

- 2 Análisis Exploratorio: Filtros por región y grupo de desarrollo. Visualizaciones de distribución, correlaciones y tendencias.

- 3 Preguntas Clave: Comparaciones significativas -> Países que mejoran/retroceden, Desarrollo vs ingreso, Factores explicativos del HDI

- 4 Modelo de Predicción: Entrenamiento y evaluación de modelo ML.

📁 4. Estructura del proyecto
📦 ProyectoFinal_HDI
 ┣ 📂 Data
 │  ┗ 📄 HumanDevelopmentIndex.csv
 ┣ 📄 Final.py
 ┣ 📄 Notebook_Analisis_HDI.ipynb
 ┣ 📄 README.md
 ┣ 📄 ReporteEjecutivo.pdf
 ┗ 📄 VideoPitch.mp4 

▶️ 5. Ejecutar la aplicación

Ejecutar Streamlit: streamlit run Final.py

📌 6. Dataset

Fuente: Kaggle – Human Development Index Dataset

Registros: 7,600 observaciones

Cobertura: 1990–2021, 190 países

Variables: HDI y sus componentes 

🎥 7. Video Pitch

Liga: (https://drive.google.com/file/d/1xkTQPMzhoOoD08Gr0QJHIkTF_uZNGAVK/view?usp=sharing)

🧠 8. Conclusiones clave

El HDI ha mostrado una tendencia global positiva, aunque con retrocesos en países afectados por crisis.

No siempre un mayor ingreso se traduce en mayor desarrollo humano.

Salud y educación son los factores más determinantes del HDI.

El modelo de ML permite estimar escenarios futuros y comparar países bajo condiciones hipotéticas.

El proyecto crea una herramienta valiosa para política pública, análisis social y toma de decisiones.
