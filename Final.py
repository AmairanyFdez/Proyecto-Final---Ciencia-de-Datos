import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# FUNCIÓN DE LIMPIEZA
# =========================================================
def clean_hdi_data(df):
    """Limpieza de dataset"""

    df = df.copy()

    # 1) Se cambiaron valores raros por NaN
    df = df.replace({"..": np.nan, "--": np.nan, "—": np.nan})

    # 2) Se eliminaron duplicados por país
    if "Country" in df.columns:
        df = df.drop_duplicates(subset=["Country"], keep="first")

    # 3) Se detectaron columnas numéricas (dejamos fuera las categóricas)
    cat_cols = ["Country", "Human Development Groups", "UNDP Developing Regions"]
    num_cols = [c for c in df.columns if c not in cat_cols]

    # 4) Conversión de esas columnas a numéricas
    for col in num_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # 5) Se rellenaron nulos en columnas numéricas con la mediana
    for col in num_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    # 6) Se rellenaron nulos en variables categóricas
    for c in ["Human Development Groups", "UNDP Developing Regions"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    return df



# =========================================================
# CARGA DE DATOS YA CON LIMPIEZA
# =========================================================
@st.cache_data
def load_data():
    df_raw = pd.read_csv("Data/HumanDevelopmentIndex.csv")
    df_clean = clean_hdi_data(df_raw)
    return df_clean


# =========================================================
# FUNCIONES AUXILIARES DE TRANSFORMACIÓN
# =========================================================
@st.cache_data
def get_hdi_columns(df):
    """Obtiene las columnas de HDI por año y las convierte a formato largo."""
    hdi_cols = [
        c for c in df.columns
        if c.startswith("Human Development Index (") and "Planetary" not in c
    ]
    years = [int(c.split("(")[1].split(")")[0]) for c in hdi_cols]

    hdi_long = df[["Country"] + hdi_cols].melt(
        id_vars="Country",
        value_vars=hdi_cols,
        var_name="Year",
        value_name="HDI"
    )
    hdi_long["Year"] = hdi_long["Year"].str.extract(r"(\d{4})").astype(int)

    # Asegurar que HDI sea numérico
    hdi_long["HDI"] = (
        hdi_long["HDI"]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
    )
    hdi_long["HDI"] = pd.to_numeric(hdi_long["HDI"], errors="coerce")

    return hdi_long, sorted(list(set(years)))

@st.cache_data
def get_main_2021_df(df):
    """Subset con variables clave para 2021 y nombres simplificados."""
    cols = [
        "Country",
        "Human Development Groups",
        "UNDP Developing Regions",
        "Human Development Index (2021)",
        "Gross National Income Per Capita (2021)",
        "Life Expectancy at Birth (2021)",
        "Expected Years of Schooling (2021)",
        "Mean Years of Schooling (2021)"
    ]
    existing = [c for c in cols if c in df.columns]
    sub = df[existing].copy()

    # Renombrar columnas
    sub = sub.rename(columns={
        "Human Development Groups": "HDI_Group",
        "UNDP Developing Regions": "Region",
        "Human Development Index (2021)": "HDI_2021",
        "Gross National Income Per Capita (2021)": "GNIpc_2021",
        "Life Expectancy at Birth (2021)": "LE_2021",
        "Expected Years of Schooling (2021)": "EYS_2021",
        "Mean Years of Schooling (2021)": "MYS_2021"
    })

    # Forzar a numéricas las columnas que usaremos en cálculos
    for col in ["HDI_2021", "GNIpc_2021", "LE_2021", "EYS_2021", "MYS_2021"]:
        if col in sub.columns:
            sub[col] = (
                sub[col]
                .astype(str)
                .str.replace(",", "")
                .str.strip()
            )
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    return sub

@st.cache_data
def compute_hdi_trends(df, start_year=1990, end_year=2021):
    """
    Calcula el cambio de HDI entre dos años para cada país
    y clasifica si mejoró, empeoró o se estancó.
    """
    # Usamos la versión "larga" del HDI
    hdi_long, years = get_hdi_columns(df)

    # Filtrar solo años dentro del rango seleccionado
    hdi_period = hdi_long[hdi_long["Year"].between(start_year, end_year)].copy()

    # Asegurar que HDI sea numérico
    hdi_period["HDI"] = pd.to_numeric(hdi_period["HDI"], errors="coerce")

    # HDI al inicio del periodo
    start_df = (
        hdi_period[hdi_period["Year"] == start_year][["Country", "HDI"]]
        .rename(columns={"HDI": "HDI_start"})
    )

    # HDI al final del periodo
    end_df = (
        hdi_period[hdi_period["Year"] == end_year][["Country", "HDI"]]
        .rename(columns={"HDI": "HDI_end"})
    )

    # Unir inicio y fin
    temp = pd.merge(start_df, end_df, on="Country", how="inner")

    # Asegurar numéricos
    for col in ["HDI_start", "HDI_end"]:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    # Calcular cambio
    temp["HDI_change"] = temp["HDI_end"] - temp["HDI_start"]

    # Quitar filas sin datos válidos
    temp = temp.dropna(subset=["HDI_start", "HDI_end", "HDI_change"])

    # Umbrales para clasificar – puedes ajustarlos si quieres
    mejora_umbral = 0.05    # +0.05 o más en HDI = mejora fuerte
    empeora_umbral = -0.02  # -0.02 o menos = retroceso

    categorias = []
    for delta in temp["HDI_change"]:
        if delta >= mejora_umbral:
            categorias.append("Mejora importante")
        elif delta <= empeora_umbral:
            categorias.append("Retroceso")
        else:
            categorias.append("Estancado")

    # Nombre de columna alineado con el resto del código
    temp["Trend_Category"] = categorias

    return temp

@st.cache_data
def prepare_ml_data(df_full):
    """Prepara datos de 2021 para un modelo de regresión que predice HDI."""
    d2021 = get_main_2021_df(df_full)
    features = ["GNIpc_2021", "LE_2021", "EYS_2021", "MYS_2021"]
    target = "HDI_2021"
    data = d2021.dropna(subset=features + [target]).copy()
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test, features, data


# =========================================================
# CONFIGURACIÓN BÁSICA DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Desarrollo Humano y Desigualdad",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ESTILOS
# =========================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 15px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 10px 0px;
    }
    section[data-testid="stSidebar"] {
    background-color: #1A1A1A;  /* Negro bonito */
    color: #FFFFFF;             /* Texto blanco */
    }
    </style>
""", unsafe_allow_html=True)

# Paleta de colores global
PRIMARY = "#2B6CB0"     # Azul principal
SECONDARY = "#4A5568"   # Gris oscuro
ACCENT = "#38B2AC"      # Verde agua
DANGER = "#E53E3E"      # Rojo para alertas

# =========================================================
# CARGA DE DATOS
# =========================================================
df = load_data()
hdi_long, hdi_years = get_hdi_columns(df)
df_2021 = get_main_2021_df(df)

# =========================================================
# SIDEBAR - NAVEGACIÓN
# =========================================================
st.sidebar.markdown("""
<div style="
    padding:12px; 
    border-radius:10px; 
    background:#2A2A2A; 
    border:1px solid #444;
">
    <h3 style="color:#FFFFFF; margin-bottom:5px;"> Panel de Desarrollo Humano</h3>
    <p style="font-size:13px; color:#CCCCCC; line-height:1.4;">
        Explora la evolución del HDI, detecta desigualdades y contrasta riqueza vs calidad de vida.
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🏠 Inicio",
        "📈 Análisis Exploratorio",
        "🧩 Preguntas Clave de Desarrollo Humano",
        "🤖 Modelo de Predicción HDI"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**📚 Proyecto: Desarrollo Humano**

Tecnologías:
- 🐼 Pandas
- 🤖 Scikit-learn
- 📊 Plotly
- 🚀 Streamlit
""")

# =========================================================
# PÁGINA: INICIO
# =========================================================
if page == "🏠 Inicio":
    # Hero banner
    st.markdown("""
    <div style="text-align:center; padding:25px; background:#f0f2f6; border-radius:16px; margin-bottom:10px;">
        <h1 style="color:#2B6CB0; margin-bottom:0.4rem;">🌎 Análisis Global del Desarrollo Humano</h1>
        <p style="font-size:16px; color:#4A5568; max-width:700px; margin:0 auto;">
            Exploración interactiva del Índice de Desarrollo Humano (HDI) y su relación con la riqueza,
            la salud y la educación, con el objetivo de identificar los países que han mejorado, empeorado
            o se han estancado.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌍 Países en el dataset",
            f"{df['Country'].nunique():,}"
        )
    with col2:
        hdi_mean = df_2021["HDI_2021"].mean()
        st.metric(
            "📊 HDI promedio (2021)",
            f"{hdi_mean:.3f}"
        )
    with col3:
        gni_mean = df_2021["GNIpc_2021"].mean()
        st.metric(
            "💰 GNI per cápita promedio (2021)",
            f"${gni_mean:,.0f}"
        )
    with col4:
        le_mean = df_2021["LE_2021"].mean()
        st.metric(
            "🩺 Esperanza de vida promedio (2021)",
            f"{le_mean:.1f} años"
        )


    st.markdown("### 🗺️ Mapa mundial del HDI (2021)")
    if "HDI_2021" in df_2021.columns:
        fig_map = px.choropleth(
            df_2021,
            locations="Country",
            locationmode="country names",
            color="HDI_2021",
            color_continuous_scale="Viridis",
            title="Mapa interactivo del Índice de Desarrollo Humano (2021)",
            labels={"HDI_2021": "HDI (2021)"}
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No se encontró la columna HDI_2021 para generar el mapa.")

    st.markdown("### 📋 Vista preliminar de los datos")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### ℹ️ Estadísticas básicas (2021)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Variables clave:**")
        for col in ["HDI_2021", "GNIpc_2021", "LE_2021", "EYS_2021", "MYS_2021"]:
            if col in df_2021.columns:
                st.write(f"- `{col}`")
    with c2:
        subcols = [c for c in ["HDI_2021", "GNIpc_2021", "LE_2021", "EYS_2021", "MYS_2021"] if c in df_2021.columns]
        if subcols:
            st.dataframe(df_2021[subcols].describe(), use_container_width=True)


# =========================================================
# PÁGINA: ANÁLISIS EXPLORATORIO
# =========================================================
elif page == "📈 Análisis Exploratorio":
    st.markdown('<h2 class="sub-header">Análisis Exploratorio de Desarrollo Humano</h2>', unsafe_allow_html=True)

    # Filtros básicos
    regions = sorted(df_2021["Region"].dropna().unique())
    groups = sorted(df_2021["HDI_Group"].dropna().unique())

    col_filters = st.columns(2)
    with col_filters[0]:
        region_filter = st.multiselect(
            "Filtrar por región (UNDP):",
            options=regions,
            default=regions
        )
    with col_filters[1]:
        group_filter = st.multiselect(
            "Filtrar por grupo de desarrollo humano:",
            options=groups,
            default=groups
        )

    df_filt = df_2021.copy()
    if region_filter:
        df_filt = df_filt[df_filt["Region"].isin(region_filter)]
    if group_filter:
        df_filt = df_filt[df_filt["HDI_Group"].isin(group_filter)]

    tab1, tab2, tab3 = st.tabs(["📊 Distribuciones", "🔗 Relaciones", "📈 Tendencias"])

    # ------------------ DISTRIBUCIONES ------------------
    with tab1:
        st.markdown("### Distribuciones globales")

        c1, c2 = st.columns(2)
        with c1:
            if "HDI_2021" in df_filt.columns:
                fig_hdi = px.histogram(
                    df_filt,
                    x="HDI_2021",
                    nbins=30,
                    title="Distribución del HDI (2021)",
                    labels={"HDI_2021": "HDI (2021)"},
                    color_discrete_sequence=[PRIMARY]
                )
                fig_hdi.update_layout(bargap=0.05)
                st.plotly_chart(fig_hdi, use_container_width=True)
        with c2:
            if "GNIpc_2021" in df_filt.columns:
                fig_gni = px.histogram(
                    df_filt,
                    x="GNIpc_2021",
                    nbins=30,
                    title="Distribución de GNI per cápita (2021)",
                    labels={"GNIpc_2021": "GNI per cápita (USD)"},
                    color_discrete_sequence=[ACCENT]
                )
                fig_gni.update_layout(bargap=0.05)
                st.plotly_chart(fig_gni, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            group_counts = df_filt["HDI_Group"].value_counts().reset_index()
            group_counts.columns = ["HDI_Group", "Count"]
            fig_group = px.bar(
                group_counts,
                x="HDI_Group",
                y="Count",
                title="Países por grupo de desarrollo humano",
                labels={"HDI_Group": "Grupo", "Count": "Países"},
                color="Count",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_group, use_container_width=True)
        with c4:
            region_counts = df_filt["Region"].value_counts().reset_index()
            region_counts.columns = ["Region", "Count"]
            fig_reg = px.bar(
                region_counts,
                x="Region",
                y="Count",
                title="Países por región (UNDP)",
                labels={"Region": "Región", "Count": "Países"},
                color="Count",
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig_reg, use_container_width=True)

    # ------------------ RELACIONES ------------------
    with tab2:
        st.markdown("### Relaciones entre desarrollo, riqueza y salud")
        c1, c2 = st.columns(2)
        with c1:
            if {"GNIpc_2021", "HDI_2021"}.issubset(df_filt.columns):
                fig = px.scatter(
                    df_filt,
                    x="GNIpc_2021",
                    y="HDI_2021",
                    color="Region",
                    hover_name="Country",
                    title="HDI vs GNI per cápita (2021)",
                    labels={"GNIpc_2021": "GNI per cápita (USD)", "HDI_2021": "HDI (2021)"},
                )
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if {"LE_2021", "HDI_2021"}.issubset(df_filt.columns):
                fig2 = px.scatter(
                    df_filt,
                    x="LE_2021",
                    y="HDI_2021",
                    color="HDI_Group",
                    hover_name="Country",
                    title="HDI vs Esperanza de vida (2021)",
                    labels={"LE_2021": "Esperanza de vida (años)", "HDI_2021": "HDI (2021)"},
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ------------------ TENDENCIAS ------------------
    with tab3:
        st.markdown("### Tendencias de HDI a lo largo del tiempo")

        countries_available = sorted(hdi_long["Country"].unique())
        default_countries = []
        for ctry in ["Mexico", "United States", "Norway"]:
            if ctry in countries_available:
                default_countries.append(ctry)
        if not default_countries:
            default_countries = countries_available[:3]

        selected_countries = st.multiselect(
            "Selecciona países:",
            options=countries_available,
            default=default_countries
        )

        hdi_plot = hdi_long[hdi_long["Country"].isin(selected_countries)]
        fig_ts = px.line(
            hdi_plot,
            x="Year",
            y="HDI",
            color="Country",
            title="Evolución del HDI por país",
            labels={"Year": "Año", "HDI": "HDI"}
        )
        fig_ts.update_traces(mode="lines+markers")
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("### 🌍 Mapa animado: evolución del HDI en el mundo")
        try:
            fig_anim = px.choropleth(
                hdi_long,
                locations="Country",
                locationmode="country names",
                color="HDI",
                animation_frame="Year",
                color_continuous_scale="Plasma",
                title="Evolución del HDI por país (animación por año)",
                labels={"HDI": "HDI"}
            )
            fig_anim.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_anim, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar el mapa animado: {e}")

# =========================================================
# PÁGINA: PREGUNTAS CLAVE
# =========================================================
elif page == "🧩 Preguntas Clave de Desarrollo Humano":
    st.markdown('<h2 class="sub-header">Preguntas Clave de Desarrollo Humano</h2>', unsafe_allow_html=True)
    st.write(
        "En esta sección se responden tres preguntas centrales usando el HDI."
    )

    # ------------------ PREGUNTA 1 ------------------
    st.markdown("## 1️⃣ ¿Qué países han mejorado, empeorado o se han estancado en HDI?")

    c1, c2 = st.columns(2)
    with c1:
        start_year = st.select_slider(
            "Año inicial:",
            options=hdi_years,
            value=min(hdi_years)
        )
    with c2:
        end_year = st.select_slider(
            "Año final:",
            options=hdi_years,
            value=max(hdi_years)
        )

    if start_year >= end_year:
        st.warning("El año inicial debe ser menor que el año final.")
    else:
        trends_df = compute_hdi_trends(df, start_year=start_year, end_year=end_year)

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("Países analizados", len(trends_df))
        with col_kpi2:
            st.metric("Mejora importante", int((trends_df["Trend_Category"] == "Mejora importante").sum()))
        with col_kpi3:
            st.metric("Estancamiento", int((trends_df["Trend_Category"] == "Estancamiento").sum()))
        with col_kpi4:
            st.metric("Empeoramiento", int((trends_df["Trend_Category"] == "Empeoramiento").sum()))

        st.markdown("### Top 10 países que más mejoraron su HDI")
        top_up = trends_df.sort_values("HDI_change", ascending=False).head(10)
        fig_up = px.bar(
            top_up,
            x="Country",
            y="HDI_change",
            title=f"Top 10 mejoras en HDI ({start_year}–{end_year})",
            labels={"Country": "País", "HDI_change": "Δ HDI"},
            color="HDI_change",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_up, use_container_width=True)

        st.markdown("### Top 10 países que más retrocedieron en HDI")
        top_down = trends_df.sort_values("HDI_change", ascending=True).head(10)
        fig_down = px.bar(
            top_down,
            x="Country",
            y="HDI_change",
            title=f"Top 10 retrocesos en HDI ({start_year}–{end_year})",
            labels={"Country": "País", "HDI_change": "Δ HDI"},
            color="HDI_change",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_down, use_container_width=True)

        with st.expander("Ver tabla completa de cambios en HDI"):
            st.dataframe(
                trends_df[["Country", "HDI_start", "HDI_end", "HDI_change", "Trend_Category"]]
                .sort_values("HDI_change", ascending=False),
                use_container_width=True
            )

    st.markdown("---")
    # ------------------ PREGUNTA 2 ------------------
    st.markdown("## 2️⃣ ¿Qué países tienen alto GNI pero bajo HDI?")

    c1, c2 = st.columns(2)
    with c1:
        high_income_percentile = st.slider(
            "Percentil para 'alto GNI per cápita':",
            min_value=60,
            max_value=95,
            value=75,
            step=5
        )
    with c2:
        low_hdi_percentile = st.slider(
            "Percentil máximo para 'bajo HDI':",
            min_value=10,
            max_value=60,
            value=50,
            step=5
        )

    gni_thresh = np.nanpercentile(df_2021["GNIpc_2021"].dropna(), high_income_percentile)
    hdi_thresh = np.nanpercentile(df_2021["HDI_2021"].dropna(), low_hdi_percentile)

    cond_high_gni = df_2021["GNIpc_2021"] >= gni_thresh
    cond_low_hdi = df_2021["HDI_2021"] <= hdi_thresh

    high_gni_low_hdi = df_2021[cond_high_gni & cond_low_hdi].copy()
    high_gni_low_hdi = high_gni_low_hdi.sort_values("GNIpc_2021", ascending=False)

    st.markdown(
        f"Con estos umbrales, considero **alto GNI ≥ ${gni_thresh:,.0f}** "
        f"y **bajo HDI ≤ {hdi_thresh:.3f}**."
    )

    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.metric("Países con alto GNI y bajo HDI", int(len(high_gni_low_hdi)))
    with col_kpi2:
        if len(high_gni_low_hdi) > 0:
            mean_hdi_group = high_gni_low_hdi["HDI_2021"].mean()
            st.metric("HDI promedio del grupo", f"{mean_hdi_group:.3f}")

    df_2021["Grupo_P2"] = "Otros países"
    df_2021.loc[high_gni_low_hdi.index, "Grupo_P2"] = "Alto GNI - Bajo HDI"

    fig_scatter_p2 = px.scatter(
        df_2021,
        x="GNIpc_2021",
        y="HDI_2021",
        color="Grupo_P2",
        hover_name="Country",
        title="Países con alto GNI pero bajo HDI",
        labels={"GNIpc_2021": "GNI per cápita (USD)", "HDI_2021": "HDI (2021)"},
        color_discrete_map={
            "Otros países": SECONDARY,
            "Alto GNI - Bajo HDI": DANGER
        }
    )
    st.plotly_chart(fig_scatter_p2, use_container_width=True)

    st.markdown("### Lista de países con alto GNI pero bajo HDI")
    st.dataframe(
        high_gni_low_hdi[["Country", "Region", "HDI_Group", "GNIpc_2021", "HDI_2021"]],
        use_container_width=True
    )

    st.markdown("---")

    # ------------------ PREGUNTA 3 ------------------

    st.markdown("---")
    st.markdown("### 3️⃣ ¿Qué factores explican mejor el HDI?")

    st.markdown(
        """
        En esta sección analizamos qué tanto contribuyen la **salud** (esperanza de vida), 
        la **educación** (años esperados y promedio de escolaridad) y la **riqueza** (GNI per cápita)
        al nivel de HDI de cada país en 2021.
        """
    )

    # Usamos el subset limpio de 2021
    d2021 = get_main_2021_df(df)

    # Asegurarnos de que existan las columnas necesarias
    cols_q3 = ["Region", "HDI_2021", "LE_2021", "EYS_2021", "MYS_2021", "GNIpc_2021"]
    cols_q3_exist = [c for c in cols_q3 if c in d2021.columns]

    # Armamos el DataFrame para análisis, incluyendo SIEMPRE 'Country' para el hover
    q3_df = d2021[["Country"] + cols_q3_exist].dropna().copy()

    # ---- Correlación entre HDI y los factores ----
    st.markdown("#### 🔍 Correlación entre HDI y factores clave (2021)")

    corr_cols = [c for c in ["HDI_2021", "LE_2021", "EYS_2021", "MYS_2021", "GNIpc_2021"] if c in q3_df.columns]
    corr = q3_df[corr_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        labels={"color": "Correlación"},
        title="Matriz de correlación entre HDI y sus determinantes (2021)"
    )
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(
        """
        Valores cercanos a **1** indican que el factor se mueve casi igual que el HDI,  
        valores cercanos a **0** indican poca relación,  
        y negativos indicarían que se mueven en sentido contrario.
        """
    )

    # ---- Relación HDI vs un factor elegido ----
    st.markdown("#### 📈 Relación entre el HDI y un factor específico ↪")

    factor_opcion = st.selectbox(
        "Elige un factor para comparar contra el HDI:",
        ["Esperanza de vida", "Años esperados de escolaridad", "Años promedio de escolaridad", "GNI per cápita"]
    )

    factor_map = {
        "Esperanza de vida": ("LE_2021", "Esperanza de vida (años)"),
        "Años esperados de escolaridad": ("EYS_2021", "Años esperados de escolaridad"),
        "Años promedio de escolaridad": ("MYS_2021", "Años promedio de escolaridad"),
        "GNI per cápita": ("GNIpc_2021", "GNI per cápita (US$)")
    }

    factor_col, factor_label = factor_map[factor_opcion]

    if factor_col in q3_df.columns:
        fig_scatter = px.scatter(
    q3_df,
    x=factor_col,
    y="HDI_2021",
    color="Region" if "Region" in q3_df.columns else None,
    hover_name="Country",
    labels={
        factor_col: factor_label,
        "HDI_2021": "HDI (2021)"
    },
    title=f"Relación entre {factor_label} y el HDI (2021)"
)
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown(
            f"""
            Cada punto es un país. Si la nube de puntos tiende a subir hacia la derecha, 
            significa que **a mayor {factor_label.lower()}, mayor HDI**.  
            La línea de tendencia ayuda a ver la relación general.
            """
        )
    else:
        st.warning("El factor seleccionado no está disponible en el dataset procesado.")

    # ------------ CONCLUSIÓN AUTOMÁTICA ------------

default_trends = compute_hdi_trends(df, start_year=1990, end_year=2021)

st.markdown("### 🧾 Conclusión automática del análisis")

# Usamos las tendencias globales 1990–2021
top_improver_row = default_trends.sort_values("HDI_change", ascending=False).iloc[0]
top_decliner_row = default_trends.sort_values("HDI_change", ascending=True).iloc[0]

st.write(
    f"- 🌱 El país con **mayor mejora** en HDI entre 1990 y 2021 es "
    f"**{top_improver_row['Country']}**, con un cambio de **{top_improver_row['HDI_change']:.3f} puntos**."
)

st.write(
    f"- ⚠️ El país con **mayor retroceso** en HDI entre 1990 y 2021 es "
    f"**{top_decliner_row['Country']}**, con un cambio de **{top_decliner_row['HDI_change']:.3f} puntos**."
)

st.write(
    f"- En total, **{(default_trends['Trend_Category'] == 'Mejora importante').sum()} países** muestran una "
    f"mejora importante, mientras que **{(default_trends['Trend_Category'] == 'Retroceso').sum()}** registran "
    f"retrocesos y **{(default_trends['Trend_Category'] == 'Estancado').sum()}** se mantienen prácticamente estancados."
)

# =========================================================
# PÁGINA: MODELO DE PREDICCIÓN 
# =========================================================
if page == "🤖 Modelo de Predicción HDI":
    st.markdown('<h2 class="sub-header">Modelo de Predicción del HDI</h2>', unsafe_allow_html=True)
    st.info("🤖 Modelo de regresión para predecir el HDI (2021) a partir de variables socioeconómicas básicas.")

    X_train, X_test, y_train, y_test, features, data_full = prepare_ml_data(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Datos de entrenamiento", f"{len(X_train)} países")
    with col2:
        st.metric("🧪 Datos de prueba", f"{len(X_test)} países")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        model_type = st.selectbox(
            "Elige el algoritmo de Machine Learning:",
            ["Regresión Lineal", "Random Forest"],
            help="Regresión Lineal: simple e interpretable. Random Forest: más flexible y no lineal."
        )
    with col2:
        st.markdown("**Variables usadas:**")
        st.markdown("- GNI per cápita (2021)")
        st.markdown("- Esperanza de vida (2021)")
        st.markdown("- Expected years of schooling (2021)")
        st.markdown("- Mean years of schooling (2021)")

    if st.button("🚀 Entrenar modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo..."):
            if model_type == "Regresión Lineal":
                model = LinearRegression()
                model_name = "Linear Regression"
            else:
                model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=6,
                    random_state=42
                )
                model_name = "Random Forest Regressor"

            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)

            st.success(f"✅ Modelo {model_name} entrenado correctamente.")

            st.markdown("### 📈 Métricas del modelo")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("RMSE (test)", f"{rmse_test:.3f}")
            with c2:
                st.metric("R² (test)", f"{r2_test:.3f}")
            with c3:
                st.metric("MAE (test)", f"{mae_test:.3f}")
            with c4:
                st.metric("Países en test", f"{len(y_test)}")

            st.markdown("### 🎯 HDI real vs HDI predicho (datos de prueba)")
            pred_df = pd.DataFrame({
                "Real": y_test,
                "Predicho": y_pred_test
            })

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=pred_df["Real"],
                y=pred_df["Predicho"],
                mode="markers",
                name="Predicciones",
                marker=dict(size=8, opacity=0.7, color=PRIMARY)
            ))

            min_val = min(pred_df["Real"].min(), pred_df["Predicho"].min())
            max_val = max(pred_df["Real"].max(), pred_df["Predicho"].max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Línea perfecta",
                line=dict(color=DANGER, dash="dash")
            ))

            fig_pred.update_layout(
                title="HDI real vs HDI predicho",
                xaxis_title="HDI real",
                yaxis_title="HDI predicho",
                height=500
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            if model_type == "Random Forest":
                st.markdown("### 🔍 Importancia de las variables")
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importancia": model.feature_importances_
                }).sort_values("Importancia", ascending=True)
                fig_imp = px.bar(
                    importance_df,
                    x="Importancia",
                    y="Feature",
                    orientation="h",
                    title="Importancia de cada variable en el modelo",
                    color="Importancia",
                    color_continuous_scale="Teal"
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("### 🧪 Simulador de país hipotético")
            st.write(
                "Aquí puedes mover las variables para crear un país hipotético y ver qué HDI le asignaría el modelo."
            )

            col_a, col_b = st.columns(2)
            with col_a:
                gni_input = st.slider(
                    "GNI per cápita (USD, 2021)",
                    min_value=int(data_full["GNIpc_2021"].min()),
                    max_value=int(data_full["GNIpc_2021"].max()),
                    value=int(data_full["GNIpc_2021"].median())
                )
                le_input = st.slider(
                    "Esperanza de vida (años)",
                    min_value=float(data_full["LE_2021"].min()),
                    max_value=float(data_full["LE_2021"].max()),
                    value=float(data_full["LE_2021"].median())
                )
            with col_b:
                eys_input = st.slider(
                    "Expected years of schooling",
                    min_value=float(data_full["EYS_2021"].min()),
                    max_value=float(data_full["EYS_2021"].max()),
                    value=float(data_full["EYS_2021"].median())
                )
                mys_input = st.slider(
                    "Mean years of schooling",
                    min_value=float(data_full["MYS_2021"].min()),
                    max_value=float(data_full["MYS_2021"].max()),
                    value=float(data_full["MYS_2021"].median())
                )

            X_new = np.array([[gni_input, le_input, eys_input, mys_input]])
            hdi_pred_new = model.predict(X_new)[0]

            st.metric(
                "HDI estimado para el país hipotético",
                f"{hdi_pred_new:.3f}"
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 10px 0;'>
        <p>Proyecto de Ciencia de Datos | Desarrollo Humano</p>
        <p>Construido con <b>Streamlit</b>, <b>Pandas</b>, <b>Scikit-learn</b> y <b>Plotly</b></p>
    </div>
    """,
    unsafe_allow_html=True
)