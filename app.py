
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress
import plotly.express as px
from PIL import Image


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brno | Urban Heat Explorer",
    page_icon="🌡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

BLUE = "#2679AC"
ORANGE = "#E18A42"
RED = "#B05A4A"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    landsat = pd.read_csv(DATA_DIR / "landsat.csv")
    era5 = pd.read_csv(DATA_DIR / "era5_land.csv")

    landsat = landsat.sort_values("year")
    era5 = era5.sort_values("year")

    return landsat, era5


df_lst, df_era5 = load_data()

# Landsat: exclude years with less than 90% coverage
lst_valid = df_lst[df_lst["coverage_pct"] >= 90]

lst_trend = linregress(
    lst_valid["year"],
    lst_valid["mean_lst_c"]
)

era5_trend = linregress(
    df_era5["year"],
    df_era5["air_temp_c"]
)


# --------------------------------------------------
# PLOTLY CHART
# --------------------------------------------------

def create_chart(df, y_column, trend, title, is_landsat=False):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df[y_column],
            mode="lines+markers",
            name="Annual summer temperature",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=8),
            hovertemplate=(
                "Year: %{x}<br>"
                "Temperature: %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

    years = np.arange(2001, 2026)

    fig.add_trace(
        go.Scatter(
            x=years,
            y=trend.intercept + trend.slope * years,
            mode="lines",
            name="Estimated trend",
            line=dict(
                color=ORANGE,
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "Year: %{x}<br>"
                "Trend estimate: %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

    if is_landsat:
        excluded = df[df["coverage_pct"] < 90]

        fig.add_trace(
            go.Scatter(
                x=excluded["year"],
                y=excluded[y_column],
                mode="markers",
                name="Low coverage (excluded from trend)",
                marker=dict(
                    color=RED,
                    size=12,
                    symbol="x"
                ),
                customdata=excluded["coverage_pct"],
                hovertemplate=(
                    "Year: %{x}<br>"
                    "Temperature: %{y:.2f} °C<br>"
                    "Spatial coverage: %{customdata:.1f}%<br>"
                    "Excluded from regression"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=370,
        font=dict(
            family="Arial",
            size=13,
            color="#263746"
        ),
        margin=dict(l=20, r=20, t=55, b=35),
        xaxis=dict(
            title="Year",
            dtick=4,
            gridcolor="#E9EDF1"
        ),
        yaxis=dict(
            title="Temperature (°C)",
            gridcolor="#E9EDF1"
        ),
        legend=dict(
            orientation="h",
            y=-0.28,
            x=0
        ),
        hovermode="closest"
    )

    # Softer grid lines for the dark dashboard
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(150, 160, 175, 0.15)",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(150, 160, 175, 0.20)",
        zeroline=False
    )

    return fig


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("BRNO | URBAN HEAT EXPLORER")

st.caption(
    "Interactive urban heat assessment | "
    "Summer (June–August), 2001–2025"
)

st.markdown(
    "Explore how summer temperatures have changed "
    "in Brno using satellite observations and "
    "climate reanalysis."
)

st.divider()


# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.metric(
            "Air temperature trend",
            f"{era5_trend.slope * 10:+.2f} °C / decade"
        )
        st.caption("ERA5-Land | 2001–2025")

with col2:
    with st.container(border=True):
        st.metric(
            "Surface temperature change",
            "+0.83 °C"
        )
        st.caption("2021–2025 vs. 2001–2005")

with col3:
    with st.container(border=True):
        st.metric(
            "Analysis period",
            "25 years"
        )
        st.caption("Summer months | 2001–2025")

st.caption(
    "The +0.83 °C surface change represents a spatial "
    "comparison of five-year Landsat composites, "
    "not the long-term regression trend."
)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📈 Temperature Trends",
    "🗺️ Heat Change Map",
    "📋 Methodology & Recommendations"
])



# --------------------------------------------------
# TAB 1: TEMPERATURE TRENDS
# --------------------------------------------------

with tab1:

    st.header("25 YEARS IN NUMBERS")

    st.write(
        "Compare annual summer surface temperatures "
        "with estimated air temperatures."
    )

    # Select the period displayed in both charts
    start_year, end_year = st.slider(
        "Select years to display",
        min_value=2001,
        max_value=2025,
        value=(2001, 2025),
        step=1
    )

    st.caption(
        "The selected period changes the chart view only. "
        "Trend estimates remain based on 2001–2025."
    )

    # LANDSAT - SURFACE TEMPERATURE
    fig_surface = create_chart(
        df_lst,
        "mean_lst_c",
        lst_trend,
        "Summer Surface Temperature — Landsat",
        is_landsat=True
    )

    fig_surface.update_xaxes(
        range=[start_year - 0.5, end_year + 0.5]
    )

    st.plotly_chart(
        fig_surface,
        width="stretch"
    )

    st.caption(
        f"Landsat trend: {lst_trend.slope * 10:+.2f} "
        f"°C per decade (p = {lst_trend.pvalue:.3f}). "
        "Not statistically significant."
    )

    st.divider()

    # ERA5-LAND - AIR TEMPERATURE
    fig_air = create_chart(
        df_era5,
        "air_temp_c",
        era5_trend,
        "Summer Air Temperature — ERA5-Land"
    )

    fig_air.update_xaxes(
        range=[start_year - 0.5, end_year + 0.5]
    )

    st.plotly_chart(
        fig_air,
        width="stretch"
    )

    st.caption(
        f"ERA5-Land trend: {era5_trend.slope * 10:+.2f} "
        f"°C per decade (p = {era5_trend.pvalue:.3f}). "
        "Statistically significant."
    )

    st.info(
        "Summer air temperatures show a statistically "
        "significant warming trend. The estimated "
        "surface temperature trend is positive, "
        "but remains statistically uncertain."
    )



    # --------------------------------------------------
    # ANNUAL DATA TABLES AND DOWNLOADS
    # --------------------------------------------------

    with st.expander("Explore & download annual temperature data"):

        c1, c2 = st.columns(2)

        # Display-friendly column names
        landsat_table = df_lst.rename(columns={
            "year": "Year",
            "mean_lst_c": "Surface temperature (°C)",
            "coverage_pct": "Spatial coverage (%)",
            "scene_count": "Satellite scenes"
        }).round({
            "Surface temperature (°C)": 2,
            "Spatial coverage (%)": 1
        })

        era5_table = df_era5.rename(columns={
            "year": "Year",
            "air_temp_c": "Air temperature (°C)",
            "day_count": "Days analysed"
        }).round({
            "Air temperature (°C)": 2
        })

        with c1:
            st.subheader("Landsat | Surface Temperature")

            st.dataframe(
                landsat_table,
                hide_index=True,
                width="stretch"
            )

            st.download_button(
                label="Download Landsat CSV",
                data=df_lst.to_csv(index=False).encode("utf-8"),
                file_name="brno_landsat_2001_2025.csv",
                mime="text/csv",
                key="download_landsat"
            )

        with c2:
            st.subheader("ERA5-Land | Air Temperature")

            st.dataframe(
                era5_table,
                hide_index=True,
                width="stretch"
            )

            st.download_button(
                label="Download ERA5-Land CSV",
                data=df_era5.to_csv(index=False).encode("utf-8"),
                file_name="brno_era5_land_2001_2025.csv",
                mime="text/csv",
                key="download_era5"
            )

        st.caption(
            "Landsat surface temperatures are derived from "
            "satellite observations. ERA5-Land air temperatures "
            "are derived from climate reanalysis."
        )


# --------------------------------------------------
# TAB 2: HEAT CHANGE MAP
# --------------------------------------------------

with tab2:

    st.header("WHERE HAS BRNO WARMED OR COOLED?")

    st.write(
        "Summer surface temperature change between "
        "2001–2005 and 2021–2025."
    )


    # Interactive zoomable map
    map_image = Image.open(
        ASSETS_DIR / "brno_heat_change.png"
    )

    fig_map = px.imshow(map_image)

    fig_map.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        paper_bgcolor="white"
    )

    fig_map.update_xaxes(visible=False)
    fig_map.update_yaxes(visible=False)
    fig_map.update_traces(hoverinfo="skip")

    st.plotly_chart(
        fig_map,
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False
        }
    )


    st.markdown(
        "**Key finding:** The city-wide mean surface "
        "temperature was approximately 0.83 °C "
        "higher in 2021–2025 than in 2001–2005. "
        "Changes vary substantially across Brno."
    )

    st.caption(
        "Source: Landsat Collection 2 Level-2 "
        "Surface Temperature, processed in "
        "Google Earth Engine."
    )


# --------------------------------------------------
# TAB 3: METHODOLOGY & RECOMMENDATIONS
# --------------------------------------------------

with tab3:

    st.header("METHODOLOGY")

    st.markdown("""
    **Study period:** 2001–2025, June–August.

    **Surface temperature:** Landsat Collection 2
    Level-2 thermal observations.

    **Air temperature:** ERA5-Land climate reanalysis.

    **Processing:** Google Earth Engine and Python.

    **Trend estimation:** Ordinary least squares
    linear regression.

    **Quality control:** Landsat years with less than
    90% spatial coverage were excluded from the
    trend calculation.
    """)

    st.header("KEY LIMITATIONS")

    st.markdown("""
    - Cloud cover reduces valid satellite observations.
    - Acquisition times and weather conditions vary.
    - Different Landsat sensors introduce uncertainty.
    - Surface temperature and air temperature
      represent different physical quantities.
    - ERA5-Land has limited spatial detail and
      cannot resolve individual urban hotspots.
    """)

    st.header("RECOMMENDATIONS FOR BRNO")

    st.markdown("""
    **1. Investigate priority locations**

    Verify areas showing stronger surface warming
    against vegetation, land cover and local conditions.

    **2. Support cooling measures**

    Consider vegetation, shading, reflective surfaces
    and energy-efficient cooling.

    **3. Monitor long-term changes**

    Repeat observations and evaluate whether urban
    interventions reduce local heat exposure.
    """)

st.divider()

st.caption(
    "Brno Urban Heat Assessment | "
    "Landsat & ERA5-Land | 2001–2025"
)
