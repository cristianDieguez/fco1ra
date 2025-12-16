import streamlit as st

def render(df_1,df_2):
    """Contenido para la sección 'Análisis de Temporada'.
    Recibe el DataFrame filtrado ya según los selects del sidebar.
    """
    st.markdown("""
        <div class='glass-card' style='padding:14px 16px; margin-bottom:12px;'>
            <h2 style='color:#fff;margin:0;'>Análisis de Temporada Sergio</h2>
        </div>
    """, unsafe_allow_html=True)

    # Placeholder de contenido (reemplazar por KPIs/Gráficos de temporada)
    st.dataframe(df_1, use_container_width=True)
    st.dataframe(df_2, use_container_width=True)
