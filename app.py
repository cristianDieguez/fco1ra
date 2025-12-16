# === Part 1: Setup and Login ===
import streamlit as st
import pandas as pd
from PIL import Image
import base64
from pages import analisis_individual, analisis_partido, analisis_temporada



def map_players_from_excel(botonera_df, torneo_excel) -> pd.DataFrame:
    """
    Reads the roster Excel file and maps:
      - Nombre
      - Posicion1 (only)

    into the Botonera DataFrame which must contain a 'Jugadores' int column.
    """

    df_map = torneo_excel

    # Fix column names (first row is titles)
    df_map.columns = ["Numero", "Nombre", "Posicion1", "Posicion2", "Posicion3"]

    # Remove duplicated header row inside data
    df_map = df_map[df_map["Numero"] != "Numero"]

    # Convert Numero to int
    df_map["Numero"] = df_map["Numero"].astype(int)

    # Build mapping dictionaries
    name_map = dict(zip(df_map["Numero"], df_map["Nombre"]))
    pos1_map = dict(zip(df_map["Numero"], df_map["Posicion1"]))

    # Map to botonera df
    botonera_df["Nombre"] = botonera_df["Jugadores"].map(name_map)
    botonera_df["Posicion"] = botonera_df["Jugadores"].map(pos1_map)

    return botonera_df

USERS = {
    "admin": 'admin'
}

def login():
    # Set background image for login using base64-embedded CSS
    with open("img/fondo.jpg", "rb") as f:
        _bg_bytes = f.read()
    _bg_base64 = base64.b64encode(_bg_bytes).decode()
    # Read logo for inline display
    try:
        with open("img/logo.png", "rb") as lf:
            _logo_b64 = base64.b64encode(lf.read()).decode()
    except FileNotFoundError:
        _logo_b64 = None

    _css = """
        <style>
        .stApp {
            background-image: url('data:image/jpeg;base64,REPLACE_BG');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Hide sidebar and its toggle while on login */
        section[data-testid="stSidebar"] { display: none !important; }
        div[data-testid="collapsedControl"] { display: none !important; }

        /* Space from top to vertically balance the card */
        section.main > div {
            padding-top: 15vh; /* more vertical centering */
            padding-bottom: 12vh;
            padding-left: 1rem;  /* compensate hidden sidebar */
            padding-right: 1rem;
        }

        /* Card styling for the login form */
        div[data-testid="stForm"] {
            background: rgba(0, 0, 0, 0.55); /* thicker/less transparent */
            border: 2px solid rgba(255, 255, 255, 0.25); /* thicker border */
            box-shadow: 0 16px 40px rgba(0,0,0,0.55);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 18px;
            padding: 32px 26px 26px 26px; /* thicker padding */
            max-width: 700px; /* wider card per request */
            margin: 0 auto; /* center horizontally */
        }

        /* Typography and widgets inside the box */
        div[data-testid="stForm"] h2 {
            color: #ffffff;
            margin-top: 6px;
            margin-bottom: 18px;
            font-weight: 700;
        }

        /* Labels */
        div[data-testid="stForm"] label {
            color: #e9eef1;
            font-weight: 600;
        }

        /* Inputs (stronger selectors to override Streamlit theme) */
        /* Container that BaseWeb uses around the input */
        div[data-testid="stForm"] div[data-baseweb="input"] {
            background: rgba(255,255,255,0.6) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0,0,0,0.18) !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.08) !important;
        }
        /* Some themes add an inner wrapper, ensure it's transparent */
        div[data-testid="stForm"] div[data-baseweb="input"] > div:first-child {
            background: transparent !important;
        }
        div[data-testid="stForm"] .stTextInput input,
        div[data-testid="stForm"] div[data-baseweb="input"] input,
        div[data-testid="stForm"] input[type="password"] {
            background: rgba(255,255,255,0.6) !important;
            color: #102015 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0,0,0,0.18) !important;
            height: 44px !important;
            padding: 6px 12px !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.08) !important;
        }
        div[data-testid="stForm"] .stTextInput input::placeholder,
        div[data-testid="stForm"] div[data-baseweb="input"] input::placeholder {
            color: rgba(16,32,21,0.65) !important;
        }
        div[data-testid="stForm"] .stTextInput input:focus,
        div[data-testid="stForm"] div[data-baseweb="input"] input:focus,
        div[data-testid="stForm"] input[type="password"]:focus {
            outline: none !important;
            border-color: rgba(11,61,46,0.35) !important;
            box-shadow: 0 0 0 3px rgba(168,230,207,0.35) !important;
            background: rgba(255,255,255,0.82) !important;
        }

        /* Button full width and style */
        div[data-testid="stForm"] .stButton > button {
            width: 60%; /* centered narrower button */
            border-radius: 12px;
            height: 46px;
            font-weight: 700;
            background: #A8E6CF; /* pastel mint green */
            color: #0b3d2e; /* deep green text for contrast */
            border: 1px solid rgba(11,61,46,0.28);
            display: block;
            margin: 10px auto 0; /* center button */
        }
        div[data-testid="stForm"] .stButton > button:hover {
            background: #93dcbf;
            border-color: rgba(11,61,46,0.38);
        }
        
        /* Responsive: keep card readable on small screens */
        @media (max-width: 640px) {
            div[data-testid="stForm"] { max-width: 92%; }
            div[data-testid="stForm"] .stButton > button { width: 100%; }
        }
        </style>
    """
    st.markdown(_css.replace("REPLACE_BG", _bg_base64), unsafe_allow_html=True)
    # Centered column layout with a styled translucent card
    col_left, col_center, col_right = st.columns([0.5, 2.2, 0.5])
    with col_center:
        with st.form("login_form"):
            if _logo_b64:
                st.markdown(
                    f"""
                    <div style='text-align:center; margin-bottom:10px;'>
                        <img src='data:image/png;base64,{_logo_b64}' alt='logo' style='height:110px;'>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("<h2 style='text-align:center;'>Log In</h2>", unsafe_allow_html=True)
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            login_clicked = st.form_submit_button("Ingresar")

    if login_clicked:
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Credenciales inválidas.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    login()
    st.stop()

# Page config
st.set_page_config(page_title="Análisis Pre-Partido", layout="wide")

# Background styling for the authenticated app (match login aesthetic)
with open("img/fondo.jpg", "rb") as _f_main:
    _bg_main_b64 = base64.b64encode(_f_main.read()).decode()

_app_css = """
    <style>
    .stApp {
        background-image: url('data:image/jpeg;base64,REPLACE_BG_MAIN');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Tighter top spacing now that we're in the app */
    section.main > div.block-container { padding-top: 18px; }

    /* Sidebar translucent glass */
    section[data-testid="stSidebar"] > div {
        background: rgba(0,0,0,100.0);
        border-right: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    /* Hide Streamlit's default pages navigation (we use our own buttons) */
    section[data-testid="stSidebar"] nav,
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"],
    nav[data-testid="stSidebarNav"],
    nav[aria-label*="nav" i] {
        display: none !important;
    }

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label { color: #e9eef1; }

    /* Inputs and multiselects globally */
    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        background: rgba(255,255,255,0.65) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0,0,0,0.48) !important;
    }
    div[data-baseweb="input"] > div:first-child, 
    div[data-baseweb="select"] > div:first-child { background: transparent !important; }
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] input { background: transparent !important; color: #102015 !important; }

    /* Dataframe wrapper glass effect */
    div[data-testid="stDataFrame"] { 
        background: rgba(0,0,0,0.35); 
        border-radius: 12px; 
        padding: 8px; 
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        max-height: 72vh;                 /* keep table inside viewport */
        max-width: 100%;                  /* do not overflow horizontally */
        overflow-x: auto;                 /* horizontal scroll if needed */
        overflow-y: auto;                 /* vertical scroll if needed */
    }
    /* Dataframe inner grid styling (headers, cells, scrollbars) */
    div[data-testid="stDataFrame"] div[role="grid"] {
        background: rgba(0,0,0,0.18) !important;
        border-radius: 8px;
        width: max-content;               /* shrink to content, allow horizontal scroll */
        min-width: 100%;                  /* but at least fill container */
    }
    div[data-testid="stDataFrame"] div[role="columnheader"] {
        background: rgba(0,0,0,0.45) !important;
        color: #e9eef1 !important;
        border-bottom: 1px solid rgba(255,255,255,0.12) !important;
    }
    div[data-testid="stDataFrame"] div[role="row"] div[role="gridcell"] {
        background: rgba(0,0,0,0.25) !important;
        color: #eef6f0 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    }
    /* Zebra striping */
    div[data-testid="stDataFrame"] div[role="row"]:nth-child(even) div[role="gridcell"] {
        background: rgba(0,0,0,0.30) !important;
    }
    div[data-testid="stDataFrame"] div[role="row"]:nth-child(odd) div[role="gridcell"] {
        background: rgba(0,0,0,0.22) !important;
    }
    div[data-testid="stDataFrame"] div[role="row"] div[role="gridcell"]:hover {
        background: rgba(168,230,207,0.16) !important; /* mint hover */
    }
    /* Scrollbar styling */
    div[data-testid="stDataFrame"] ::-webkit-scrollbar { height: 10px; width: 10px; }
    div[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.28);
        border-radius: 8px;
    }

    /* Global button style to match login */
    div.stButton > button {
        background: #A8E6CF; /* pastel mint */
        color: #0b3d2e;
        border: 1px solid rgba(11,61,46,0.28);
        border-radius: 10px;
        height: 42px;
        font-weight: 700;
    }
    div.stButton > button:hover {
        background: #93dcbf;
        border-color: rgba(11,61,46,0.38);
    }

    /* Force filter widgets (class st-ae) to white background with black text */
    .st-ae { 
        background-color: white !important; 
        color: black !important; 
    }
    section[data-testid="stSidebar"] .st-ae { 
        
        color: black !important; 
    }

    /* Sidebar navigation buttons */
    section[data-testid="stSidebar"] div.stButton { 
        margin: 0 !important;               /* remove gaps between buttons */
    }
    section[data-testid="stSidebar"] div.stButton > button { 
        width: 100%;
        margin: 0 !important;               /* no vertical spacing */
        border-radius: 0 !important;        /* square corners */
        border: none;                        /* we'll draw separators manually */
        border-bottom: 1px solid rgba(255,255,255,0.28);
    }
    section[data-testid="stSidebar"] div.stButton:first-of-type > button {
        border-top: 1px solid rgba(255,255,255,0.28);
    }
    section[data-testid="stSidebar"] div.stButton:last-of-type > button {
        border-bottom: 1px solid rgba(255,255,255,0.28);
    }
    section[data-testid="stSidebar"] div.stButton > button:disabled {
        background: #84d991;                 /* active state */
        border-color: rgba(11,61,46,0.50);
        color: #073a2a;
        opacity: 1; /* keep readable */
        cursor: default;
    }

    /* Glass hero card for page title */
    #main-hero.glass-card {
        background: rgba(0,0,0,0.40);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 16px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.45);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        padding: 14px 18px;
        margin: 8px auto 18px auto;
        max-width: 1100px;
        text-align: center;
    }
    #main-hero.glass-card h1 { color: #ffffff; margin: 0; }
    </style>
"""
st.markdown(_app_css.replace("REPLACE_BG_MAIN", _bg_main_b64), unsafe_allow_html=True)

# --- Page content ---
_section_title_map = {
    "Análisis Individual": "Análisis Individual",
    "Análisis de Partido": "Análisis de Partido",
    "Análisis de Temporada": "Análisis de Temporada",
}
_current_title = _section_title_map.get(st.session_state.get("section", "Análisis Individual"), "Análisis")
st.markdown(
    f"""
    <div id='main-hero' class='glass-card'>
        <h1>{_current_title}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data

def load_data():
    df_botonera1 = pd.read_parquet("datasets/botonera_1.parquet")
    df_botonera2 = pd.read_parquet("datasets/botonera_2.parquet")
    df_botonera3 = pd.read_parquet("datasets/botonera_3.parquet")
    df_botonera4 = pd.read_parquet("datasets/botonera_4.parquet")
    df_tiempos = pd.read_parquet("datasets/df_tiempo_efectivo.parquet")
    return df_botonera1, df_botonera2, df_botonera3, df_botonera4, df_tiempos

df_1, df_2, df_3, df_4, df_tiempos = load_data()

TOURNAMENT_DATA = pd.read_excel("datasets/Resultados_partidos.xlsx", sheet_name="PLANTEL")

df_1 = map_players_from_excel(df_1, TOURNAMENT_DATA)
df_2 = map_players_from_excel(df_2, TOURNAMENT_DATA)
df_3 = map_players_from_excel(df_3, TOURNAMENT_DATA)
df_tiempos = map_players_from_excel(df_tiempos, TOURNAMENT_DATA)



""" Sidebar: Logo pequeño y navegación principal """
if "section" not in st.session_state:
    st.session_state["section"] = "Análisis Individual"

with st.sidebar:
    # Logo reducido y centrado para dejar espacio a opciones
    try:
        logo_sidebar_small = Image.open("img/logo.png")
        _ls_c1, _ls_c2, _ls_c3 = st.columns([1,2,1])
        with _ls_c2:
            st.image(logo_sidebar_small, width=96)
    except Exception:
        st.markdown("<h2 style='text-align:center;color:#e9eef1;'>Navegación</h2>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#e9eef1;margin-bottom:6px;'>Navegación</h4>", unsafe_allow_html=True)
    current = st.session_state.get("section", "Análisis Individual")
    btn_ind = st.button("Análisis Individual", use_container_width=True, disabled=(current=="Análisis Individual"), key="nav_ind")
    btn_part = st.button("Análisis de Partido", use_container_width=True, disabled=(current=="Análisis de Partido"), key="nav_part")
    btn_temp = st.button("Análisis de Temporada", use_container_width=True, disabled=(current=="Análisis de Temporada"), key="nav_temp")
    if btn_ind:
        st.session_state["section"] = "Análisis Individual"
        st.rerun()
    if btn_part:
        st.session_state["section"] = "Análisis de Partido"
        st.rerun()
    if btn_temp:
        st.session_state["section"] = "Análisis de Temporada"
        st.rerun()





    
_section = st.session_state.get("section", "Análisis Individual")
if _section == "Análisis Individual":
    analisis_individual.render(df_1, df_2, df_3, df_tiempos)
    

elif _section == "Análisis de Partido":
    analisis_partido.render(df_1,df_2, df_tiempos, df_4)
elif _section == "Análisis de Temporada":
    analisis_temporada.render(df_1,df_2)
else:
    # Fallback
    analisis_individual.render(df_1,df_2)

