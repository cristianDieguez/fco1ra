import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Arc, Rectangle, Circle as MplCircle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import os
from PIL import Image
from matplotlib.patches import FancyBboxPatch
from mplsoccer import PyPizza, add_image

FERRO_NAME = "Ferro"  # change here if the name ever changes

ACTION_STYLE = {
    "shots": {
        "line_color": "green",
        "line_alpha": 0.8,
        "goal_icon": "ball2.png",
        "save_icon": "gloves2.png",
        "blocked_symbol": "■",
        "blocked_color": "#7a7a7a",
        "icon_size": 0.025,
    },

    "loss": {
        "symbol": "×",
        "color": "red",
        "size": 8,
    },

    "1v1_off": {
        "symbol": "▲",
        "color": "green",
        "size": 11,
    },

    "key_pass": {
        "correct_color": "green",
        "wrong_color": "red",
        "arrow_width": 1.5,
        "head_size": 10,
    },

    "foul_received": {
        "color": "green",
        "radius": 0.008,
    },

    # ---------------- DEFENSIVE ---------------- #

    "recovery": {
        "symbol": "✦",
        "color": "blue",
        "size": 15,
    },

    "pressure": {
        "symbol": "P",
        "color": "blue",
        "size": 11,
    },

    "1v1_def": {
        "size": 0.01,
        "win_color": "green",
        "lose_color": "red",
    },

    "foul_committed": {
        "color": "red",
        "radius": 0.008,
    },
}


# -----------------------------
# CAMPOGRAMA Y SHOTMAP SETTINGS
# -----------------------------

TEAM_COLORS = {
    "Ferro": "#00b050",   # green
    "Rival": "#ff8c00"    # orange
}

# Colors (optional)
RESULT_COLORS = {
    "Desviado": "red",
    "Atajado": "blue",
    "Bloqueado": "orange",
    "Gol": "green",
    "Ganado": "green",
    "Perdido": "red",
    "Correcto": "green",
    "Fallado": "red",
    "Forzada": "blue",
    "No Forzada": "yellow",
    "En Salida": "green",
    "Conducción Lat.": "blue",
    "Espalda": "yellow",
    "Mal Posicionado": "purple",
    "Tras Recuperación": "orange",


    None: "gray"
}



RESULT_MARKERS = {
    "Desviado": "x",
    "Atajado": "s",
    "Bloqueado": "o",
    "Gol": "*",
    None: "."
}

ANCHO, ALTO = 1.0, 1.0
N_COLS, N_ROWS = 3, 3

def preprocess_transiciones(df, modo="Ofensiva"):
    """
    Collapse player-level rows into EVENT-level transitions.
    Works for Ofensiva and Defensiva.

    modo: "Ofensiva" | "Defensiva"
    """

    if modo == "Ofensiva":
        action_col = "Tipo_de_Accion"
        PRIORITY = ["Gol", "Tiro", "Pelota Parada", "Recuperacion", "Perdida", None]
    else:
        action_col = "Tipo_de_Accion"
        PRIORITY = ['Pelota Parada', 'Repliegue', 'Tiro', 'Gol', 'Recuperacion', None,
       'Perdida']

    def resolve_tipo(x):
        for p in PRIORITY:
            if p in x.values:
                return p
        return None

    grouped = (
        df.groupby(["Jornada", "Evento"])
        .agg(
            Situacion=("Situacion", "first"),
            Resultado=(action_col, resolve_tipo),
            Duración=(
                "Duración",
                lambda x: (
                    "Larga" if "Larga (+3)" in x.values
                    else "Corta" if "Corta (H 3)" in x.values
                    else None
                )
            )
        )
        .reset_index()
    )

    return grouped

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


def plot_transiciones(df_events, df_raw, modo="Ofensiva"):
    """
    Shared visualization for Transiciones Ofensivas / Defensivas
    """

    situaciones = ["1 vs 1", "2 vs 1", "3 vs 1", "3 vs 2"]

    if modo == "Ofensiva":
        ACTION_COLORS = {
            "Gol": "#2ecc71",
            "Tiro": "#27ae60",
            "Pelota Parada": "#f1c40f",
            "Recuperacion": "#3498db",
            "Perdida": "#e74c3c",
            None: "#7f8c8d"
        }
    else:
        ACTION_COLORS = {
            "Gol": "#2ecc71",           # Green → success
            "Tiro": "#27ae60",          # Dark green → attempt
            "Pelota Parada": "#f1c40f", # Yellow → restart / set piece
            "Recuperacion": "#3498db",  # Blue → regain control
            "Repliegue": "#9b59b6",     # Purple → defensive organization
            "Perdida": "#e74c3c",       # Red → loss
            None: "#7f8c8d"             # Grey → undefined / no outcome
        }

    fig = plt.figure(figsize=(16, 10), facecolor="black")
    gs = GridSpec(
        len(situaciones),
        4,
        width_ratios=[1.2, 2, 2, 2],
        wspace=0.05
    )

    for i, sit in enumerate(situaciones):
        df_sit = df_events[df_events["Situacion"] == sit]

        # ================= AX 1 — LABEL =================
        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.axis("off")
        ax_label.text(
            0.95, 0.5, sit,
            color="white", fontsize=14, fontweight="bold",
            ha="right", va="center"
        )

        # ================= AX 2 — DONUT =================
        ax_donut = fig.add_subplot(gs[i, 1])
        ax_donut.set_facecolor("black")

        counts = df_sit["Resultado"].value_counts(dropna=False)
        values = counts.values
        labels = counts.index.tolist()
        colors = [ACTION_COLORS[l] for l in labels]

        if len(values) > 0:
            ax_donut.pie(
                values,
                colors=colors,
                startangle=90,
                wedgeprops=dict(width=0.35, edgecolor="black")
            )
            ax_donut.text(
                0, 0, str(len(df_sit)),
                color="white", fontsize=14, fontweight="bold",
                ha="center", va="center"
            )

        ax_donut.axis("off")
        ax_donut.set_aspect("equal")

        # ================= AX 3 — DURACIÓN =================
        ax_bar = fig.add_subplot(gs[i, 2])
        ax_bar.set_facecolor("black")

        df_dur = df_sit[df_sit["Duración"].notna()]
        total = len(df_dur)

        corta = (df_dur["Duración"] == "Corta").sum()
        larga = (df_dur["Duración"] == "Larga").sum()

        pct_corta = (corta / total * 100) if total > 0 else 0
        pct_larga = (larga / total * 100) if total > 0 else 0

        ax_bar.barh([0], [pct_corta], color="#9b59b6", height=0.2)
        ax_bar.barh([0], [pct_larga], left=pct_corta, color="#2c2c2c", height=0.2)

        ax_bar.text(0, -0.6, f"Corta: {pct_corta:.0f}%", color="white", ha="left", fontsize=10)
        ax_bar.text(100, -0.6, f"Larga: {pct_larga:.0f}%", color="white", ha="right", fontsize=10)

        ax_bar.set_xlim(0, 100)
        ax_bar.set_ylim(-1, 1)
        ax_bar.axis("off")

        # ================= AX 4 — TOP PLAYERS =================
        ax_players = fig.add_subplot(gs[i, 3])
        ax_players.axis("off")

        df_raw_sit = df_raw[
            df_raw["Evento"].isin(df_sit["Evento"]) &
            (df_raw["Situacion"] == sit)
        ]

        top_players = df_raw_sit["Nombre"].value_counts().head(3)

        for idx, (player, count) in enumerate(top_players.items()):
            ax_players.text(
                0.25, 0.75 - idx * 0.25,
                f"{idx + 1}. {player} ({count})",
                color="white", fontsize=11
            )

    # ================= LEGEND =================
    legend_elements = [
        Patch(facecolor=color, label=label)
        for label, color in ACTION_COLORS.items()
        if label is not None
    ]

    legend = fig.legend(
        handles=legend_elements,
        title="Resultado",
        fontsize=11,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03)
    )

    # --- Force white text ---
    legend.get_title().set_color("white")
    for text in legend.get_texts():
        text.set_color("white")

    return fig

def draw_futsal_pitch_grid(ax):

    dx, dy = ANCHO / N_COLS, ALTO / N_ROWS
    ax.set_facecolor("white")

    # ===== OUTER LINES =====
    ax.plot([0, ANCHO], [0, 0], color="grey")
    ax.plot([0, ANCHO], [ALTO, ALTO], color="grey")
    ax.plot([0, 0], [0, ALTO], color="grey")
    ax.plot([ANCHO, ANCHO], [0, ALTO], color="grey")

    # ===== CENTER LINE =====
    ax.plot([ANCHO/2, ANCHO/2], [0, ALTO], color="grey")

    # =====================================================
    #          IMPORTANT: NORMALIZATION FOR ARCS
    # =====================================================
    # Your original values:
    # Arc left: center (0, ALTO/2), width=8, height=12
    # Arc right: center (ANCHO, ALTO/2), width=8, height=12
    #
    # These are “drawing units”, so we convert them to:
    # width_norm  = 8 / ANCHO
    # height_norm = 12 / ALTO
    #
    # This keeps your exact proportions, no changes.
    # =====================================================

    width_norm  = 10 / (40)   # Normalize relative to full length
    height_norm = 12 / (20)  # Normalize relative to full width

    # Put into ANCHO × ALTO space
    width_scaled  = width_norm * ANCHO * 40   # revert to your original styling
    height_scaled = height_norm * ALTO * 20   # revert to your original styling

    # ===== PENALTY ARCS (YOUR ORIGINAL SHAPE, NOW NORMALIZED) =====
    ax.add_patch(Arc((0, ALTO/2),
                     width=width_norm,
                     height=height_norm,
                     angle=0, theta1=270, theta2=90,
                     color="grey"))

    ax.add_patch(Arc((ANCHO, ALTO/2),
                     width=width_norm,
                     height=height_norm,
                     angle=0, theta1=90, theta2=270,
                     color="grey"))

    # ===== CENTER CIRCLE (KEEP YOUR EXACT ORIGINAL SIZE) =====
    ax.add_patch(MplCircle((ANCHO/2, ALTO/2),
                           radius=4 / 40,     # normalized radius
                           color="grey",
                           fill=False))

    ax.add_patch(MplCircle((ANCHO/2, ALTO/2),
                           radius=0.2 / 40,   # same logic
                           color="grey"))

    # ===== GRID 3×3 (UNCHANGED) =====
    for j in range(N_ROWS):
        for i in range(N_COLS):
            x0, y0 = i * dx, j * dy
            ax.add_patch(Rectangle((x0, y0), dx, dy,
                                   linewidth=0.6,
                                   edgecolor='darkgrey',
                                   facecolor='none'))
            zona = j * N_COLS + i + 1
            ax.text(x0 + dx - 0.03,
                    y0 + dy - 0.03,
                    str(zona),
                    ha='right', va='top',
                    fontsize=9, color='darkgrey')

    ax.set_xlim(0, ANCHO)
    ax.set_ylim(0, ALTO)
    ax.axis('off')

    return ax

def add_team_shot_tag(df):
    df = df.copy()
    df["TeamShot"] = df["Acción"].apply(
        lambda x: "Ferro" if x == "Tiro Ferro"
        else "Rival" if x == "Tiro Rival"
        else None
    )
    return df

def render_futsal_shotmap(df, local, visitante):
    df = add_team_shot_tag(df)
    shots = df[df["TeamShot"].notna()]

    fig, ax = plt.subplots(figsize=(6,6))

    draw_futsal_pitch_grid(ax)
    plot_team_shots(shots, ax)

    fig.patch.set_facecolor("#062e16")
    ax.set_title(f"Mapa de Tiros ({local} vs {visitante})", color="white", fontsize=14)

    # -----------------------------------------
    # NEW LEGEND (Compact and outside)
    # -----------------------------------------
    markers = list({r:m for r,m in RESULT_MARKERS.items() if r is not None}.items())

    legend_fig = []
    for result, marker in markers:
        legend_fig.append(plt.Line2D([0],[0],
                        marker=marker, color="white",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        linestyle="None", markersize=10,
                        label=result))

    team_patches = [
        plt.Line2D([0],[0], marker="s", color=TEAM_COLORS["Ferro"], markersize=12, label="Ferro"),
        plt.Line2D([0],[0], marker="s", color=TEAM_COLORS["Rival"], markersize=12, label="Rival")
    ]

    legend = ax.legend(
        handles=team_patches + legend_fig,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
        fontsize=9
    )

    # 🔥 Make legend text white
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    return fig

def plot_team_shots(df, ax, size=150, alpha=0.85):
    df = df[df["TeamShot"].notna()]

    # Draw scatter points (same as earlier)
    for (team,res), group in df.groupby(["TeamShot","Resultado"]):
        ax.scatter(
            group["FieldXfrom"],
            group["FieldYfrom"],
            s=size,
            alpha=alpha,
            color=TEAM_COLORS.get(team,"gray"),
            marker=RESULT_MARKERS.get(res,"o"),
            edgecolors="black",
            linewidths=0.9
        )

    return ax

def plot_donut(ax, results, colors, logo_path=None, team_color=None):
    """Draw one donut chart with % inside + logo center"""

    wedges, texts, autotexts = ax.pie(
        results.values,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        pctdistance=0.75,
        colors=colors,
        textprops={'color': 'white', 'fontsize': 11}
    )

    # cut center out → donut
    centre = plt.Circle((0,0), 0.50, fc='#062e16')
    ax.add_artist(centre)

    # add logo inside
    if logo_path:
        logo = Image.open(logo_path)
        ax_img = ax.inset_axes([0.38, 0.38, 0.24, 0.24])  # centered
        ax_img.imshow(logo)
        ax_img.axis("off")

    ax.set_aspect('equal')
    ax.axis("off")

    # External text = Resultado + (count)
    angle = 0
    for i, w in enumerate(wedges):
        ang = (w.theta2 - w.theta1)/2 + w.theta1
        x = math.cos(math.radians(ang))*1.25
        y = math.sin(math.radians(ang))*1.25
        label = f"{results.index[i]} ({results.values[i]})"  # <--- key upgrade!
        ax.text(x, y, label, ha='center', va='center',
                color='white', fontsize=10, fontweight="bold")

    return ax

def plot_shot_result_pies(df, local_team, visit_team):
    """Two stacked sexy donuts with logos"""

    ferro = df[df["Acción"]=="Tiro Ferro"]["Resultado"]
    rival = df[df["Acción"]=="Tiro Rival"]["Resultado"]

    ferro_counts = ferro.value_counts()
    rival_counts = rival.value_counts()

    ferro_colors = [RESULT_COLORS.get(r,"gray") for r in ferro_counts.index]
    rival_colors = [RESULT_COLORS.get(r,"gray") for r in rival_counts.index]

    local_logo = find_logo(local_team)
    visitante_logo = find_logo(visit_team)

    fig, ax = plt.subplots(2,1, figsize=(6,10), facecolor="#062e16")

    if df["Local"].iloc[0] == FERRO_NAME:
        # Ferro is local
        plot_donut(ax[0], ferro_counts, ferro_colors, local_logo)
        plot_donut(ax[1], rival_counts, rival_colors, visitante_logo)
    else:
        # Rival is local
        plot_donut(ax[0], rival_counts, rival_colors, local_logo)
        plot_donut(ax[1], ferro_counts, ferro_colors, visitante_logo)


    plt.tight_layout()
    return fig

# -----------------------------
# 1. STATS FROM BOTONERA_1
# -----------------------------
def compute_match_stats_b1(df: pd.DataFrame) -> dict:
    """
    Returns FERRO vs RIVAL stats from botonera_1 for a single match dataframe.
    All values are from Ferro's perspective.
    """

    # Tiros
    tiros_ferro = (df["Acción"] == "Tiro Ferro").sum()
    tiros_rival = (df["Acción"] == "Tiro Rival").sum()

    # Tiros al arco (Atajado o Gol)
    tiros_arco_ferro = (
        (df["Acción"] == "Tiro Ferro")
        & (df["Resultado"].isin(["Atajado", "Gol"]))
    ).sum()
    tiros_arco_rival = (
        (df["Acción"] == "Tiro Rival")
        & (df["Resultado"].isin(["Atajado", "Gol"]))
    ).sum()

    # Precisión
    prec_ferro = round((tiros_arco_ferro / tiros_ferro) * 100, 1) if tiros_ferro > 0 else 0.0
    prec_rival = round((tiros_arco_rival / tiros_rival) * 100, 1) if tiros_rival > 0 else 0.0

    # Recuperaciones (nuestras) / Pérdidas (del rival)
    rec_ferro = (df["Acción"] == "Recuperación").sum()
    rec_rival = (df["Acción"] == "Pérdida").sum()

    # Faltas
    faltas_ferro = (df["Acción"] == "Falta Cometida").sum()
    faltas_rival = (df["Acción"] == "Falta Recibida").sum()

    # Sanciones
    sanc_ferro = (df["Acción"] == "Sanción Ferro").sum()
    sanc_rival = (df["Acción"] == "Sanción Rival").sum()

    return {
        "Tiros": (int(tiros_ferro), int(tiros_rival)),
        "Tiros al arco": (int(tiros_arco_ferro), int(tiros_arco_rival)),
        "Precisión (%)": (prec_ferro, prec_rival),
        "Recuperaciones": (int(rec_ferro), int(rec_rival)),
        "Faltas": (int(faltas_ferro), int(faltas_rival)),
        "Sanciones": (int(sanc_ferro), int(sanc_rival)),
    }

def get_score_ferro_rival(df: pd.DataFrame) -> tuple[int, int]:
    """
    Final score Ferro vs Rival from botonera_1.
    """
    goles_ferro = (
        (df["Acción"] == "Tiro Ferro") & (df["Resultado"] == "Gol")
    ).sum()
    goles_rival = (
        (df["Acción"] == "Tiro Rival") & (df["Resultado"] == "Gol")
    ).sum()
    return int(goles_ferro), int(goles_rival)

# -----------------------------
# 2. TRANSICIONES FROM BOTONERA_2
# -----------------------------
def compute_transitions(df: pd.DataFrame) -> tuple[int, int]:
    """
    From botonera_2, compute Offensives for Ferro and Offensives for Rival
    (your 'Defensivo' definition).
    """
    df_t = df[df["Acción"].isin(["Transición Ofensiva", "Transición Defensiva"])]
    
    # Ofensivas Ferro: Resultado == Positivo
    ofensivas_ferro = df_t[(df_t["Acción"] == "Transición Ofensiva") & (df_t["Tipo_de_Accion"] != "Recuperacion")
        ].shape[0]

    # "Defensivo": Ofensivas Rival -> contamos nuestras Transición Defensiva con cualquier Resultado
    ofensivas_rival = df_t[(df_t["Acción"] == "Transición Defensiva") & (df_t["Tipo_de_Accion"] != "Recuperacion")
        ].shape[0]

    return int(ofensivas_ferro), int(ofensivas_rival)

# -----------------------------
# 3. MATCH / RIVAL HANDLING
# -----------------------------
def add_rival_column(df: pd.DataFrame, ferro_name: str = FERRO_NAME) -> pd.DataFrame:
    """
    Adds a 'Rival' column based on Local / Visitante:
    Rival = Local if Local != Ferro, else Visitante.
    """
    df = df.copy()

    def _rival(row):
        if row["Local"] == ferro_name:
            return row["Visitante"]
        elif row["Visitante"] == ferro_name:
            return row["Local"]
        else:
            # fallback, in theory Ferro is always one of the two
            return row["Visitante"]

    df["Rival"] = df.apply(_rival, axis=1)
    return df

def build_stats_df(
    stats_dict: dict,
    trans_ferro: int,
    trans_rival: int,
    ferro_is_local: bool,
) -> pd.DataFrame:
    """
    Transform FERRO vs RIVAL stats into Local vs Visitante rows for plotting.
    """
    rows = []

    for stat_name, (val_ferro, val_rival) in stats_dict.items():
        if ferro_is_local:
            local_raw, visit_raw = val_ferro, val_rival
        else:
            local_raw, visit_raw = val_rival, val_ferro

        if "Precisión" in stat_name:
            local_label = f"{local_raw:.1f}%"
            visit_label = f"{visit_raw:.1f}%"
        else:
            local_label = str(local_raw)
            visit_label = str(visit_raw)

        rows.append(
            {
                "Stat": stat_name,
                "Local": float(local_raw),
                "Visitante": float(visit_raw),
                "Local_label": local_label,
                "Visitante_label": visit_label,
            }
        )

    # Transiciones (Ofensivas Ferro vs Ofensivas Rival)
    if ferro_is_local:
        local_raw, visit_raw = trans_ferro, trans_rival
    else:
        local_raw, visit_raw = trans_rival, trans_ferro

    rows.append(
        {
            "Stat": "Transiciones Ofensivas",
            "Local": float(local_raw),
            "Visitante": float(visit_raw),
            "Local_label": str(local_raw),
            "Visitante_label": str(visit_raw),
        }
    )

    return pd.DataFrame(rows)

# -----------------------------
# 4. PLOTTING (NORMALIZED BARS)
# -----------------------------
def normalize_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_vals = df[["Local", "Visitante"]].max(axis=1)
    max_vals = max_vals.replace(0, 1.0)
    df["Local_norm"] = df["Local"] / max_vals
    df["Visitante_norm"] = df["Visitante"] / max_vals
    return df

def find_logo(team_name: str, folder="img/equipos"):
    """
    Searches img/equipos for a matching logo.
    If none found → returns DUMMY.png if exists.
    """

    team_key = team_name.lower().replace(" ", "").replace("-", "")
    dummy_path = os.path.join(folder, "DUMMY.png")

    if not os.path.exists(folder):
        return dummy_path if os.path.exists(dummy_path) else None

    # Check all files first (no early exit)
    for f in os.listdir(folder):
        name = f.lower().replace(".png", "").replace(" ", "").replace("-", "")
        if name == team_key and f.endswith(".png"):
            return os.path.join(folder, f)

    # If no match → return fallback dummy
    return dummy_path if os.path.exists(dummy_path) else None

def plot_stats(
    df: pd.DataFrame,
    local_name: str,
    visit_name: str,
    score_local: int,
    score_visit: int,
):
    df_n = normalize_stats(df)
    n = len(df_n)
    
    # Thinner bars → better visual
    bar_height = 0.55  
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Bars (reduced size & centered better)
    ax.barh(y, -df_n["Local_norm"], height=bar_height, color="#0d8f3d")
    ax.barh(y, df_n["Visitante_norm"], height=bar_height, color="#f7a20b")

    # Background
    fig.patch.set_facecolor("#0b5c2a")
    ax.set_facecolor("#0b5c2a")
    # ax.axvline(0, color="white", lw=1.8)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(-1.05, 1.05)
    ax.invert_yaxis()

    # 🟩 Limit label position inside area to avoid overflow
    for idx, row in df_n.iterrows():
        yy = y[idx]
        
        ax.text(-0.98, yy, row["Local_label"], ha="left", va="center",
                color="white", fontsize=11, fontweight="bold")
        
        ax.text(0.98, yy, row["Visitante_label"], ha="right", va="center",
                color="white", fontsize=11, fontweight="bold")
        
        ax.text(0, yy, row["Stat"], ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    # =============================
    # 🔥 Logos + Header Formatting
    # =============================

    # Load logos if available
    logo_left = find_logo(local_name)
    logo_right = find_logo(visit_name)

    # Title spacing improved
    fig.text(0.5, 0.97, f"{local_name.upper()} vs {visit_name.upper()}",
             ha="center", color="white", fontsize=26, fontweight="heavy")

    fig.text(0.5, 0.90, f"{score_local} - {score_visit}",
             ha="center", color="white", fontsize=22, fontweight="bold")

    fig.text(0.5, 0.85, "ESTADÍSTICAS",
             ha="center", color="white", fontsize=17, fontweight="bold")

    # Display logos
    if logo_left:
        imgL = Image.open(logo_left)
        ax_logoL = fig.add_axes([0.03, 0.90, 0.10, 0.10])
        ax_logoL.imshow(imgL); ax_logoL.axis("off")

    if logo_right:
        imgR = Image.open(logo_right)
        ax_logoR = fig.add_axes([0.87, 0.90, 0.10, 0.10])
        ax_logoR.imshow(imgR); ax_logoR.axis("off")

    # Frame margin to prevent crowding
    fig.subplots_adjust(top=0.78, bottom=0.05, left=0.18, right=0.82)

    return fig


# CONVERT SECONDS TO MM:SS
def format_mmss(seconds):
    if seconds is None or pd.isna(seconds):
        return None
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"

def compute_player_stats_for_pizza(df_events, df_minutes, df_2, player_name):
    """
    Computes ALL futsal KPIs for the Pizza Plot + MT OF/DEF absolute +
    MT OF/DEF positive + % positive in both phases.
    """

    df_p = df_events[df_events["Nombre"] == player_name]

    # ---------------- MINUTOS JUGADOS (MM:SS only for UI) ----------------
    df_m = df_minutes[df_minutes["Nombre"] == player_name]
    total_seconds = df_m["Tiempo_Efectivo"].sum() if len(df_m) > 0 else 0
    tiempo_MMSS = format_mmss(total_seconds)

    # ---------------- MT OFENSIVO / DEFENSIVO (ABSOLUTE SECONDS) ----------------
    df_mt = df_2[df_2["Nombre"] == player_name]

    mt_of = (df_mt["Acción"] == "MT Ofensivo").sum()
    mt_def = (df_mt["Acción"] == "MT Defensivo").sum()
    
    # Positives based on Resultado == "Positivo"
    mt_of_pos = ((df_mt["Acción"] == "MT Ofensivo") & (df_mt["Resultado"] == "Positivo")).sum()
    mt_def_pos = ((df_mt["Acción"] == "MT Defensivo") & (df_mt["Resultado"] == "Positivo")).sum()
    # ---------------- HELPERS ----------------
    def count(action):
        return df_p[df_p["Acción"] == action].shape[0]

    def count_result(action, results):
        df_tmp = df_p[df_p["Acción"] == action]
        return df_tmp[df_tmp["Resultado"].isin(results)].shape[0]

    # ---------------- OFENSIVO ----------------
    tiros = count("Tiro Ferro")
    tiros_arco = count_result("Tiro Ferro", ["Gol", "Atajado"])
    goles = df_p[df_p["Resultado"] == "Gol"].shape[0]
    pases_clave = count("Pase Clave")
    pases_clave_correctos = count_result("Pase Clave", ["Correcto"])
    asistencias = count("Asistencia")
    perdida = count("Pérdida")
    falta_recibida = count("Falta Recibida")

    # 1v1 ofensivo
    of_total = count("1 VS 1 Ofensivo")
    of_ganados = count_result("1 VS 1 Ofensivo", ["Ganado"])
    pct_of = (of_ganados / of_total * 100) if of_total > 0 else 0

    # ---------------- DEFENSIVO ----------------
    def_total = count("1 VS 1 Defensivo")
    def_ganados = count_result("1 VS 1 Defensivo", ["Ganado"])
    pct_def = (def_ganados / def_total * 100) if def_total > 0 else 0

    recuperaciones = count("Recuperación")
    presiones = count("Presión")
    sanciones = df_p[df_p["Sanción"] == "Ferro"].shape[0]

    # ---------------- POSITIVE ACTIONS % ----------------

    pct_ofensivo_positivo = (mt_of_pos / mt_of * 100) if mt_of > 0 else 0

    pct_defensivo_positivo = (mt_def_pos / mt_def * 100) if mt_def > 0 else 0

    # ---------------- OUTPUT ----------------
    stats_off_abs = {
        "Tiros": tiros,
        "Tiros al Arco": tiros_arco,
        "Goles": goles,
        "MT Ofensivo": mt_of,
        "% MT Of": round(pct_ofensivo_positivo, 2),
        "Pases Claves": pases_clave,
        "Asistencias": asistencias,
        "Pérdida": perdida,
        "Falta Recibida": falta_recibida,
        "1v1 Of Ganado %": round(pct_of, 2),

    }

    stats_def_abs = {
        "1v1 Def Total": def_total,
        "1v1 Def Ganado %": round(pct_def, 2),
        "Recuperaciones": recuperaciones,
        "Presiones": presiones,
        "Sanciones": sanciones,
        "MT Defensivo": mt_def,
        "% MT Def": round(pct_defensivo_positivo, 2),
        }

    return stats_off_abs, stats_def_abs, tiempo_MMSS

def compute_team_max_values(df_events, df_minutes, df_mt):

    team_stats = {
        "Tiros": 0,
        "Tiros al Arco": 0,
        "Goles": 0,
        "MT Ofensivo": 0,
        "% MT Of": 100,


        "Pases Claves": 0,
        "Asistencias": 0,
        "Pérdida": 0,
        "Falta Recibida": 0,
        "1v1 Of Ganado %": 100,

        "1v1 Def Total": 0,
        "1v1 Def Ganado %": 100,
        "Recuperaciones": 0,
        "Presiones": 0,
        "Sanciones": 0,
        "MT Defensivo": 0,
        "% MT Def": 100,
        }

    for player in df_events["Nombre"].unique():
        stats_off, stats_def, _ = compute_player_stats_for_pizza(df_events, df_minutes, df_mt, player)

        for k, v in stats_off.items():
            if "%" not in k:
                team_stats[k] = max(team_stats[k], v)

        for k, v in stats_def.items():
            if "%" not in k:
                team_stats[k] = max(team_stats[k], v)

    return team_stats

def prepare_pizza_with_team_ranges(stats_off, stats_def, team_max):

    params = [
        "Tiros",
        "Tiros al Arco",
        "Goles",
        "MT Ofensivo",
        "% MT Of",

        "Pases Claves",
        "Asistencias",
        "Pérdida",
        "Falta Recibida",
        "1v1 Of Ganado %",
        
        "1v1 Def Total",
        "1v1 Def Ganado %",
        "Recuperaciones",
        "Presiones",
        "Sanciones",
        "MT Defensivo",
        "% MT Def",
    ]

    values = [
        stats_off["Tiros"],
        stats_off["Tiros al Arco"],
        stats_off["Goles"],
        stats_off["MT Ofensivo"],
        stats_off["% MT Of"],


        stats_off["Pases Claves"],
        stats_off["Asistencias"],
        stats_off["Pérdida"],
        stats_off["Falta Recibida"],
        stats_off["1v1 Of Ganado %"],
        
        stats_def["1v1 Def Total"],
        stats_def["1v1 Def Ganado %"],
        stats_def["Recuperaciones"],
        stats_def["Presiones"],
        stats_def["Sanciones"],
        stats_def["MT Defensivo"],
        stats_def["% MT Def"]
    ]

    min_range = [0] * len(params)

    max_range = []
    for p in params:
        if "%" in p:
            max_range.append(100)
        else:
            mx = team_max.get(p, 1)
            if mx == 0:
                mx = 1
            max_range.append(mx)

    # ✅ FIX: return OUTSIDE the loop
    return params, values, min_range, max_range

def plot_player_pizza_with_ranges(
    player_name,
    params,
    values,
    min_range,
    max_range,
    rival
):

    slice_colors = (
        ["#1A78CF"] * 5 +   # offensive
        ["#FF9300"] * 5 +   # possession
        ["#D70232"] * 7     # defensive
    )

    baker = PyPizza(
        params=params,
        min_range=min_range,
        max_range=max_range,
        background_color="#111111",
        straight_line_color="#222222",
        last_circle_color="#222222",
        last_circle_lw=1.5,
        straight_line_lw=1,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(9, 9),
        color_blank_space="same",
        blank_alpha=0.45,
        param_location=110,

        kwargs_slices=dict(
            facecolor=slice_colors,
            edgecolor="#000000",
            linewidth=1.2
        ),

        kwargs_params=dict(
            color="#FFFFFF",
            fontsize=10,
            va="center"
        ),

        kwargs_values=dict(
            color="#000000",
            fontsize=10,
            bbox=dict(
                facecolor="#FFFFFF",
                edgecolor="#000000",
                lw=1,
                boxstyle="round,pad=0.2"
            )
        )
    )

    fig.text(0.5, 0.97, f"{player_name} vs {rival}",
             ha="center", color="white", fontsize=18)

    # fig.text(0.5, 0.95, "Rango: 0 → Máximo de la Plantilla",
    #          ha="center", color="white", fontsize=12)

    fig.patch.set_facecolor("#111111")
    return fig

def plot_player_action_map(pdf, player_name, modo, images_path="img"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.image as mpimg
    import os

    style = ACTION_STYLE

    fig, ax = plt.subplots(figsize=(9, 7))
    ax = draw_futsal_pitch_grid(ax)
    ax.set_title(f"Mapa de Acciones - {player_name} ({modo})", fontsize=14)

    pdf = pdf[pdf["Nombre"] == player_name]

    # Load images
    goal_icon = "goal.png"
    save_icon = "gloves.png"

    goal_path = os.path.join(images_path, style["shots"]["goal_icon"])
    save_path = os.path.join(images_path, style["shots"]["save_icon"])

    if os.path.exists(goal_path):
        goal_icon = mpimg.imread(goal_path)

    if os.path.exists(save_path):
        save_icon = mpimg.imread(save_path)

    # Helper → arrows
    def draw_arrow(x1, y1, x2, y2, color):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=style["key_pass"]["arrow_width"]
            ),
            zorder=6
        )

    # ===================================================
    #                     OFENSIVO
    # ===================================================
    if modo == "Ofensivo":

        # ---- Tiro Ferro ----
        tiros = pdf[pdf["Acción"] == "Tiro Ferro"]
        for _, r in tiros.iterrows():

            # Draw shooting line
            ax.annotate(
                "",
                xy=(r["FieldXto"], r["FieldYto"]),
                xytext=(r["FieldXfrom"], r["FieldYfrom"]),
                arrowprops=dict(
                    arrowstyle="-",
                    color=style["shots"]["line_color"],
                    alpha=style["shots"]["line_alpha"],
                    lw=2
                ),
                zorder=4,
            )

            # RESULT ICONS
            end_x, end_y = r["FieldXto"], r["FieldYto"]

            # Gol → ball.png
            if r["Resultado"] == "Gol" and goal_icon is not None:
                s = style["shots"]["icon_size"]/2
                ax.imshow(goal_icon,
                    extent=(end_x - s, end_x + s, end_y - s, end_y + s),
                    zorder=10)
            
            # Atajado → gloves.png
            elif r["Resultado"] == "Atajado" and save_icon is not None:
                s = style["shots"]["icon_size"]/2
                ax.imshow(save_icon,
                    extent=(end_x - s, end_x + s, end_y - s, end_y + s),
                    zorder=10)

            # Bloqueado → grey small square
            elif r["Resultado"] == "Bloqueado":
                ax.text(end_x, end_y, style["shots"]["blocked_symbol"],
                        color=style["shots"]["blocked_color"],
                        fontsize=10, ha="center", va="center")

        # ---- Pérdida ----
        for _, r in pdf[pdf["Acción"] == "Perdida"].iterrows():
            ax.text(r["FieldX"], r["FieldY"], style["loss"]["symbol"],
                    color=style["loss"]["color"],
                    fontsize=style["loss"]["size"],
                    ha="center", va="center")

        # ---- 1v1 Ofensivo ----
        for _, r in pdf[pdf["Acción"] == "1 VS 1 Ofensivo"].iterrows():
            ax.text(r["FieldX"], r["FieldY"], style["1v1_off"]["symbol"],
                    color=style["1v1_off"]["color"],
                    fontsize=style["1v1_off"]["size"],
                    ha="center", va="center")

    # ---- Pase Clave ----
    for _, r in pdf[pdf["Acción"] == "Pase Clave"].iterrows():
        color = (
            style["key_pass"]["correct_color"]
            if r["Resultado"] == "Correcto"
            else style["key_pass"]["wrong_color"]
        )

        # always draw an ARROW →
        ax.annotate(
            "",
            xy=(r["FieldXto"], r["FieldYto"]),
            xytext=(r["FieldXfrom"], r["FieldYfrom"]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=style["key_pass"]["arrow_width"],
                mutation_scale=style["key_pass"]["head_size"]
            ),
            zorder=7
        )


        # ---- Falta Recibida ----
        for _, r in pdf[pdf["Acción"] == "Falta Recibida"].iterrows():
            circ = plt.Circle(
                (r["FieldX"], r["FieldY"]),
                style["foul_received"]["radius"],
                edgecolor=style["foul_received"]["color"],
                fill=False, lw=2
            )
            ax.add_patch(circ)

    # ===================================================
    #                     DEFENSIVO
    # ===================================================
    if modo == "Defensivo":

        # ---- Recuperación ----
        for _, r in pdf[pdf["Acción"] == "Recuperación"].iterrows():
            ax.text(r["FieldX"], r["FieldY"],
                    style["recovery"]["symbol"],
                    color=style["recovery"]["color"],
                    fontsize=style["recovery"]["size"],
                    ha="center", va="center")

        # ---- Presión ----
        for _, r in pdf[pdf["Acción"] == "Presión"].iterrows():
            ax.text(r["FieldX"], r["FieldY"],
                    style["pressure"]["symbol"],
                    color=style["pressure"]["color"],
                    fontsize=style["pressure"]["size"],
                    ha="center", va="center")

        # ---- 1v1 Defensivo ----
        for _, r in pdf[pdf["Acción"] == "1 VS 1 Defensivo"].iterrows():
            col = (
                style["1v1_def"]["win_color"]
                if r["Resultado"] == "Ganado"
                else style["1v1_def"]["lose_color"]
            )
            s = style["1v1_def"]["size"]
            square = patches.Rectangle(
                (r["FieldX"] - s/2, r["FieldY"] - s/2),
                s, s,
                edgecolor=col, fill=False, lw=2
            )
            ax.add_patch(square)

        # ---- Falta Cometida ----
        for _, r in pdf[pdf["Acción"] == "Falta Cometida"].iterrows():
            circ = plt.Circle(
                (r["FieldX"], r["FieldY"]),
                style["foul_committed"]["radius"],
                edgecolor=style["foul_committed"]["color"],
                fill=False, lw=2
            )
            ax.add_patch(circ)

    # ===================================================
    #                 LEGEND AT BOTTOM
    # ===================================================
    legend_items = [
        plt.Line2D([0], [0], color="green", lw=2, label="Tiro / Pase Clave"),

        plt.Line2D([0], [0], marker="x", color="red", lw=0, markersize=8,
                label="Pérdida"),

        plt.Line2D([0], [0], marker="^", color="green", lw=0, markersize=8,
                label="1v1 Ofensivo"),

        plt.Line2D([0], [0], marker="*", color="blue", lw=0, markersize=8,
                label="Recuperación"),

        plt.Line2D([0], [0], marker="P", color="blue", lw=0, markersize=8,
                label="Presión"),

        plt.Line2D([0], [0], marker="s", color="green", lw=0, markersize=8,
                label="1v1 Def Ganado"),

        plt.Line2D([0], [0], marker="s", color="red", lw=0, markersize=8,
                label="1v1 Def Perdido"),

        plt.Line2D([0], [0], marker="o", color="green", fillstyle="none", lw=1,
                label="Falta Recibida"),

        plt.Line2D([0], [0], marker="o", color="red", fillstyle="none", lw=1,
                label="Falta Cometida"),
    ]


    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10
    )

    plt.tight_layout()
    return fig


# Map DFs to effective timeline

def map_df1_to_effective_time(df_1, timeline, max_delay=180, min_delay=30):
    """
    Maps df_1 actions (Tiempo = video timestamp) to effective match time.

    Steps:
    1. Direct match on (Jornada, Tiempo_Video)
    2. If missing → backward search (late tagging)
    3. If still missing → forward search (early tagging)
    """

    df_1 = df_1.copy()
    df_1["Tiempo_Video"] = df_1["Tiempo"].astype(int)

    # Direct merge
    merged = df_1.merge(
        timeline,
        on=["Jornada", "Tiempo_Video"],
        how="left",
        suffixes=("", "_tl")
    )

    # Missing rows
    missing_mask = merged["Tiempo_Partido_Segundos"].isna()
    if missing_mask.sum() == 0:
        return merged

    timeline_sorted = timeline.sort_values(["Jornada", "Tiempo_Video"])

    # Iterate missing rows
    for idx in merged[missing_mask].index:
        jor = merged.at[idx, "Jornada"]
        t   = merged.at[idx, "Tiempo_Video"]

        found = None

        # 1) BACKWARD search (late tag)
        for d in range(1, max_delay + 1):
            t_search = t - d
            match = timeline_sorted[
                (timeline_sorted["Jornada"] == jor) &
                (timeline_sorted["Tiempo_Video"] == t_search)
            ]
            if len(match) > 0:
                found = match.iloc[0]
                break

        # 2) FORWARD search (early tag)
        if found is None:
            for d in range(1, min_delay + 1):
                t_search = t + d
                match = timeline_sorted[
                    (timeline_sorted["Jornada"] == jor) &
                    (timeline_sorted["Tiempo_Video"] == t_search)
                ]
                if len(match) > 0:
                    found = match.iloc[0]
                    break

        # Apply if found
        if found is not None:
            for col in [
                "Parte",
                "Tiempo_Partido_Segundos",
            ]:
                merged.at[idx, col] = found[col]

    return merged

def build_effective_timeline(df_4: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un timeline efectivo segundo a segundo para TODOS los partidos (Jornada),
    usando únicamente los intervalos donde el balón estuvo en juego.
    
    df_4 debe tener columnas:
        - Jornada
        - Parte
        - Inicio (segundos de vídeo donde empieza un tramo en juego)
        - Fin    (segundos de vídeo donde termina ese tramo en juego)
    """

    rows = []

    for jornada in sorted(df_4["Jornada"].unique()):
        df_j = df_4[df_4["Jornada"] == jornada]

        eff_match = 0  # contador acumulado del partido completo

        for parte in sorted(df_j["Parte"].unique()):
            df_jp = df_j[df_j["Parte"] == parte].sort_values("Tiempo")
            eff_parte = 0  # contador que se reinicia en cada parte

            for _, row in df_jp.iterrows():
                start = int(row["Tiempo"])
                end   = int(row["Fin"])

                for t in range(start, end + 1):  # segundo a segundo
                    rows.append({
                        "Jornada": jornada,
                        "Parte": parte,
                        "Tiempo_Video": t,
                        "Tiempo_Partido_Segundos": eff_match,
                    })
                    eff_match += 1
                    eff_parte += 1

    return pd.DataFrame(rows)

def build_minute_timeline(df_timeline: pd.DataFrame):

    df = df_timeline.copy()

    # Ensure effective time exists
    df = df[df["Tiempo_Partido_Segundos"].notna()]

    # Compute minute from seconds
    df["minuto"] = (df["Tiempo_Partido_Segundos"] // 60).astype(int)

    results = []

    # Process per match (per Jornada)
    for jornada in sorted(df["Jornada"].dropna().unique()):
        df_j = df[df["Jornada"] == jornada].copy()

        # Identify teams (should be constant)
        try:
            local = df_j["Local"].iloc[0]
            visit = df_j["Visitante"].iloc[0]
        except:
            local, visit = None, None

        # Shot indicators
        df_j["tF"] = (df_j["Acción"] == "Tiro Ferro").astype(int)
        df_j["gF"] = ((df_j["Acción"] == "Tiro Ferro") &
                      (df_j["Resultado"] == "Gol")).astype(int)

        df_j["tR"] = (df_j["Acción"] == "Tiro Rival").astype(int)
        df_j["gR"] = ((df_j["Acción"] == "Tiro Rival") &
                      (df_j["Resultado"] == "Gol")).astype(int)

        # Aggregate per minute
        df_min = (
            df_j.groupby("minuto")[["tF", "gF", "tR", "gR"]]
            .sum()
            .reset_index()
        )

        # Create full minute range (0 → last minute of match)
        max_min = int(df_j["minuto"].max())
        full = pd.DataFrame({"minuto": range(0, max_min + 1)})

        df_full = full.merge(df_min, on="minuto", how="left").fillna(0)
        df_full[["tF", "gF", "tR", "gR"]] = df_full[["tF", "gF", "tR", "gR"]].astype(int)

        # Accumulated goals
        df_full["gF_acum"] = df_full["gF"].cumsum()
        df_full["gR_acum"] = df_full["gR"].cumsum()

        # Match status (POV Ferro)
        def estado(row):
            if row["gF_acum"] > row["gR_acum"]:
                return "Ganando"
            elif row["gF_acum"] == row["gR_acum"]:
                return "Empatando"
            else:
                return "Perdiendo"

        df_full["estado"] = df_full.apply(estado, axis=1)

        # Store Jornada + Team metadata
        df_full["Jornada"] = jornada
        df_full["Local"] = local
        df_full["Visitante"] = visit

        results.append(df_full)

    # Combine all matches
    df_out = pd.concat(results, ignore_index=True)

    return df_out



from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


# ============================
#  ICON FUNCTION (UNCHANGED)
# ============================
def add_png_icon(ax, x, y, png_path, zoom=0.03):
    """
    Places a tiny PNG on the plot that does NOT affect scaling.
    zoom=0.02 → very small
    zoom=0.03 → small speck (recommended)
    """
    try:
        img = mpimg.imread(png_path)
    except Exception:
        return  # fail silently if missing

    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        (x, y),         # position on the chart
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=20
    )
    ax.add_artist(ab)


# ============================
#  CORE MOMENTUM LOGIC
# ============================
def add_momentum_bars(df_minuto, w_shot=1, w_goal=1):
    """
    Adds a 'momentum' column to the per-minute dataframe.
    """
    df = df_minuto.copy()

    df["momentum"] = (
        df["tF"] * w_shot +
        df["gF"] * w_goal -
        df["tR"] * w_shot -
        df["gR"] * w_goal
    )

    return df


def plot_momentum_bars(df_mom, jornada, ball_path="img/ball.png"):
    """
    Plots the momentum bars for a given jornada.
    Returns a Matplotlib figure (ready for st.pyplot).
    """

    df = df_mom[df_mom["Jornada"] == jornada].copy()
    if df.empty:
        # Return empty figure with message so Streamlit doesn't explode
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(
            0.5, 0.5,
            f"No hay datos de momentum para la Jornada {jornada}",
            color="white", ha="center", va="center", fontsize=14,
            transform=ax.transAxes
        )
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")
        ax.axis("off")
        return fig

    minutos = df["minuto"]
    momentum = df["momentum"]

    local = df["Local"].iloc[0]
    visit = df["Visitante"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))

    # ============================
    #  BACKDROP & LIMITS
    # ============================
    max_abs = float(np.nanmax(np.abs(momentum))) if len(momentum) > 0 else 1.0
    max_abs = max(max_abs, 1.0)

    # Soft green band above, red band below
    ax.axhspan(0,  max_abs * 1.1, facecolor="#00ff88", alpha=0.06, zorder=0)
    ax.axhspan(-max_abs * 1.1, 0, facecolor="#ff4f4f", alpha=0.06, zorder=0)

    # Zero reference line
    ax.axhline(0, color="white", linewidth=1.2, alpha=0.8, zorder=1)

    # ============================
    #  BARS
    # ============================
    colors = np.where(momentum >= 0, "#00ff88", "#ff4f4f")

    bars = ax.bar(
        minutos,
        momentum,
        color=colors,
        width=0.8,
        zorder=3,
        alpha=0.9,
        edgecolor="#111111",
        linewidth=0.5,
    )

    # ============================
    #  TREND LINE (ROLLING 3')
    # ============================
    if len(df) >= 3:
        trend = momentum.rolling(window=3, center=True).mean()
        ax.plot(
            minutos,
            trend,
            linewidth=2.2,
            color="white",
            alpha=0.9,
            zorder=4,
        )

    # ============================
    #  GOAL ICONS
    # ============================
    ICON_OFFSET = -0.25   # how far outside the bar tip (up/down)
    ICON_ZOOM   = 0.16    # tiny speck size

    for _, row in df.iterrows():
        x = row["minuto"]
        mom = row["momentum"]

        # Ferro goal → ball at END of positive or zero bar
        if row["gF"] > 0:
            if mom >= 0:
                y = mom + ICON_OFFSET     # above top
            else:
                y = mom - ICON_OFFSET     # below bottom (rare)
            add_png_icon(ax, x, y, ball_path, zoom=ICON_ZOOM)

        # Rival goal → also at end of bar
        if row["gR"] > 0:
            if mom >= 0:
                y = mom + ICON_OFFSET
            else:
                y = mom - ICON_OFFSET
            add_png_icon(ax, x, y, ball_path, zoom=ICON_ZOOM)

    # ============================
    #  AXES, GRID, TICKS
    # ============================
    ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

    # X ticks every 5 minutes (if minutes are numeric)
    try:
        xmin, xmax = int(min(minutos)), int(max(minutos))
        step = 5 if (xmax - xmin) > 10 else 1
        ax.set_xticks(np.arange(xmin, xmax + 1, step))
    except Exception:
        # Fallback if something weird happens
        pass

    ax.set_xlabel("Minuto", color="white", fontsize=12)
    ax.set_ylabel("Momentum", color="white", fontsize=12)

    # Nice grid on y-axis only
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    # ============================
    #  TITLE & LEGEND
    # ============================
    total_gF = int(df["gF"].sum())
    total_gR = int(df["gR"].sum())

    ax.set_title(
        f"Momentum del Partido – Jornada {jornada}",
        color="white",
        fontsize=16,
        pad=14,
    )

    # Dummy handles for legend
    ax.bar([], [], color="#00ff88", label="Momentum Ferro")
    ax.bar([], [], color="#ff4f4f", label="Momentum Rival")
    if len(df) >= 3:
        ax.plot([], [], color="white", linewidth=2.2, label="Tendencia (3′)")


    # ============================
    #  BACKGROUND
    # ============================
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    # Small horizontal margin
    ax.margins(x=0.01)

    return fig


# MEDIOS TÁCTICOS OFENSIVOS PLOT

def draw_player_bar(ax, y, pct, label, color="#00A65A"):
    """
    Draws a clean horizontal bar:
    - left side = percentage fill
    - right side = text label
    """

    # BAR POSITION (axis coordinates)
    bar_x0 = 0.02
    bar_w  = 0.50
    bar_h  = 0.015

    # --- background ---
    ax.barh(
        y=y,
        width=bar_w,
        height=bar_h,
        color="#E0E0E0",
        edgecolor="none",
        align="center"
    )

    # --- fill ---
    ax.barh(
        y=y,
        width=bar_w * (pct / 100),
        height=bar_h,
        color=color,
        edgecolor="none",
        align="center"
    )

    # --- text ---
    ax.text(
        bar_x0 + bar_w + 0.03,
        y,
        label,
        fontsize=15,
        va="center",
    )


def plot_medios_tacticos_ofensivos(
    df,
    tipo_col="Tipo_de_Accion",
    player_col="Nombre",
    result_col="Resultado",
    evento_col="Evento",
    success_value="Positivo"
):
    """
    Figura con:
      1) Barras solapadas del total de acciones (evento único)
      2) Bloques por MT Acción mostrando TODOS los jugadores con barras tipo loading
    """

    # ======================================================
    # 0️⃣ BUILD EVENT TABLE (ONE ROW PER EVENTO)
    # ======================================================
    event_table = (
        df.groupby(evento_col)
        .agg(
            tipo=(tipo_col, "first"),
            resultado_evento=(result_col, lambda x: (x == success_value).any())
        )
        .reset_index()
    )
    event_table["intento"] = 1
    event_table["exito"] = event_table["resultado_evento"].astype(int)

    # ======================================================
    # 1️⃣ AGG PER TIPO ACCIÓN (TOP BAR CHART)
    # ======================================================
    grouped = (
        event_table.groupby("tipo")
        .agg(intentos=("intento", "sum"),
             exitos=("exito", "sum"))
        .reset_index()
    )
    grouped["pct"] = grouped["exitos"] / grouped["intentos"] * 100
    grouped = grouped.sort_values("intentos", ascending=False)

    tipos = grouped["tipo"].tolist()
    intentos = grouped["intentos"].to_numpy()
    exitos = grouped["exitos"].to_numpy()

    # ======================================================
    # FIGURE BASE
    # ======================================================
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    # fig.suptitle("Medios Tácticos Ofensivos", fontsize=20, fontweight="bold")

    # ======================================================
    # 2️⃣ TOP CHART (STACKED OVERLAPPING BARS)
    # ======================================================
    x = np.arange(len(tipos))
    width = 0.65

    ax1.bar(x, intentos, width=width, color="#B6F2B0", alpha=0.7, label="Intentos")
    ax1.bar(x, exitos, width=width*0.75, color="#00A65A", alpha=0.9, label="Éxitos")

    ax1.set_xticks(x)
    ax1.set_xticklabels(tipos, rotation=0, ha="right", fontsize=15)
    # Bigger y-axis labels
    ax1.tick_params(axis='y', labelsize=14)
    
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend()

    # ======================================================
    # 3️⃣ BOTTOM SECTION — ALL PLAYERS PER MT WITH LOADING BARS
    # ======================================================
    ax2.axis("off")
    y_cursor = 0.90

    # Build player-event table (each player per unique event)
    player_events = (
        df.groupby([player_col, evento_col])
        .agg(event_success=(result_col, lambda x: (x == success_value).any()))
        .reset_index()
    )

    for tipo in tipos:

        # ---- EVENT SUMMARY ----
        ev = event_table[event_table["tipo"] == tipo]
        tot = ev["intento"].sum()
        suc = ev["exito"].sum()
        pct = suc / tot * 100 if tot > 0 else 0

        # TITLE
        ax2.text(0.02, y_cursor, f"{tipo}  ({suc}/{tot})  {pct:.1f}%",
                fontsize=15, fontweight="bold", ha="left")
        y_cursor -= 0.035

        # ---- PLAYERS ----
        pe = df[df[tipo_col] == tipo][[player_col, evento_col]].drop_duplicates()
        pe = pe.merge(player_events, on=[player_col, evento_col], how="left")

        players = (
            pe.groupby(player_col)
            .agg(
                intentos=(evento_col, "count"),
                exitos=("event_success", "sum"),
            )
            .reset_index()
        )
        



        # sort players by success %, then attempts
        players["pct"] = players["exitos"] / players["intentos"] * 100
        players = (
                players.sort_values("intentos", ascending=False)
                .head(3))
        
        # ---- DRAW CLEAN HORIZONTAL BARS ----
        # for each player in this MT
        for _, pl in players.iterrows():
            p_name = pl[player_col]
            att = int(pl["intentos"])
            suc_p = int(pl["exitos"])
            pct_p = (suc_p / att) * 100 if att > 0 else 0

            label = f"{p_name} ({suc_p}/{att}) {pct_p:.1f}%"

            draw_player_bar(ax2, y_cursor, pct_p, label)
            
            y_cursor -= 0.03


        y_cursor -= 0.04  # space between MT blocks


    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ============================
#  MT OFENSIVOS – NETWORK PLOT
# ============================
from itertools import combinations


# BASE POSITIONS FOR FUTSAL ROLE COORDINATES
POSITION_COORDS = {
    "Arquero":     (0.10, 0.50),
    "Cierre":      (0.30, 0.50),
    "Ala Zurdo":   (0.55, 0.75),
    "Ala Diestro": (0.55, 0.25),
    "Pivot":       (0.85, 0.50),
    "Otros":       (0.50, 0.50),
}


def assign_player_positions_from_m2(m2,
                                    player_col="Nombre",
                                    pos_col="Posicion"):
    """
    Takes m2 → assigns futsal pitch coordinates based on position,
    jittering players who share the same role.
    """
    df_nodes = m2[[player_col, pos_col]].dropna().drop_duplicates().copy()
    df_nodes[pos_col] = df_nodes[pos_col].fillna("Otros")

    nodes = {}

    for pos, group in df_nodes.groupby(pos_col):
        base_x, base_y = POSITION_COORDS.get(pos, POSITION_COORDS["Otros"])
        n = len(group)

        if n == 1:
            nodes[group[player_col].iloc[0]] = (base_x, base_y)
        else:
            radius = 0.04
            for idx, (_, row) in enumerate(group.iterrows()):
                angle = 2 * np.pi * idx / n
                x = base_x + radius * np.cos(angle)
                y = base_y + radius * np.sin(angle)
                x = min(max(x, 0.05), 0.95)
                y = min(max(y, 0.05), 0.95)
                nodes[row[player_col]] = (x, y)

    return nodes

def plot_mt_network_two_axes(df_mt_of,
                             m2,
                             evento_col="Evento",
                             player_col="Nombre",
                             pos_col="Posicion",
                             result_col="Resultado",
                             success_value="Positivo"):
    """
    LEFT AXIS (ax1): Full pitch network (all edges, no text)
    RIGHT AXIS (ax2): Top 5 connections with progress bars
    Layout uses manual axis positioning → NO whitespace issues.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations

    # --------------------------------------------
    # 0️⃣ NO DATA CASE
    # --------------------------------------------
    if df_mt_of.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No hay datos MT Ofensivos", ha="center")
        ax.axis("off")
        return fig

    # --------------------------------------------
    # 1️⃣ MERGE POSITIONS FROM m2
    # --------------------------------------------
    players = df_mt_of[player_col].dropna().unique()
    pos_info = m2[[player_col, pos_col]].dropna().drop_duplicates()
    pos_info = pos_info[pos_info[player_col].isin(players)]

    # Add missing players as "Otros"
    missing = set(players) - set(pos_info[player_col])
    if missing:
        extra = pd.DataFrame({player_col: list(missing),
                              pos_col: ["Otros"] * len(missing)})
        pos_info = pd.concat([pos_info, extra], ignore_index=True)

    # Assign pitch coordinates
    node_positions = assign_player_positions_from_m2(
        pos_info, player_col=player_col, pos_col=pos_col
    )

    # --------------------------------------------
    # 2️⃣ BUILD EDGE TOTALS + SUCCESS COUNTS
    # --------------------------------------------
    edge_total = {}
    edge_success = {}

    for ev, group in df_mt_of.groupby(evento_col):
        plist = sorted(group[player_col].dropna().unique())
        if len(plist) < 2:
            continue

        is_positive = (group[result_col] == success_value).any()

        for a, b in combinations(plist, 2):
            key = tuple(sorted((a, b)))

            edge_total[key] = edge_total.get(key, 0) + 1
            if is_positive:
                edge_success[key] = edge_success.get(key, 0) + 1

    if not edge_total:
        max_weight = 1
    else:
        max_weight = max(edge_total.values())

    # --------------------------------------------
    # 3️⃣ FIND TOP 5 CONNECTED PAIRS
    # --------------------------------------------
    sorted_edges = sorted(edge_total.items(),
                          key=lambda x: x[1],
                          reverse=True)
    top5 = sorted_edges[:5]

    # --------------------------------------------
    # 4️⃣ FIGURE & MANUAL AXIS POSITIONING
    # --------------------------------------------
    fig = plt.figure(figsize=(20, 10))

    # LEFT AX → PITCH NETWORK
    ax1 = fig.add_axes([0.03, 0.05, 0.65, 0.90])
    # [left, bottom, width, height]

    # RIGHT AX → TOP 5 PANEL
    ax2 = fig.add_axes([0.72, 0.20, 0.25, 0.60])
    # [left, bottom, width, height]

    # --------------------------------------------
    # 5️⃣ DRAW LEFT AX (FULL NETWORK)
    # --------------------------------------------
    draw_futsal_pitch_grid(ax1)

    # ----- edges (all shown) -----
    for (a, b), w in edge_total.items():
        x1, y1 = node_positions[a]
        x2, y2 = node_positions[b]

        lw = 1 + 3 * (w / max_weight)
        alpha = 0.35 + 0.40 * (w / max_weight)

        ax1.plot(
            [x1, x2], [y1, y2],
            color="#00A65A",
            linewidth=lw,
            alpha=alpha,
            zorder=1
        )

    # ----- nodes -----
    inv = (
        df_mt_of.groupby(player_col)[evento_col]
        .nunique()
        .reset_index(name="n_eventos")
    )
    pos_info = pos_info.merge(inv, on=player_col, how="left").fillna({"n_eventos": 0})

    for _, row in pos_info.iterrows():
        name = row[player_col]
        n_ev = row["n_eventos"]
        x, y = node_positions[name]

        size = 90 + 25 * n_ev

        ax1.scatter(x, y,
                    s=size,
                    color="white",
                    edgecolors="#006837",
                    linewidths=2,
                    zorder=3)

        ax1.text(x, y + 0.035,
                 name.split()[-1],
                 fontsize=12,
                 fontweight="medium",
                 ha="center", va="bottom",
                 zorder=4)

    ax1.set_title("Red MT Ofensivos", fontsize=22, fontweight="bold", pad=10)
    ax1.axis("off")

    # --------------------------------------------
    # 6️⃣ DRAW RIGHT AX (TOP 5 LIST + PROGRESS BARS)
    # --------------------------------------------
    ax2.axis("off")

    # TITLE
    ax2.text(
        0.5, 0.98,
        "Top 5 Conexiones",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="top"
    )

    # Separator line
    ax2.hlines(0.94, xmin=0.0, xmax=1.0, color="#CCCCCC", linewidth=1.2)

    # SPACE FOR TOP 5
    y = 0.88

    for (a, b), total_w in top5:

        success_w = edge_success.get((a, b), 0)
        pct = success_w / total_w if total_w > 0 else 0

        # ----- Line 1: Names -----
        ax2.text(
            0.0, y,
            f"{a}  ↔  {b}",
            fontsize=15, fontweight="bold",
            ha="left"
        )

        y -= 0.06

        # ----- Line 2: PROGRESS BAR -----
        bar_x0 = 0.0
        bar_w = 0.75
        bar_h = 0.045

        ax2.barh(y, bar_w, height=bar_h, color="#E0E0E0")
        ax2.barh(y, bar_w * pct, height=bar_h, color="#00A65A")

        ax2.text(
            bar_x0 + bar_w + 0.05, y,
            f"({success_w}/{total_w})",
            fontsize=14,
            va="center"
        )

        y -= 0.10

    plt.tight_layout()
    return fig

# -----------------------------
# 5. STREAMLIT RENDER
# -----------------------------
def render(df_1: pd.DataFrame, df_2: pd.DataFrame, df_tiempos: pd.DataFrame, df_4):
    st.markdown("<h2 style='color:white;text-align:center;'>Estadísticas del Partido</h2>",
                unsafe_allow_html=True)

    # Add Rival column derived from Local/Visitante 
    df1 = add_rival_column(df_1)  
    df2 = add_rival_column(df_2)


    # Filters

    # Build unique match list directly (NO new df columns)
    matches = (
        df1[["Jornada", "Rival", "Local"]]
        .drop_duplicates()
        .sort_values("Jornada")
    )

    # Create display labels on the fly
    labels = [
        f"Fecha {row.Jornada} - {row.Rival} - {'Casa' if row.Local=='Ferro' else 'Fuera'}"
        for row in matches.itertuples()
    ]

    # Single filter
    selected = st.selectbox("Seleccionar Partido", labels)

    # Extract the selected row
    idx = labels.index(selected)
    row = matches.iloc[idx]

    # Final outputs
    jornada = row.Jornada
    rival   = row.Rival
    lugar   = "Casa" if row.Local == "Ferro" else "Fuera"

    # Slice match data
    m1 = df1[(df1.Jornada==jornada)&(df1.Rival==rival)]
    m2 = df2[(df2.Jornada==jornada)&(df2.Rival==rival)]
    m3 = df_tiempos[(df_tiempos.Jornada==jornada)]
    m4 = df_4[(df_4.Jornada==jornada)]

    timeline = build_effective_timeline(m4)

    df_1_timeline = map_df1_to_effective_time(m1, timeline)
    df_1_minute = build_minute_timeline(df_1_timeline)


    if m1.empty:
        st.warning("No hay datos disponibles para este partido.")
        return

    # Detect sides
    local_team = m1.iloc[0]["Local"]
    visit_team = m1.iloc[0]["Visitante"]
    ferro_is_local = local_team=="Ferro"

    # --- Compute stats ---
    stats_dict = compute_match_stats_b1(m1)
    t_ofe, t_def = compute_transitions(m2)
    gF, gR = get_score_ferro_rival(m1)

    score_local, score_visit = (gF,gR) if ferro_is_local else (gR,gF)
    stats_df = build_stats_df(stats_dict,t_ofe,t_def,ferro_is_local)

    # --- Create main stats graphic ---
    fig_stats = plot_stats(stats_df,local_team,visit_team,score_local,score_visit)
    st.pyplot(fig_stats, use_container_width=True)

    st.markdown("---")

    # ========= SHOTMAP SECTION ========

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_shots = render_futsal_shotmap(m1, local_team, visit_team)
        st.pyplot(fig_shots, use_container_width=True)

    with col2:
        fig_pies = plot_shot_result_pies(m1, local_team, visit_team)
        st.pyplot(fig_pies, use_container_width=True)

    # ========================================
    # 🔥🔥 MOMENTUM SECTION (NEW)
    # ========================================

    w_shot = 1
    w_goal = 1

    # compute momentum
    df_mom = add_momentum_bars(df_1_minute, w_shot=w_shot, w_goal=w_goal)

    fig_mom = plot_momentum_bars(df_mom, jornada, ball_path="img/ball.png")
    st.pyplot(fig_mom, use_container_width=True)

    st.markdown("---")
        
    # ============================  
    # 📍 PÉRDIDA / RECUPERACIÓN SECTION  
    # ============================

    st.markdown("<h3 style='color:white;text-align:center;'>Pérdidas & Recuperaciones</h3>",
                unsafe_allow_html=True)

    option = st.radio("Seleccionar vista", ["Pérdida", "Recuperación"], horizontal=True)

    df_event = m1[m1["Acción"] == option].copy()

    if df_event.empty:
        st.warning(f"No hay eventos registrados para {option} en este partido.")
    else:
        # --- Clean coordinates ---
        df_event["FieldX"] = pd.to_numeric(df_event["FieldX"], errors="coerce")
        df_event["FieldY"] = pd.to_numeric(df_event["FieldY"], errors="coerce")
        df_event = df_event.dropna(subset=["FieldX","FieldY"]).copy()

        # ================= FIG LAYOUT =================
        fig = plt.figure(figsize=(11,6), facecolor="#062e16")
        gs = fig.add_gridspec(
            2,2,
            width_ratios=[2.4,0.9],     # 🔥 Zone plot now thinner
            height_ratios=[2,1],
            wspace=0.28, hspace=0.35
        )

        # ---------- 1) Campograma ----------
        ax_pitch = fig.add_subplot(gs[:,0])
        draw_futsal_pitch_grid(ax_pitch)

        color = "#ff4f4f" if option=="Pérdida" else "#00ff88"

        ax_pitch.scatter(
            df_event["FieldX"],
            df_event["FieldY"],
            s=140, alpha=0.85, color=color,
            edgecolors="black", linewidths=1.1
        )

        # 1A — Add surname next to point
        for _, row in df_event.iterrows():
            surname = str(row["Nombre"]).split()[-1]
            ax_pitch.text(
                row["FieldX"] + 0.015,
                row["FieldY"] + 0.015,
                surname,
                fontsize=9, color="white", weight="bold"
            )

        ax_pitch.set_title(f"{option} - Mapa General", color="white", fontsize=15)


        # ---------- 2) Zone Distribution ----------
        ax_zonas = fig.add_subplot(gs[0,1])

        df_event["Zona"] = (
            (df_event["FieldY"]*3).fillna(0).floordiv(1).astype(int)*3 +
            (df_event["FieldX"]*3).fillna(0).floordiv(1).astype(int) + 1
        )
        df_event = df_event[df_event["Zona"].between(1,9)]

        zonas_count = df_event["Zona"].value_counts().sort_index()

        ax_zonas.bar(
            zonas_count.index.astype(str),
            zonas_count.values,
            color=color, edgecolor="black"
        )

        ax_zonas.set_title("Zonas", color="white", fontsize=12)
        ax_zonas.tick_params(colors="white")
        ax_zonas.set_yticks(zonas_count.values)   # 🔥 whole numbers only
        for spine in ax_zonas.spines.values():
            spine.set_color("white")


        # ---------- 3) Top 3 Players (Surname Only) ----------
        ax_players = fig.add_subplot(gs[1,1])

        # convert full names → surname only
        df_event["Apellido"] = df_event["Nombre"].apply(lambda x: str(x).split()[-1])

        top_players = df_event["Apellido"].value_counts().head(3)

        ax_players.barh(
            top_players.index[::-1],
            top_players.values[::-1],
            color=color, edgecolor="black"
        )

        ax_players.set_title("Top 3", color="white", fontsize=12)
        ax_players.tick_params(colors="white")

        for spine in ax_players.spines.values():
            spine.set_color("white")

        # value labels
        for i,(name,val) in enumerate(zip(top_players.index[::-1], top_players.values[::-1])):
            ax_players.text(val+0.2, i, val, color="white", fontsize=11, va="center")

        st.pyplot(fig, use_container_width=True)


# ============================================
# 🔵 MEDIOS TÁCTICOS OFENSIVOS – VISUALIZACIÓN
# ============================================

    st.markdown(
        "<h3 style='color:white;text-align:center;'>Medios Tácticos Ofensivos</h3>",
        unsafe_allow_html=True
    )

    # Filter your dataframe to only MT Ofensivos (adjust condition if needed)
    df_mt_of = m2[m2["Acción"] == "MT Ofensivo"]

    if df_mt_of.empty:
        st.warning("No hay datos de MT Ofensivos para este partido.")
    else:
        # Generate the figure
        fig_mt = plot_medios_tacticos_ofensivos(
            df_mt_of,
            tipo_col="Tipo_de_Accion",
            player_col="Nombre",      # change to "Jugador" if needed
            result_col="Resultado",
            success_value="Positivo"     # change to Correcto / Completa / 1 if needed
        )

        # Center figure
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(fig_mt, use_container_width=True)


    # ==========================================================
    # 🔗 RED DE MEDIOS TÁCTICOS OFENSIVOS (PLAYER NETWORK)
    # ==========================================================

    # df_1 = main events dataframe
    df_mt_of = m2[m2["Acción"] == "MT Ofensivo"]

    st.markdown(
        "<h3 style='color:white;text-align:center;'>Red básica de MT Ofensivos</h3>",
        unsafe_allow_html=True
    )

    if df_mt_of.empty:
        st.info("No hay datos de MT Ofensivos para construir la red.")
    else:
        fig_net_basic = plot_mt_network_two_axes(
            df_mt_of,
            m2,
            evento_col="Evento",
            player_col="Nombre",
            pos_col="Posicion"
        )

        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.pyplot(fig_net_basic, use_container_width=True)

    # ============================
    # TRANSICIONES OFENSIVAS PLOT
    # ============================

    
    st.markdown("<h3 style='color:white;text-align:center;'>Transiciones Ofensivas y Defensivas</h3>",
                unsafe_allow_html=True)


    modo = st.radio(
        "Seleccionar Transiciones",
        ["Ofensiva", "Defensiva"],
        horizontal=True
    )

    df_trans = preprocess_transiciones(m2, modo=modo)
    fig = plot_transiciones(df_trans, m2, modo=modo)
    st.pyplot(fig, use_container_width=True)



    # ============================
    # 🟩 PLAYER PIZZA VISUALIZATION
    # ============================

    all_players = sorted(m1["Nombre"].dropna().unique())
    player_name = st.selectbox("Selecciona jugador", all_players)

    if player_name:

        # NEW — now returns 3 items (stats_off, stats_def, tiempo_MMSS)
        stats_off, stats_def, tiempo_MMSS = compute_player_stats_for_pizza(m1, m3, m2, player_name)

        # Team max (correct signature)
        team_max = compute_team_max_values(m1, m3, m2)

        # Prepare pizza
        params, values, min_range, max_range = prepare_pizza_with_team_ranges(
            stats_off, stats_def, team_max
        )

        # Position
        try:
            posicion = m3[m3["Nombre"] == player_name]["Posicion"].iloc[0]
        except:
            posicion = "Sin dato"

        # Dorsal (jersey number)
        try:
            dorsal = int(m3[m3["Nombre"] == player_name]["Jugadores"].iloc[0])
        except:
            dorsal = None

        img_path = f"img/players/{dorsal}.png" if dorsal else "img/players/DUMMY.png"
        headshot_exists = os.path.exists(img_path)
        
        # ====================================================
        # 🔥 2 COLUMNS → LEFT = Matplotlib Figure, RIGHT = Pizza
        # ====================================================
        col_info, col_pizza = st.columns([1.3,1.7])

        # ▓ LEFT FIGURE : PLAYER CARD ▓
        with col_info:

            fig_info = plt.figure(figsize=(4,6), facecolor="#111111")
            ax = fig_info.add_subplot(111)
            ax.axis("off")

            # Image block
            if headshot_exists:
                img = Image.open(img_path)
                img.thumbnail((500,500))
                ax_img = ax.inset_axes([0.18, 0.50, 0.64, 0.45])  # adjust placement
                ax_img.imshow(img)
                ax_img.axis("off")
            else:
                img = Image.open("img/players/DUMMY.png")
                img.thumbnail((500,500))
                ax_img = ax.inset_axes([0.18, 0.50, 0.64, 0.45])  # adjust placement
                ax_img.imshow(img)
                ax_img.axis("off")

            # Player Text
            ax.text(0.5,0.42,player_name.upper(),
                    ha="center",color="white",fontsize=17,fontweight="bold")

            ax.text(0.5,0.29,f"Posición: {posicion}",
                    ha="center",color="#30e6a5",fontsize=14)

            ax.text(0.5,0.18,f"Minutos: {tiempo_MMSS}",
                    ha="center",color="#9df792",fontsize=16,fontweight="bold")

            fig_info.tight_layout()
            st.pyplot(fig_info)  # <── render immediately (print-friendly)


        # ---------- RIGHT: Pizza Radar Plot ----------
        with col_pizza:
            pizza_fig = plot_player_pizza_with_ranges(
                player_name, params, values, min_range, max_range, rival
            )
            st.pyplot(pizza_fig, use_container_width=True)

    # st.markdown("<h3 style='color:white;text-align:center;'>Mapa de Acciones del Jugador</h3>",
    #             unsafe_allow_html=True)

    # modo = st.radio("Tipo de Acciones", ["Ofensivo", "Defensivo"], horizontal=True)
    
    # col1, col2, col3 = st.columns([1,3,1])
    # with col2:
    #     fig_actions = plot_player_action_map(m1, player_name, modo)
    #     st.pyplot(fig_actions, use_container_width=True)

    
