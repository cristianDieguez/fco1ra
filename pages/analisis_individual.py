import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Circle as MplCircle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch
from mplsoccer import PyPizza, add_image

ANCHO, ALTO = 1.0, 1.0
N_COLS, N_ROWS = 3, 3

st.markdown("""
<style>

/* ===============================
   RADIO — CLEAN GLASS PILL STYLE
   =============================== */

div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    gap: 18px;
    margin-top: 6px;
}

/* Pill container */
div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.10);
    border-radius: 999px;
    padding: 8px 20px;
    cursor: pointer;
    transition: all 0.25s ease;
    border: 1px solid rgba(255, 255, 255, 0.18);
}

/* Text */
div[role="radiogroup"] > label span {
    color: rgba(255, 255, 255, 0.75);
    font-weight: 600;
    font-size: 14px;
}

/* Hover */
div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.16);
}

/* Selected (Streamlit-safe) */
div[role="radiogroup"] > label[data-selected="true"] {
    background: rgba(155, 89, 182, 0.35);
    box-shadow: 0 0 14px rgba(155, 89, 182, 0.75);
    border-color: rgba(155, 89, 182, 0.9);
}

/* Selected text */
div[role="radiogroup"] > label[data-selected="true"] span {
    color: white;
}

/* Hide default radio circle */
input[type="radio"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)



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

RESULT_MARKERS = {
    "Desviado": "x",
    "Atajado": "s",
    "Bloqueado": "o",
    "Gol": "*",
    None: "."
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
    "Ataja": "green", 
    "Gol Recibido": "red", 
    "Rebote": "blue",


    None: "gray"
}


def plot_team_shots(df, ax, size=120, alpha=0.9):
    """
    df must include:
    - FieldXfrom (0-1)
    - FieldYfrom (0-1)
    - Resultado
    """

    for resultado, group in df.groupby("Resultado"):
        ax.scatter(
            group["FieldXfrom"],
            group["FieldYfrom"],
            s=size,
            alpha=alpha,
            marker=RESULT_MARKERS.get(resultado, "o"),
            color=RESULT_COLORS.get(resultado, "black"),
            edgecolors="black",
            linewidths=0.7,
            label=resultado if resultado is not None else "Sin dato"
        )

    ax.legend(loc="upper left", fontsize=8)
    return ax


def render_futsal_shotmap(df, equipo):
    """
    Streamlit-ready function.
    Input: full dataframe (Botonera 1)
    Output: matplotlib fig
    """

    # Filter only shots
    shots = df[df["Acción"] == equipo]

    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw your normalized futsal pitch
    draw_futsal_pitch_grid(ax)

    # Plot shots
    plot_team_shots(shots, ax)

    plt.tight_layout()
    return fig


# ============================================================
#  GLASS CARD HELPERS
# ============================================================

def glass_card_start(title=None):
    html = """
    <div class="glass-card" style="
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 12px 32px rgba(0,0,0,0.45);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 18px;
    ">
    """
    if title:
        html += f"<h3 style='color:white;margin-top:0;margin-bottom:12px;'>{title}</h3>"
    st.markdown(html, unsafe_allow_html=True)

def glass_card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def plot_event_result_donut(ax, df_event, result_column="Resultado", title=None):
    """
    Donut chart for result distribution of ANY event.
    Handles events where Resultado is NaN (e.g. fouls).
    """

    ax.set_facecolor("#0F0F0F")

    if df_event.empty or result_column not in df_event.columns:
        ax.text(
            0.5, 0.5, "Sin datos",
            ha="center", va="center",
            color="white", fontsize=12
        )
        ax.axis("off")
        return

    # ✅ FIX: treat NaN Resultado as valid category
    counts = (
        df_event[result_column]
        .fillna("Sin Resultado")
        .value_counts()
    )

    total = counts.sum()
    labels = counts.index.tolist()
    values = counts.values
    colors = [RESULT_COLORS.get(r, "gray") for r in labels]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="#0F0F0F"),
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        pctdistance=0.78,
        labeldistance=1.08,
        textprops=dict(color="white")
    )

    for t in texts:
        t.set_fontsize(9)

    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.text(
        0, 0.05,
        f"{total}",
        ha="center", va="center",
        fontsize=20,
        fontweight="bold",
        color="white"
    )

    ax.text(
        0, -0.18,
        "acciones",
        ha="center", va="center",
        fontsize=10,
        color="#BBBBBB"
    )

    if title:
        ax.set_title(
            title,
            color="white",
            fontsize=13,
            fontweight="bold",
            pad=14
        )

    ax.axis("equal")

def get_event_coordinates(df_event, event_name):
    """
    Returns coordinates suitable for plotting depending on event type.
    Produces:
        X, Y (single point)
        X2, Y2 (optional for passes)
        is_pass (bool) -> whether to draw trajectory lines
    """

    # --------------------------
    # 1. Shot events
    # --------------------------
    shot_events = ["Tiro Ferro", "Tiro Rival"]

    if event_name in shot_events:
        df_event = df_event.assign(
            X=df_event["FieldXfrom"],
            Y=df_event["FieldYfrom"],
            X2=None,
            Y2=None,
            is_pass=False
        )
        return df_event

    # --------------------------
    # 2. Duel / recovery / foul events
    # --------------------------
    point_events = [
        "Recuperación",
        "Presión",
        "1 VS 1 Ofensivo",
        "1 VS 1 Defensivo",
        "Falta Cometida",
        "Falta Recibida"
    ]

    if event_name in point_events:
        df_event = df_event.assign(
            X=df_event["FieldX"],
            Y=df_event["FieldY"],
            X2=None,
            Y2=None,
            is_pass=False
        )
        return df_event

    # --------------------------
    # 3. Pass events (trajectories)
    # --------------------------
    pass_events = ["Pase Clave", "Asistencia"]

    if event_name in pass_events:
        df_event = df_event.assign(
            X=df_event["FieldXfrom"],
            Y=df_event["FieldYfrom"],
            X2=df_event["FieldXto"],
            Y2=df_event["FieldYto"],
            is_pass=True
        )
        return df_event

    # --------------------------
    # Default: use center coords
    # --------------------------
    df_event = df_event.assign(
        X=df_event["FieldX"],
        Y=df_event["FieldY"],
        X2=None,
        Y2=None,
        is_pass=False
    )
    return df_event


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def plot_event_summary_two_axes(
    df_player,
    df_team,
    event_name,
    result_column="Resultado",
    title="Resumen del Evento"
):
    """
    Two-axis event summary visualization:
    - Left: Campograma
    - Right: Vertical panels: Donut, slim bars, summary text
    Handles missing 'is_pass' column safely.
    """

    # ============================ DATA ============================
    df_team_event = df_team[df_team["Acción"] == event_name]
    team_total = len(df_team_event)

    df_player_event = df_player[df_player["Acción"] == event_name]
    player_total = len(df_player_event)

    position_mode = df_player["Posicion"].mode().iloc[0] if "Posicion" in df_player.columns else "Desconocido"

    df_team_position_group = df_team[df_team["Posicion"] == position_mode]
    df_pos_event = df_team_position_group[df_team_position_group["Acción"] == event_name]
    pos_total = len(df_pos_event)

    player_pct = round((player_total / team_total) * 100, 1) if team_total > 0 else 0
    pos_pct = round((player_total / pos_total) * 100, 1) if pos_total > 0 else 0

    # ============================ FIGURE ============================
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

    # ---------- LEFT: Campograma ----------
    ax1 = fig.add_subplot(gs[0])
    draw_futsal_pitch_grid(ax1)  # your function to draw the pitch

    df_plot = get_event_coordinates(df_player_event, event_name)
    df_plot[result_column] = df_plot[result_column].fillna("Sin Resultado")

    for resultado, group in df_plot.groupby(result_column, dropna=False):
        is_pass = group["is_pass"].iloc[0] if "is_pass" in group.columns else False

        if len(group) > 0 and is_pass:
            for _, row in group.iterrows():
                ax1.plot([row["X"], row["X2"]],
                         [row["Y"], row["Y2"]],
                         color=RESULT_COLORS.get(resultado, "gray"),
                         linewidth=2, alpha=0.8)
                ax1.scatter(row["X"], row["Y"], s=120,
                            color=RESULT_COLORS.get(resultado, "gray"),
                            edgecolor="black", linewidth=0.7)
        else:
            ax1.scatter(group["X"], group["Y"], s=120, alpha=0.9,
                        marker=RESULT_MARKERS.get(resultado, "o"),
                        color=RESULT_COLORS.get(resultado, "gray"),
                        edgecolor="black", linewidths=0.7, label=str(resultado))

    ax1.set_title(f"{event_name} – Campograma", color="white")
    ax1.legend(loc="upper left", fontsize=8)

    # ---------- RIGHT: Vertical Panels ----------
    gs_right = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], height_ratios=[1, 0.3, 0.3,0.2], hspace=0.35)

    # 1️⃣ Distribución del Resultado
    ax_donut = fig.add_subplot(gs_right[0])
    plot_event_result_donut(ax=ax_donut,
                            df_event=df_player_event,
                            result_column=result_column,
                            title="Distribución del Resultado")

    # 2️⃣ Slim bar graphs: player % of team
    ax_bar1 = fig.add_subplot(gs_right[1])

        # Example colors like your percentile bars
    color_player = "#4CAF50"
    color_position = "#2196F3"

    draw_percent_bar(ax_bar1, event_name, player_pct, color_player)

    ax_bar1.set_title("Jugador (% del Equipo)", color="white", fontsize=12)
    
    # 3️⃣ Slim bar graphs: player % of position
    ax_bar2 = fig.add_subplot(gs_right[2])

    draw_percent_bar(ax_bar2, position_mode, pos_pct, color_position)
    ax_bar2.set_title(f"Jugador (% Posición: {position_mode})", color="white", fontsize=12)

    # 4️⃣ Summary text below bars
    ax_text3 = fig.add_subplot(gs_right[3])

    ax_text3.text(0.5, 0.75,
                 f"Totales de {event_name}",
                 color="gold", fontsize=16, ha="center", va="top", fontweight="bold")
    

    ax_text3.text(0.5, 0.35,
                 f"Equipo: {team_total}    Posición: {pos_total}    Jugador: {player_total}",
                 color="white", fontsize=12, ha="center", va="top")
    ax_text3.axis('off')

    fig.patch.set_facecolor("#0F0F0F")
    return fig

def draw_percent_bar(ax, label, value, color):
    """
    Simple horizontal bar from 0 to 100 with fill according to value
    """
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([0])
    ax.set_yticklabels([label], color="white", fontsize=12)
    ax.invert_yaxis()
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', length=0)

    # Background track
    ax.barh(0, 100, color="#222222", height=1, zorder=1)

    # Foreground fill
    ax.barh(0, value, color=color, height=0.95, zorder=2)

    # Percentage text at the end of the fill
    ax.text(value + 1, 0, f"{value}%", va="center", ha="left",
            color="white", fontsize=10, fontweight='bold')



# ================================
# MAX CARDS

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

def make_player_stat_card(ax, title, name, dorsal, value):
    """
    Draws a single player card in the given axis.
    Now with:
    - Black background
    - White text for title + name
    - Green number
    """

    # ---- BLACK CARD BACKGROUND ----
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- TITLE (white) ----
    ax.text(
        0.5, 0.95, title,
        ha="center", va="top",
        fontsize=14, fontweight="bold",
        color="white"
    )

    # ---- PLAYER IMAGE ----
    img_path = f"img/players/{dorsal}.png"

    if os.path.exists(img_path):
        img = Image.open(img_path).resize((160, 160))
        imagebox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(imagebox, (0.5, 0.60), frameon=False)
        ax.add_artist(ab)
    else:
        ax.text(
            0.5, 0.60, "No Photo",
            ha="center", va="center",
            fontsize=12,
            color="white"
        )

    # ---- PLAYER NAME (white) ----
    ax.text(
        0.5, 0.28, name,
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color="white"
    )

    # ---- VALUE (green — unchanged) ----
    ax.text(
        0.5, 0.10, str(value),
        ha="center", va="center",
        fontsize=20, fontweight="bold",
        color="#00A65A"
    )

def get_top_player_unified(df_events, df_m2, df_source, filter_col, filter_value,
                           name_col="Nombre", dorsal_col="Jugadores"):
    """
    df_source = 'm1' or 'm2'
    If m1 → filter df_events
    If m2 → read df_m2 column directly
    Tie-breaking: first appearance in dataframe
    Returns: (name, dorsal, count)
    """

    # ---------- CASE 1: USE M1 (df_events) ----------
    if df_source == "m1":

        # ---- GK modes ----
        if filter_col == "GK_TOTAL":
            df_f = df_events[df_events["Acción"] == "Arquero"]

        elif filter_col == "GK_RESULT":
            df_f = df_events[
                (df_events["Acción"] == "Arquero") &
                (df_events["Resultado"] == filter_value)
            ]

        # ---- Normal m1 filter ----
        else:
            df_f = df_events[df_events[filter_col] == filter_value]

        # ---- EMPTY CHECK (COMMON) ----
        if df_f.empty:
            return None, None, 0

        # ---- PRESERVE ROW ORDER ----
        df_f = df_f.reset_index().rename(columns={"index": "orig_order"})

        counts = (
            df_f.groupby([name_col, dorsal_col], as_index=False)
                .agg(
                    count=("orig_order", "count"),
                    first_appearance=("orig_order", "min")
                )
        )

        top = (
            counts.sort_values(
                by=["count", "first_appearance"],
                ascending=[False, True]
            ).iloc[0]
        )

        return top[name_col], top[dorsal_col], int(top["count"])

    # ---------- CASE 2: USE M2 ----------
    elif df_source == "m2":

        df_f = df_m2[df_m2[filter_col] == filter_value]

        if df_f.empty:
            return None, None, 0

        df_f = df_f.reset_index().rename(columns={"index": "orig_order"})

        counts = (
            df_f.groupby([name_col, dorsal_col], as_index=False)
                .agg(
                    count=("orig_order", "count"),
                    first_appearance=("orig_order", "min")
                )
        )

        top = (
            counts.sort_values(
                by=["count", "first_appearance"],
                ascending=[False, True]
            ).iloc[0]
        )

        return top[name_col], top[dorsal_col], int(top["count"])

    # ---------- SAFETY NET ----------
    return None, None, 0

ALL_STATS = [
    # Offensive stats
    ("Tiros",                  "m1", "Acción", "Tiro Ferro"),
    ("Goles",                  "m1", "Resultado", "Gol"),
    ("Asistencia",             "m1", "Acción", "Asistencia"),
    ("Pases Claves",           "m1", "Acción", "Pase Clave"),
    ("1 VS 1 Ofensivo",        "m1", "Acción", "1 VS 1 Ofensivo"),
    ("Pérdida",                "m1", "Acción", "Pérdida"),
    ("Faltas Recibidas",       "m1", "Acción", "Falta Recibida"),
    ("MT Ofensivo",            "m2", "Acción", "MT Ofensivo"),
    ("Transiciones Ofensivas", "m2", "Acción", "Transición Ofensiva"),
    # Defensive stats
    ("1 VS 1 Defensivo",        "m1", "Acción", "1 VS 1 Defensivo"),
    ("Recuperaciónes",          "m1", "Acción", "Recuperación"),
    ("Presión",                 "m1", "Acción", "Presión"),
    ("Falta Cometida",          "m1", "Acción", "Falta Cometida"),
    ("MT Defensivo",            "m2", "Acción", "MT Defensivo"),
    ("Transiciones Defensivas", "m2", "Acción", "Transición Defensiva"),
    ("Sanciones",               "m1", "Acción", "Sanción Ferro"),
]


OFENSIVO_STATS = [
    ("Tiros",                  "m1", "Acción", "Tiro Ferro"),
    ("Goles",                  "m1", "Resultado", "Gol"),
    ("Asistencia",                  "m1", "Acción", "Asistencia"),
    ("Pases Claves",           "m1", "Acción", "Pase Clave"),
    ("1 VS 1 Ofensivo",        "m1", "Acción", "1 VS 1 Ofensivo"),
    ("Pérdida",                "m1", "Acción", "Pérdida"),
    ("Faltas Recibidas",       "m1", "Acción", "Falta Recibida"),
    ("MT Ofensivo",            "m2", "Acción", "MT Ofensivo"),
    ("Transiciones Ofensivas", "m2", "Acción", "Transición Ofensiva"),

]



DEFENSIVO_STATS = [
    ("1 VS 1 Defensivo",        "m1", "Acción", "1 VS 1 Defensivo"),
    ("Recuperaciónes",          "m1", "Acción", "Recuperación"),
    ("Presión",                 "m1", "Acción", "Presión"),
    ("Falta Cometida",          "m1", "Acción", "Falta Cometida"),
    ("MT Defensivo",            "m2", "Acción", "MT Defensivo"),
    ("Transiciones Defensivas", "m2", "Acción", "Transición Defensiva"),
    ("Sanciones",               "m1", "Acción", "Sanción Ferro"),
]


import math

def make_cards_grid(df_events, df_m2, stats_list, title, exclude_arqueros=True):

    if exclude_arqueros:
        df_events = df_events[df_events["Posicion"] != "Arquero"]
        df_m2 = df_m2[df_m2["Posicion"] != "Arquero"]

    n_stats = len(stats_list)
    cols = 5
    rows = math.ceil(n_stats / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 5 * rows))
    fig.patch.set_facecolor("black")

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.98, color="white")

    idx = 0

    for r in range(rows):
        for c in range(cols):

            ax = axes[r][c]

            if idx >= n_stats:
                ax.axis("off")
                ax.set_facecolor("black")
                continue

            stat_title, df_source, col, value = stats_list[idx]

            # unified extractor
            name, dorsal, count = get_top_player_unified(
                df_events, df_m2,
                df_source, col, value
            )

            if name is None:
                name, dorsal, count = "N/A", "default", 0

            make_player_stat_card(ax, stat_title, name, dorsal, count)

            idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# GK STATS

GK_STATS = [            # Acción == Arquero (all resultados)
    ("Goles Encajados", "m1", "GK_RESULT", "Gol Recibido"),
    ("Atajas",          "m1", "GK_RESULT", "Ataja"),
    ("Rebotes",         "m1", "GK_RESULT", "Rebote"),
    ("Sexta",           "m1", "GK_RESULT", "Sexta"),
    ("Penal",           "m1", "GK_RESULT", "Penal"),
]


keeper_actions = ['Ataja', 'Rebote', 'Gol Recibido']


def plot_gk_action_breakdown(df_b1):

    df_gk = df_b1[
        (df_b1["Acción"] == "Arquero") &
        (df_b1["Resultado"].isin(keeper_actions))
    ]

    fig, ax = plt.subplots(figsize=(10.5, 1.6))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#0F0F0F")

    if df_gk.empty:
        ax.text(
            0.5, 0.5,
            "Sin acciones de arquero",
            ha="center", va="center",
            color="white", fontsize=12
        )
        ax.axis("off")
        return fig

    # ---------- ORDER + COUNTS ----------
    order = ["Ataja", "Rebote", "Gol Recibido"]
    counts = df_gk["Resultado"].value_counts().reindex(order, fill_value=0)
    total = counts.sum()

    left = 0

    # ---------- DRAW SEGMENTS ----------
    for res, val in counts.items():

        if val == 0:
            continue

        pct = (val / total) * 100
        color = RESULT_COLORS.get(res, "gray")

        # Rounded bar
        bar = FancyBboxPatch(
            (left, -0.35),
            pct, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            linewidth=1,
            edgecolor="#111111",
            facecolor=color
        )
        ax.add_patch(bar)

        # Percentage (big)
        ax.text(
            left + pct / 2,
            0.08,
            f"{pct:.0f}%",
            ha="center", va="center",
            color="white",
            fontsize=14,
            fontweight="bold"
        )

        # Count (small)
        ax.text(
            left + pct / 2,
            -0.18,
            f"{res}",
            ha="center", va="center",
            color="#DDDDDD",
            fontsize=9
        )

        left += pct

        # Separator line
        ax.plot([left, left], [-0.35, 0.35],
                color="#111111", linewidth=1)

    # ---------- TITLES ----------
    ax.text(
        0, 0.75,
        "Arquero — Distribución de Resultados",
        fontsize=13,
        fontweight="bold",
        color="white",
        ha="left"
    )

    ax.text(
        0, -0.85,
        f"Total acciones: {total}",
        fontsize=10,
        color="#BBBBBB",
        ha="left"
    )

    # ---------- CLEAN AXES ----------
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    return fig

def make_gk_cards_grid(df_events, df_m2):
    df_events = df_events[df_events["Posicion"] == "Arquero"]
    df_m2 = df_m2[df_m2["Posicion"] == "Arquero"]

    fig = make_cards_grid(df_events, df_m2, GK_STATS, title="Máximos – Arquero", exclude_arqueros=False)
    return fig


TIPO_ACCION = {
    # --- Offensive / Build-up ---
    "Paralela": "dodgerblue",
    "Diagonal": "limegreen",
    "Salto Linea": "orange",
    "Puerta Atras": "deepskyblue",
    "Pared": "gold",
    "Pivot": "purple",
    "Descarga Pivot": "mediumorchid",
    "Corte": "orangered",
    "Tiro": "darkorange",
    "Pelota Parada": "goldenrod",

    # --- Defensive actions ---
    "Doble Marca": "slateblue",
    "Cobertura": "teal",
    "Cambio Marca": "steelblue",
    "Cambio de Marca": "steelblue",  # same action, different spelling
    "Repliegue": "royalblue",
    "Recuperacion": "green",
    "Lectura Salto": "olive",

    # --- Negative / Risk ---
    "Perdida": "crimson",

    # --- Defensive outcome ---
    "Gol": "red",   # conceded goal (defensive perspective)

    # --- Fallback ---
    None: "gray"
}

def plot_gk_mt_donuts(df_m2):
    df_gk = df_m2[df_m2["Posicion"] == "Arquero"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("#0F0F0F")

    for ax, mt_type, title in zip(
        axes,
        ["MT Ofensivo", "MT Defensivo"],
        ["MT Ofensivo (GK)", "MT Defensivo (GK)"]
    ):
        ax.set_facecolor("#0F0F0F")

        data = df_gk[df_gk["Acción"] == mt_type]

        if data.empty:
            ax.text(
                0.5, 0.5, "Sin datos",
                ha="center", va="center",
                color="white", fontsize=12
            )
            ax.axis("off")
            continue

        counts = data["Tipo_de_Accion"].value_counts()
        total = counts.sum()

        labels = counts.index.tolist()
        values = counts.values
        colors = [TIPO_ACCION.get(r, "gray") for r in labels]

        # --- Donut ---
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.38, edgecolor="#0F0F0F"),
            autopct=lambda p: f"{p:.0f}%",
            pctdistance=0.78,
            labeldistance=1.08,
            textprops=dict(color="white")
        )

        # --- Smaller labels ---
        for t in texts:
            t.set_fontsize(9)

        # --- Percent text styling ---
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")

        # --- Center total ---
        ax.text(
            0, 0.05,
            f"{total}",
            ha="center", va="center",
            fontsize=20,
            fontweight="bold",
            color="white"
        )

        ax.text(
            0, -0.18,
            "acciones",
            ha="center", va="center",
            fontsize=10,
            color="#BBBBBB"
        )

        # --- Title ---
        ax.set_title(
            title,
            color="white",
            fontsize=13,
            fontweight="bold",
            pad=14
        )

        ax.axis("equal")

    plt.tight_layout()
    return fig


# ============================================================
#  PLAYER CARD
# ============================================================

# CONVERT SECONDS TO MM:SS

def percentile_color(pct):
    if pct >= 90:
        return "#00E676"   # elite green
    elif pct >= 70:
        return "#69DB7C"   # good
    elif pct >= 40:
        return "#FFC107"   # average
    else:
        return "#FF6B6B"   # weak

def compute_stat_value(df_events, df_m2, player_name, stat_def):
    stat_name, df_source, col, value = stat_def

    if df_source == "m1":
        df_p = df_events[df_events["Nombre"] == player_name]

        if col == "Resultado":
            return df_p[df_p["Resultado"] == value].shape[0]
        else:
            return df_p[df_p[col] == value].shape[0]

    elif df_source == "m2":
        df_p = df_m2[df_m2["Nombre"] == player_name]
        return df_p[df_p[col] == value].shape[0]

    return 0


def compute_team_percentile(df_events, df_m2, player_name, stat_def):
    stat_name, df_source, col, value = stat_def

    df = df_events if df_source == "m1" else df_m2

    players = df["Nombre"].dropna().unique()
    values = []

    for p in players:
        v = compute_stat_value(df_events, df_m2, p, stat_def)
        values.append(v)

    player_value = compute_stat_value(df_events, df_m2, player_name, stat_def)

    if len(values) == 0:
        return player_value, 0

    percentile = (
        sum(v <= player_value for v in values) / len(values)
    ) * 100

    return player_value, round(percentile, 1)

def get_player_stat_table(df_events, df_m2, player_name):

    rows = []

    for block, stats in [
        ("OFENSIVO", OFENSIVO_STATS),
        ("DEFENSIVO", DEFENSIVO_STATS)
    ]:
        for stat in stats:
            stat_name = stat[0]
            value, pct = compute_team_percentile(
                df_events, df_m2, player_name, stat
            )

            rows.append({
                "Block": block,
                "Stat": stat_name,
                "Value": value,
                "Percentile": pct
            })

    return pd.DataFrame(rows)

def draw_percentile_bar(ax, x, y, width, height, pct):

    # Background bar
    ax.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0,
            facecolor="#2A2A2A",
            clip_on=False   # 🔥 KEY
        )
    )

    # Filled bar
    fill_w = width * (pct / 100)

    ax.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            fill_w,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0,
            facecolor=percentile_color(pct),
            clip_on=False   # 🔥 KEY
        )
    )

def draw_player_stat_block(ax, df_stats, start_y=0.62):

    y = start_y
    row_gap = 0.055

    for block in ["OFENSIVO", "DEFENSIVO"]:

        # ---- SECTION HEADER ----
        ax.text(
            0.05, y,
            block,
            fontsize=16,
            fontweight="bold",
            color="#69DB7C" if block == "OFENSIVO" else "#FF6B6B",
            ha="left",
            va="center"
        )
        y -= row_gap * 1.2

        block_df = df_stats[df_stats["Block"] == block]

        for _, row in block_df.iterrows():

            # STAT NAME
            ax.text(
                0.05, y,
                row["Stat"],
                fontsize=13,
                color="#CCCCCC",
                ha="left",
                va="center"
            )

            # VALUE (middle column)
            ax.text(
                0.60, y,
                f'{row["Value"]}',
                fontsize=14,
                color="white",
                ha="right",
                va="center"
            )

            # ---- MINI BAR ----
            draw_percentile_bar(
                ax,
                x=0.70,
                y=y,
                width=0.20,
                height=0.018,
                pct=row["Percentile"]
            )

            # ---- PERCENT TEXT (LEFT-ALIGNED, SMALLER, WHITE) ----
            ax.text(
                0.70, y,
                f'{row["Percentile"]:.0f} %',
                fontsize=11,
                fontweight="bold",
                color="white",
                ha="left",
                va="center"
            )

            y -= row_gap

        y -= row_gap * 0.6

def format_mmss(seconds):
    if seconds is None or pd.isna(seconds):
        return None
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"

def get_player_card_data(player_name, df_b1, df_minutes):

    df_p = df_b1[df_b1["Nombre"] == player_name]

    dorsal = int(df_p["Jugadores"].iloc[0])
    position = df_p["Posicion"].iloc[0]

    # Standard goals & assists
    goals = len(df_p[df_p["Resultado"].str.lower() == "gol"]) if "Resultado" in df_b1.columns else 0
    assists = len(df_p[df_p["Acción"] == "Asistencia"])

    # Goalkeeper-specific stats
    atajas = len(df_p[(df_p["Acción"] == "Arquero") & (df_p["Resultado"] == "Ataja")])
    goles_recibidos = len(df_p[(df_p["Acción"] == "Arquero") & (df_p["Resultado"] == "Gol Recibido")])

    # Sanciones
    sanciones = df_p[df_p["Acción"] == "Sanción Ferro"]
    sanciones_amarilla = sanciones[sanciones["Resultado"] == "Amarilla"].shape[0]
    sanciones_roja = sanciones[sanciones["Resultado"] == "Roja"].shape[0]

    # Extra KPIs
    pases_claves = len(df_p[df_p["Acción"].str.contains("Pase Clave", case=False, na=False)])
    uno_v_uno_def = len(df_p[df_p["Acción"].str.contains("1 VS 1 Defensivo", case=False, na=False)])
    faltas_recibidas = len(df_p[df_p["Acción"].str.contains("Falta Recibida", case=False, na=False)])
    faltas_cometidas = len(df_p[df_p["Acción"].str.contains("Falta Cometida", case=False, na=False)])

    # Minutes
    df_m = df_minutes[df_minutes["Jugadores"] == dorsal]
    total_matches = len(df_m)
    total_minutes = df_m["Tiempo_Efectivo"].sum()
    avg_minutes = total_minutes / total_matches if total_matches > 0 else 0

    return {
        "name": player_name,
        "dorsal": dorsal,
        "position": position,

        # Offensive (for non-goalkeepers)
        "goals": goals,
        "assists": assists,

        # Goalkeeper stats
        "atajas": atajas,
        "goles_recibidos": goles_recibidos,

        # Cards
        "rojas": sanciones_roja,
        "amarillas": sanciones_amarilla,

        # Extra stats
        "pases_claves": pases_claves,
        "uno_v_uno_def": uno_v_uno_def,
        "faltas_recibidas": faltas_recibidas,
        "faltas_cometidas": faltas_cometidas,

        "total_matches": total_matches,
        "total_minutes": total_minutes,
        "avg_minutes": avg_minutes
    }

def draw_player_card(
    img_path,
    name,
    dorsal,
    position,
    matches,
    total_minutes,
    avg_minutes,
    goals,
    assists,
    sanciones_roja,
    sanciones_amarilla,
    pases_clave,
    uno_v_uno_def,
    faltas_recibidas,
    faltas_cometidas,
    atajas,
    goles_recibidos,

    # 🔥 NEW (REQUIRED)
    df_events,
    df_m2,

    figsize=(10, 5)
):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.axis("off")

    # ==========================
    # PLAYER IMAGE
    # ==========================
    try:
        player_img = Image.open(img_path)
        zoom = 6
        w, h = player_img.size
        player_img = player_img.resize((int(w * zoom), int(h * zoom)))
    except:
        player_img = Image.new("RGB", (300, 300), color=(40, 40, 40))

    ax_img = fig.add_axes([0.05, 0.1, 0.30, 0.8])
    ax_img.imshow(np.array(player_img))
    ax_img.axis("off")

    # ==========================
    # STATS CONTAINER
    # ==========================
    ax_stats = fig.add_axes([0.38, 0.02, 0.6, 0.96])
    ax_stats.axis("off")

    bg = FancyBboxPatch(
        (0, -0.33), 1, 1.37,   # ⬅️ extend vertically
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1,
        edgecolor="#2A2A2A",
        facecolor="#1A1A1A",
        clip_on=False       # ⬅️ THIS IS THE KEY
    )
    ax_stats.add_patch(bg)


    # ==========================
    # HEADER
    # ==========================
    y = 0.92
    gap = 0.08

    ax_stats.text(
        0.05, y, name,
        fontsize=20,
        color="white",
        fontweight="bold",
        ha="left",
        va="center"
    )

    ax_stats.text(
        0.95, y, f"#{dorsal}",
        fontsize=22,
        color="white",
        fontweight="bold",
        ha="right",
        va="center"
    )

    y -= gap * 0.8

    ax_stats.text(
        0.05, y, position,
        fontsize=14,
        color="#BBBBBB",
        ha="left",
        va="center"
    )

    y -= gap

    # ==========================
    # BASIC INFO ROW
    # ==========================
    ax_stats.text(0.05, y, "Partidos", fontsize=13, color="#888888", ha="left")
    ax_stats.text(0.35, y, f"{matches}", fontsize=14, color="white", ha="left")

    if position.lower() == "arquero":
        ax_stats.text(0.55, y, "Atajas", fontsize=13, color="#888888", ha="left")
        ax_stats.text(0.80, y, f"{atajas}", fontsize=14, color="white", ha="left")
    else:
        ax_stats.text(0.55, y, "Goles", fontsize=13, color="#888888", ha="left")
        ax_stats.text(0.80, y, f"{goals}", fontsize=14, color="white", ha="left")

    y -= gap * 1.2

    # ==========================
    # ADVANCED STAT TABLE
    # ==========================
    df_stats = get_player_stat_table(df_events, df_m2, name)
    draw_player_stat_block(ax_stats, df_stats, start_y=y)

    return fig



# =========================================================
# RADAR CHARTS – CLEAN COMPUTATION ENGINE (WITH ARQUERO)
# =========================================================

def compute_player_kpis(df_events, df_minutes, player_name):
    """
    Compute all KPIs used in the dual radar for a given player.
    Handles:
      - Any position
      - Special defensive metrics for Arquero
      - Players with zero events (returns all-zero dict)
    """

    # Filter events for this player
    df_p = df_events[df_events["Nombre"] == player_name]

    # Minutes dataframe for this player
    df_m = df_minutes[df_minutes["Nombre"] == player_name]
    total_seconds = df_m["Tiempo_Efectivo"].sum()
    minutos = total_seconds / 60 if total_seconds > 0 else 0
    factor_40 = 40 / minutos if minutos > 0 else 0

    # If no events: return 0s for all KPIs
    if df_p.empty:
        return {
            # Offensive block
            "Tiros/40": 0,
            "Tiros al Arco/40": 0,
            "Goles": 0,
            "Pases Claves/40": 0,
            "Asistencias": 0,
            "Pérdida": 0,
            "Falta Recibida": 0,
            "1v1 Of Ganado %": 0,

            # Defensive (player)
            "1v1 Def Ganado %": 0,
            "1v1 Def Total/40": 0,
            "Recuperaciones/40": 0,
            "Presiones/40": 0,
            "Sanciones": 0,

            # Defensive (GK-specific)
            "Tiro Rival/40": 0,
            "Atajadas/40": 0,
            "Goles Encajados": 0,
            "Rebotes": 0,
        }

    # Get position safely
    if "Posicion" in df_p.columns and not df_p["Posicion"].isna().all():
        position = str(df_p["Posicion"].iloc[0]).strip()
    else:
        position = "Jugador"

    # Helper filters
    def count(action):
        return df_p[df_p["Acción"] == action].shape[0]

    def count_result(action, result_list):
        tmp = df_p[df_p["Acción"] == action]
        return tmp[tmp["Resultado"].isin(result_list)].shape[0]

    # -----------------------------
    #  COMMON OFENSIVE KPIs
    # -----------------------------
    tiros = count("Tiro Ferro")
    tiros_arco = count_result("Tiro Ferro", ["Gol", "Atajado"])
    goles = df_p[df_p["Resultado"] == "Gol"].shape[0]
    pases_clave = count("Pase Clave")
    asistencias = count("Asistencia")
    perdida = count("Pérdida")
    falta_recibida = count("Falta Recibida")

    of_tot = count("1 VS 1 Ofensivo")
    of_gan = count_result("1 VS 1 Ofensivo", ["Ganado"])
    pct_of = (of_gan / of_tot * 100) if of_tot > 0 else 0

    # -----------------------------
    #  DEFENSIVE / GK KPIs
    # -----------------------------
    # Player defensive (non-GK)
    def_tot = count("1 VS 1 Defensivo")
    def_gan = count_result("1 VS 1 Defensivo", ["Ganado"])
    pct_def = (def_gan / def_tot * 100) if def_tot > 0 else 0

    rec = count("Recuperación")
    pres = count("Presión")
    sanciones = df_p[df_p["Sanción"] == "Ferro"].shape[0]

    # GK specific events
    tiros_rival = count("Arquero")
    atajadas = count_result("Arquero", ["Ataja"])
    goles_recibidos = count_result("Arquero", ["Gol Recibido"])
    rebotes = count_result("Arquero", ["Rebote"])

    # Build full dict (always same keys!)
    stats = {
        # Offensive block
        "Tiros/40": tiros * factor_40,
        "Tiros al Arco/40": tiros_arco * factor_40,
        "Goles": goles,
        "Pases Claves/40": pases_clave * factor_40,
        "Asistencias": asistencias,
        "Pérdidas/40": perdida * factor_40,
        "Falta Recibida": falta_recibida,
        "1v1 Of Ganado %": pct_of,

        # Defensive (generic player)
        "1v1 Def Ganado %": pct_def,
        "1v1 Def Total/40": def_tot * factor_40,
        "Recuperaciones/40": rec * factor_40,
        "Presiones/40": pres * factor_40,
        "Sanciones": sanciones,

        # Defensive (GK-specific)
        "Tiro Rival/40": tiros_rival * factor_40,
        "Atajadas/40": atajadas * factor_40,
        "Goles Encajados": goles_recibidos,
        "Rebotes": rebotes,
    }

    return stats

def compute_team_max(df_events, df_minutes):
    """
    Computes the maximum value per KPI across the whole team.
    Uses compute_player_kpis, so it already handles Arquero + players with 0 events.
    """

    players = df_events["Nombre"].dropna().unique()

    if len(players) == 0:
        # Return a zero dict compatible with compute_player_kpis
        return compute_player_kpis(df_events.iloc[0:0], df_minutes.iloc[0:0], "")

    # Initialize with first player's keys
    first_stats = compute_player_kpis(df_events, df_minutes, players[0])
    team_max = {k: 0 for k in first_stats.keys()}

    # Iterate players
    for p in players:
        stats = compute_player_kpis(df_events, df_minutes, p)
        for k, v in stats.items():
            # Percent KPIs we will treat separately
            if k in ["1v1 Of Ganado %", "1v1 Def Ganado %"]:
                continue
            team_max[k] = max(team_max.get(k, 0), v)

    # Percent KPIs fixed to 100 for scaling
    team_max["1v1 Of Ganado %"] = 100
    team_max["1v1 Def Ganado %"] = 100

    # Avoid zeros for max (Radar can't scale from 0)
    for k in team_max:
        if team_max[k] == 0:
            team_max[k] = 1

    return team_max

def compute_position_avg(df_events, df_minutes, position):
    """
    Average KPIs for all players with the same Posicion.
    If no players are found for that position, falls back to team first player.
    """

    df_pos = df_events[df_events["Posicion"] == position]
    players_pos = df_pos["Nombre"].dropna().unique()

    # If no one in this position, fall back to first team player
    if len(players_pos) == 0:
        all_players = df_events["Nombre"].dropna().unique()
        if len(all_players) == 0:
            # Return empty-zero dict if absolutely nothing exists
            return compute_player_kpis(df_events.iloc[0:0], df_minutes.iloc[0:0], "")
        return compute_player_kpis(df_events, df_minutes, all_players[0])

    agg = None
    count = 0

    for p in players_pos:
        kpis = compute_player_kpis(df_events, df_minutes, p)
        if agg is None:
            agg = {k: 0 for k in kpis}
        for k in kpis:
            agg[k] += kpis[k]
        count += 1

    avg = {k: (agg[k] / count if count > 0 else 0) for k in agg}
    return avg


from mplsoccer import Radar, FontManager, grid

OFFENSIVE_PARAMS = [
    "Tiros/40",
    "Tiros al Arco/40",
    "Goles",
    "Pases Claves/40",
    "Asistencias",
    "Pérdidas/40",
    "Falta Recibida",
    "1v1 Of Ganado %",
]

DEFENSIVE_PARAMS_PLAYER = [
    "1v1 Def Ganado %",
    "1v1 Def Total/40",
    "Recuperaciones/40",
    "Presiones/40",
    "Sanciones",
]

DEFENSIVE_PARAMS_ARQUERO = [
    "Tiro Rival/40",
    "Atajadas/40",
    "Goles Encajados",
    "Rebotes",
    "Recuperaciones/40",
    "Sanciones",
]

def plot_dual_radar_with_grid(
    selected_player,
    position_name,
    player_kpis,
    pos_avg,
    team_max,
    comparison_kpis=None  # dict of extra players
):
    """
    Simplified hybrid radar:
    - If no extra players: main player vs positional average
    - If extra players: append additional players to radar
    """
    from mplsoccer import Radar, FontManager
    import matplotlib.pyplot as plt

    fm = FontManager(
        'https://github.com/google/fonts/raw/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf'
    )

    # ----- OFFENSIVE -----
    OFF = [player_kpis[k] for k in OFFENSIVE_PARAMS]
    POS_OFF = [pos_avg[k] for k in OFFENSIVE_PARAMS]
    MAX_OFF = [team_max[k] for k in OFFENSIVE_PARAMS]

    # ----- DEFENSIVE -----
    if str(position_name).strip().lower() == "arquero":
        DEF_PARAMS = DEFENSIVE_PARAMS_ARQUERO
    else:
        DEF_PARAMS = DEFENSIVE_PARAMS_PLAYER

    DEF = [player_kpis[k] for k in DEF_PARAMS]
    POS_DEF = [pos_avg[k] for k in DEF_PARAMS]
    MAX_DEF = [team_max[k] for k in DEF_PARAMS]

    # ----- Figure -----
    fig, axs = grid(
        figheight=14,
        grid_height=0.80,
        title_height=0.12,
        title_space=0,
        grid_key='radar',
        axis=False,
        ncols=2
    )

    ax_left = axs['radar'][0]
    ax_right = axs['radar'][1]

    # ================= OFFENSIVE RADAR =================
    radar_off = Radar(
        params=OFFENSIVE_PARAMS,
        min_range=[0]*len(OFFENSIVE_PARAMS),
        max_range=MAX_OFF,
        round_int=[False]*len(OFFENSIVE_PARAMS),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )
    radar_off.setup_axis(ax=ax_left)
    radar_off.draw_circles(ax=ax_left, facecolor="#222222", edgecolor="#555555")

    # main player vs positional average
    poly = radar_off.draw_radar_compare(
        OFF, POS_OFF, ax=ax_left,
        kwargs_radar={'facecolor': '#1A78CF80', 'edgecolor': '#1A78CF', 'linewidth': 2},
        kwargs_compare={'facecolor': '#69DB7C80', 'edgecolor': '#69DB7C', 'linewidth': 2},
    )
    _, _, v1, v2 = poly
    ax_left.scatter(v1[:,0], v1[:,1], c='#1A78CF', edgecolors='white', s=120)
    ax_left.scatter(v2[:,0], v2[:,1], c='#69DB7C', edgecolors='white', s=120)

    # append additional players if any
    if comparison_kpis:
        colors = ["#FFD166", "#FF6B6B", "#4D96FF", "#9B5DE5"]
        for i, (player, kpis) in enumerate(list(comparison_kpis.items())[:4]):
            OFF_X = [kpis[k] for k in OFFENSIVE_PARAMS]
            radar_off.draw_radar(
                OFF_X,
                ax=ax_left,
                kwargs_radar={'facecolor': colors[i]+'80', 'edgecolor': colors[i], 'linewidth':2},
                kwargs_rings={'alpha':0}  # hide rings for extra players
            )

    radar_off.draw_param_labels(ax=ax_left, fontsize=16, fontproperties=fm.prop, color="white")
    radar_off.draw_range_labels(ax=ax_left, fontsize=14, fontproperties=fm.prop, color="#CCCCCC")
    ax_left.text(0.5, 1.02, "OFENSIVO", transform=ax_left.transAxes, ha="center", va="center",
                 fontsize=22, color="#1A78CF", fontproperties=fm.prop)

    # ================= DEFENSIVE RADAR =================
    radar_def = Radar(
        params=DEF_PARAMS,
        min_range=[0]*len(DEF_PARAMS),
        max_range=MAX_DEF,
        round_int=[False]*len(DEF_PARAMS),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )
    radar_def.setup_axis(ax=ax_right)
    radar_def.draw_circles(ax=ax_right, facecolor="#222222", edgecolor="#555555")

    poly2 = radar_def.draw_radar_compare(
        DEF, POS_DEF, ax=ax_right,
        kwargs_radar={'facecolor': '#FF6B6B80', 'edgecolor': '#FF6B6B', 'linewidth': 2},
        kwargs_compare={'facecolor': '#69DB7C80', 'edgecolor': '#69DB7C', 'linewidth': 2},
    )
    _, _, v1_def, v2_def = poly2
    ax_right.scatter(v1_def[:,0], v1_def[:,1], c='#FF6B6B', edgecolors='white', s=120)
    ax_right.scatter(v2_def[:,0], v2_def[:,1], c='#69DB7C', edgecolors='white', s=120)

    if comparison_kpis:
        colors = ["#FFD166", "#FF6B6B", "#4D96FF", "#9B5DE5"]
        for i, (player, kpis) in enumerate(list(comparison_kpis.items())[:4]):
            DEF_X = [kpis[k] for k in DEF_PARAMS]
            radar_def.draw_radar(
                DEF_X,
                ax=ax_right,
                kwargs_radar={'facecolor': colors[i]+'80', 'edgecolor': colors[i], 'linewidth':2},
                kwargs_rings={'alpha':0}  # hide rings for extra players
            )

    radar_def.draw_param_labels(ax=ax_right, fontsize=16, fontproperties=fm.prop, color="white")
    radar_def.draw_range_labels(ax=ax_right, fontsize=14, fontproperties=fm.prop, color="#CCCCCC")
    ax_right.text(0.5, 1.02, "DEFENSIVO", transform=ax_right.transAxes, ha="center", va="center",
                  fontsize=22, color="#FF6B6B", fontproperties=fm.prop)

    # ===== Titles =====
    axs['title'].text(0.5, 0.75, selected_player, fontsize=30, fontproperties=fm.prop,
                      color="white", ha="center")
    axs['title'].text(0.5, 0.32,
                      f"Comparado con promedio de la posición: {position_name}",
                      fontsize=24, color="#69DB7C80", fontproperties=fm.prop, ha="center")

    fig.patch.set_facecolor("#0F0F0F")
    ax_left.set_facecolor("#0F0F0F")
    ax_right.set_facecolor("#0F0F0F")

    return fig


# ============================================================
#  PAGE RENDERING (FILTERS + PAGE UI)
# ============================================================
def render(df_1, df_2, df_3, df_tiempos):

    st.markdown("<h2 style='color:white;'>Análisis Individual</h2>", unsafe_allow_html=True)

    # ==========================
    # OFENSIVO CARDS
    # ==========================

    fig_of = make_cards_grid(df_1, df_2, ALL_STATS,
                            title="Máximos del Equipo")

    st.pyplot(fig_of, use_container_width=True)


    # ==========================
    # ARQUERO – OVERALL STATS
    # ==========================

    st.markdown("<h3 style='color:white;'>Arquero – Resumen Global</h3>",
                unsafe_allow_html=True)

    # Row 1: Stacked bar
    fig_gk_bar = plot_gk_action_breakdown(df_1)
    st.pyplot(fig_gk_bar, use_container_width=True)

    # Row 2: GK máximos
    fig_gk_cards = make_gk_cards_grid(df_1, df_2)
    st.pyplot(fig_gk_cards, use_container_width=True)

    # Row 3: MT donuts
    fig_gk_mt = plot_gk_mt_donuts(df_2)
    st.pyplot(fig_gk_mt, use_container_width=True)


    # =============================
    # 1. SELECT PLAYER
    # =============================
    st.markdown("<h3 style='color:white;'>Selecciona el jugador que quieres analizar</h3>", unsafe_allow_html=True)
    players = sorted(df_1["Nombre"].dropna().unique().tolist())
    selected_player = st.selectbox(
        "Seleccionar Jugador",
        options=["Todos"] + players,
        index=0
    )

    # Keep original team df for team computations
    df_team_original = df_1.copy()

    # Filter dataframes to selected player
    if selected_player != "Todos":
        df_player = df_1[df_1["Nombre"] == selected_player]
        df_tiempos_player = df_tiempos[df_tiempos["Nombre"] == selected_player]

        # =============================
        # 2. PLAYER CARD
        # =============================
        card_data = get_player_card_data(selected_player, df_player, df_tiempos_player)

        fig_card = draw_player_card(
            img_path=f"img/players/{card_data['dorsal']}.png",
            name=card_data["name"],
            dorsal=card_data["dorsal"],
            position=card_data["position"],
            matches=card_data["total_matches"],
            total_minutes=card_data["total_minutes"],
            avg_minutes=card_data["avg_minutes"],
            goals=card_data["goals"],
            assists=card_data["assists"],
            sanciones_roja=card_data["rojas"],
            sanciones_amarilla=card_data["amarillas"],
            pases_clave=card_data["pases_claves"],
            uno_v_uno_def=card_data["uno_v_uno_def"],
            faltas_recibidas=card_data["faltas_recibidas"],
            faltas_cometidas=card_data["faltas_cometidas"],
            atajas=card_data["atajas"],
            goles_recibidos=card_data["goles_recibidos"],

            # 🔥 NEW (REQUIRED)
            df_events=df_1,
            df_m2=df_2
        )



        st.pyplot(fig_card, use_container_width=True)

        # =====================================================
        #   DUAL RADAR (OFENSIVO / DEFENSIVO)
        # =====================================================

        st.markdown("<h3 style='color:white;margin-top:25px;'>Radar Comparativo</h3>",
                    unsafe_allow_html=True)

        # multiselect
        compare_players = st.multiselect(
            "Comparar con otros jugadores (máx. 4)",
            options=[p for p in df_team_original["Nombre"].unique() if p != selected_player],
            max_selections=4
        )

        # main + comparison KPIs
        player_kpis = compute_player_kpis(df_team_original, df_tiempos, selected_player)
        comparison_kpis = {p: compute_player_kpis(df_team_original, df_tiempos, p)
                        for p in compare_players}

        # positional average and max
        pos_avg = compute_position_avg(df_team_original, df_tiempos, card_data["position"])
        team_max = compute_team_max(df_team_original, df_tiempos)

        # call radar
        fig_radar = plot_dual_radar_with_grid(
            selected_player,
            card_data["position"],
            player_kpis,
            pos_avg,
            team_max,
            comparison_kpis=comparison_kpis
        )
        st.pyplot(fig_radar, use_container_width=True)



        # =============================
        # 3. EVENT TYPE SELECTION
        # =============================
        st.markdown("<h3 style='color:white;'>Seleccionar Evento</h3>", unsafe_allow_html=True)


        modo = st.radio(
            "Categoría",
            ["Ofensivo", "Defensivo"],
            horizontal=True,
            label_visibility="collapsed"
        )


        # Offensive and defensive event lists
        offensive_events = [
            "Tiro Ferro",
            "1 VS 1 Ofensivo",
            "Pase Clave",
            "Falta Recibida",
            "Asistencia",
            "Pérdida"
        ]

        defensive_events = [
            "Recuperación",
            "Presión",
            "Tiro Rival",
            "1 VS 1 Defensivo",
            "Falta Cometida",
            "Arquero"
        ]

        # =============================
        # 4. EVENT PICKER
        # =============================
        if modo == "Ofensivo":
            available_events = [ev for ev in offensive_events if ev in df_player["Acción"].unique()]
        else:
            available_events = [ev for ev in defensive_events if ev in df_player["Acción"].unique()]

        if len(available_events) == 0:
            st.warning("El jugador no tiene acciones en esta categoría.")
            return df_1, df_2

        selected_event = st.selectbox(
            "Selecciona un evento",
            options=available_events
        )

        # =============================
        # 5. UNIVERSAL 3-PANEL VISUALIZATION
        # =============================
        fig_summary = plot_event_summary_two_axes(
            df_player=df_player,
            df_team=df_team_original,
            event_name=selected_event,
            result_column="Resultado"
        )

        st.pyplot(fig_summary, use_container_width=True)




