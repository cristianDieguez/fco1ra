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



def plot_event_summary_matplotlib(
    df_player,
    df_team,
    event_name,
    result_column="Resultado",
    title="Resumen del Evento"
):
    """
    Universal 3-panel visualization for ANY offensive/defensive event.
    
    Panels:
    LEFT (2 rows): 
        - Player as % of team total
        - Player actions as % of position total

    CENTER:
        - Campograma with event actions (color + marker by RESULT)
        - Legend included

    RIGHT:
        - Pie chart showing % per result
        - No legend (campograma already includes it)
    """

    # ============================
    # 0. PREPARE DATA
    # ============================

    # TEAM actions of this event
    df_team_event = df_team[df_team["Acción"] == event_name]
    team_total = len(df_team_event)

    # PLAYER actions
    df_player_event = df_player[df_player["Acción"] == event_name]
    player_total = len(df_player_event)

    # Position mode for player
    if "Posicion" in df_player.columns and not df_player["Posicion"].isna().all():
        position_mode = df_player["Posicion"].mode().iloc[0]
    else:
        position_mode = "Desconocido"

    # Position group = all team players with same mode position
    df_team_position_group = df_team[df_team["Posicion"] == position_mode]
    df_pos_event = df_team_position_group[df_team_position_group["Acción"] == event_name]
    pos_total = len(df_pos_event)

    # --- Percentages ---
    # 1) Player as % of TEAM total
    if team_total > 0:
        player_pct = round((player_total / team_total) * 100, 1)
    else:
        player_pct = 0

    # 2) Player as % of POSITION total
    if pos_total > 0:
        pos_pct = round((player_total / pos_total) * 100, 1)
    else:
        pos_pct = 0

    # RESULT distributions (player)
    result_counts = df_player_event[result_column].value_counts()
    pie_labels = result_counts.index.tolist()
    pie_values = result_counts.values.tolist()
    pie_colors = [RESULT_COLORS.get(r, "gray") for r in pie_labels]

    # ============================
    # 1. CREATE MULTI-PANEL FIGURE
    # ============================

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, width_ratios=[1.2, 2.5, 1.3], wspace=0.25)

    # -------------------------------------------------------------
    # LEFT COLUMN (2 rows)
    # -------------------------------------------------------------
    gs_left = GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[0],
        hspace=0.35
    )

    # ------- Bar 1: Player % of Team -------
    ax1 = fig.add_subplot(gs_left[0])
    ax1.barh([event_name], [player_pct], color="#4CAF50")
    ax1.set_xlim(0, 100)
    ax1.set_title("Jugador (% del Equipo)", fontsize=12, color="white")
    ax1.set_xlabel("%", color="white")
    ax1.tick_params(colors="white")
    ax1.bar_label(ax1.containers[0], labels=[f"{player_pct}%"], color="white")

    # ------- Bar 2: Player % of Position -------
    ax2 = fig.add_subplot(gs_left[1])
    ax2.barh([position_mode], [pos_pct], color="#2196F3")
    ax2.set_xlim(0, 100)
    ax2.set_title(f"Jugador (% de la Posición: {position_mode})", fontsize=12, color="white")
    ax2.set_xlabel("%", color="white")
    ax2.tick_params(colors="white")
    ax2.bar_label(ax2.containers[0], labels=[f"{pos_pct}%"], color="white")

    # Helper text
    ax2.text(
        0.02,
        0.5,
        (
            f"Equipo: {team_total} acciones\n"
            f"Posición ({position_mode}): {pos_total} acciones\n"
            f"Jugador: {player_total} acciones"
        ),
        transform=ax2.transAxes,
        fontsize=10,
        color="white"
    )

    # -------------------------------------------------------------
    # CENTER COLUMN — SHOTMAP / ACTION MAP
    # -------------------------------------------------------------
    ax_center = fig.add_subplot(gs[1])
    draw_futsal_pitch_grid(ax_center)

    # Convert event coordinates according to event type
    df_plot = get_event_coordinates(df_player_event, event_name)

    # Plot points
    for resultado, group in df_plot.groupby(result_column):

        # If pass-type event → draw trajectory
        if len(group) > 0 and group["is_pass"].iloc[0]:
            for _, row in group.iterrows():
                ax_center.plot(
                    [row["X"], row["X2"]],
                    [row["Y"], row["Y2"]],
                    color=RESULT_COLORS.get(resultado, "gray"),
                    linewidth=2,
                    alpha=0.8
                )
                ax_center.scatter(
                    row["X"], row["Y"],
                    s=150,
                    marker="o",
                    color=RESULT_COLORS.get(resultado, "gray"),
                    edgecolor="black",
                    linewidth=0.7
                )
        else:
            # Non-pass → scatter point normally
            ax_center.scatter(
                group["X"],
                group["Y"],
                s=140,
                alpha=0.9,
                marker=RESULT_MARKERS.get(resultado, "o"),
                color=RESULT_COLORS.get(resultado, "gray"),
                edgecolor="black",
                linewidths=0.7,
                label=str(resultado)
            )

    ax_center.set_title(f"{event_name} – Campograma", color="white", fontsize=12)
    ax_center.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------
    # RIGHT COLUMN — PIE CHART
    # -------------------------------------------------------------
    ax_pie = fig.add_subplot(gs[2])

    if len(pie_values) > 0:
        ax_pie.pie(
            pie_values,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "white", "fontsize": 10}
        )
        ax_pie.set_title("Distribución del Resultado", color="white")
    else:
        ax_pie.text(0.5, 0.5, "Sin datos", color="white", ha="center", fontsize=12)

    fig.patch.set_facecolor("#0f0f0f")

    return fig


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

        df_f = df_events[df_events[filter_col] == filter_value]

        if df_f.empty:
            return None, None, 0

        # Preserve original row order
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

        # Preserve original row order
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


OFENSIVO_STATS = [
    ("Tiros",                  "m1", "Acción", "Tiro Ferro"),
    ("Goles",                  "m1", "Resultado", "Gol"),
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

def make_cards_grid(df_events, df_m2, stats_list, title):

    n_stats = len(stats_list)
    cols = 4
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


# ============================================================
#  PLAYER CARD
# ============================================================

# CONVERT SECONDS TO MM:SS
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
    figsize=(10,4)
):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.axis("off")

    # Load and zoom image
    try:
        player_img = Image.open(img_path)
        zoom = 6
        w, h = player_img.size
        player_img = player_img.resize((int(w*zoom), int(h*zoom)))
    except:
        player_img = Image.new("RGB", (300,300), color=(40,40,40))
    player_img = np.array(player_img)

    # Left image
    ax_img = fig.add_axes([0.05, 0.1, 0.30, 0.8])
    ax_img.imshow(player_img)
    ax_img.axis("off")

    # Right stats card (FIXED HEIGHT)
    ax_stats = fig.add_axes([0.38, 0.05, 0.57, 0.9])
    ax_stats.axis("off")

    bg = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1,
        edgecolor="#2A2A2A",
        facecolor="#1A1A1A"
    )
    ax_stats.add_patch(bg)

    # Spacing
    y = 0.92
    gap = 0.095

    # NAME + DORSAL
    ax_stats.text(0.05, y, name, fontsize=20, color="white",
                  weight="bold", ha="left", va="center")
    ax_stats.text(0.95, y, f"#{dorsal}", fontsize=22, color="white",
                  weight="bold", ha="right", va="center")
    y -= gap * 0.7

    # POSITION
    ax_stats.text(0.05, y, position, fontsize=14, color="#BBBBBB",
                  ha="left", va="center")
    y -= gap

    # MATCHES
    ax_stats.text(0.05, y, "Partidos", fontsize=14, color="#888888",
                  ha="left")
    ax_stats.text(0.65, y, f"{matches}", fontsize=16, color="white",
                  ha="left")
    y -= gap * 0.7

    # # TOTAL MINUTES
    # ax_stats.text(0.05, y, "Minutos Totales", fontsize=14, color="#888888",
    #               ha="left")
    # ax_stats.text(0.65, y, f"{format_mmss(total_minutes)}", fontsize=16,
    #               color="white", ha="left")
    # y -= gap * 0.7

    # # AVG MINUTES
    # ax_stats.text(0.05, y, "Promedio / Partido", fontsize=14, color="#888888",
    #               ha="left")
    # ax_stats.text(0.65, y, f"{format_mmss(avg_minutes)}", fontsize=16,
    #               color="white", ha="left")
    # y -= gap * 1.2

    # GOLES / ATAJAS depending on position
    if position.lower() == "arquero":
        ax_stats.text(0.05, y, "Atajas", fontsize=14, color="#888888", ha="left")
        ax_stats.text(0.65, y, f"{atajas}", fontsize=16, color="white", ha="left")
        y -= gap * 0.7

        ax_stats.text(0.05, y, "Goles Recibidos", fontsize=14, color="#888888", ha="left")
        ax_stats.text(0.65, y, f"{goles_recibidos}", fontsize=16, color="white", ha="left")
        y -= gap * 0.7

    else:
        ax_stats.text(0.05, y, "Goles", fontsize=14, color="#888888", ha="left")
        ax_stats.text(0.65, y, f"{goals}", fontsize=16, color="white", ha="left")
        y -= gap * 0.7

        ax_stats.text(0.05, y, "Asistencias", fontsize=14, color="#888888", ha="left")
        ax_stats.text(0.65, y, f"{assists}", fontsize=16, color="white", ha="left")
        y -= gap * 0.7

    # NEW — Pases Claves
    ax_stats.text(0.05, y, "Pases Claves", fontsize=14, color="#888888", ha="left")
    ax_stats.text(0.65, y, f"{pases_clave}", fontsize=16, color="white", ha="left")
    y -= gap * 0.7

    # NEW — Faltas Recibidas
    ax_stats.text(0.05, y, "Faltas Recibidas", fontsize=14, color="#888888", ha="left")
    ax_stats.text(0.65, y, f"{faltas_recibidas}", fontsize=16, color="white", ha="left")
    y -= gap * 1.2



    # NEW — 1v1 Defensivo
    ax_stats.text(0.05, y, "1v1 Defensivo", fontsize=14, color="#888888", ha="left")
    ax_stats.text(0.65, y, f"{uno_v_uno_def}", fontsize=16, color="white", ha="left")
    y -= gap * 0.7


    # NEW — Faltas Cometidas
    ax_stats.text(0.05, y, "Faltas Cometidas", fontsize=14, color="#888888", ha="left")
    ax_stats.text(0.65, y, f"{faltas_cometidas}", fontsize=16, color="white", ha="left")
    y -= gap * 1.0


    # TARJETAS
    ax_stats.text(0.05, y, "Tarjetas", fontsize=14, color="#888888",
                  ha="left")

    # ---- Draw the squares RELIABLY using figure coordinates ----
    fig_x, fig_y = ax_stats.transAxes.transform((0.65, y))
    inv = fig.transFigure.inverted()

    # Red card square position
    red_fig_x, red_fig_y = inv.transform((fig_x, fig_y))
    red_ax = fig.add_axes([red_fig_x, red_fig_y - 0.02, 0.03, 0.03])
    red_ax.add_patch(Rectangle((0,0), 1.5, 1, color="red"))
    red_ax.axis("off")

    # Red card number
    ax_stats.text(0.65 + 0.06, y, f"{sanciones_roja}",
                  fontsize=16, color="white", va="center")

    # Yellow card square
    yel_fig_x, yel_fig_y = inv.transform((fig_x + 50, fig_y))
    yel_ax = fig.add_axes([yel_fig_x, yel_fig_y - 0.02, 0.03, 0.03])
    yel_ax.add_patch(Rectangle((0,0), 1.5, 1, color="yellow"))
    yel_ax.axis("off")

    # Yellow card number
    ax_stats.text(0.65 + 0.17, y, f"{sanciones_amarilla}",
                  fontsize=16, color="white", va="center")


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
        "Pérdida": perdida,
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
    "Pérdida",
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
    player_name,
    position_name,
    player_kpis,
    pos_avg,
    team_max
):
    """
    Creates a figure with two radars:
      - Left: Ofensivo
      - Right: Defensivo (player or Arquero version)
    """

    fm = FontManager(
        'https://github.com/google/fonts/raw/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf'
    )

    # ----- Offensive values -----
    OFF = [player_kpis[k] for k in OFFENSIVE_PARAMS]
    POS_OFF = [pos_avg[k] for k in OFFENSIVE_PARAMS]
    MIN_OFF = [0] * len(OFFENSIVE_PARAMS)
    MAX_OFF = [team_max[k] for k in OFFENSIVE_PARAMS]

    # ----- Defensive params depend on position -----
    if str(position_name).strip().lower() == "arquero":
        DEF_PARAMS = DEFENSIVE_PARAMS_ARQUERO
    else:
        DEF_PARAMS = DEFENSIVE_PARAMS_PLAYER

    DEF = [player_kpis[k] for k in DEF_PARAMS]
    POS_DEF = [pos_avg[k] for k in DEF_PARAMS]
    MIN_DEF = [0] * len(DEF_PARAMS)
    MAX_DEF = [team_max[k] for k in DEF_PARAMS]

    # ----- Build figure -----
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

    # ================ LEFT RADAR (OFENSIVO) ================
    radar_off = Radar(
        params=OFFENSIVE_PARAMS,
        min_range=MIN_OFF,
        max_range=MAX_OFF,
        round_int=[False] * len(OFFENSIVE_PARAMS),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )

    radar_off.setup_axis(ax=ax_left)
    radar_off.draw_circles(ax=ax_left, facecolor="#222222", edgecolor="#555555")

    poly = radar_off.draw_radar_compare(
        OFF, POS_OFF, ax=ax_left,
        kwargs_radar={'facecolor': '#1A78CF80', 'edgecolor': '#1A78CF', 'linewidth': 2},
        kwargs_compare={'facecolor': '#69DB7C80', 'edgecolor': '#69DB7C', 'linewidth': 2},
    )

    _, _, v1, v2 = poly
    ax_left.scatter(v1[:, 0], v1[:, 1], c='#1A78CF', edgecolors='white', s=120)
    ax_left.scatter(v2[:, 0], v2[:, 1], c='#69DB7C', edgecolors='white', s=120)

    radar_off.draw_param_labels(ax=ax_left, fontsize=16, fontproperties=fm.prop, color="white")
    radar_off.draw_range_labels(ax=ax_left, fontsize=14, fontproperties=fm.prop, color="#CCCCCC")

    axs['radar'][0].text(
        0.5, 1.02, "OFENSIVO",
        transform=axs['radar'][0].transAxes,
        ha="center", va="center",
        fontsize=22, color="#1A78CF",
        fontproperties=fm.prop
    )

    # ================ RIGHT RADAR (DEFENSIVO) ================
    radar_def = Radar(
        params=DEF_PARAMS,
        min_range=MIN_DEF,
        max_range=MAX_DEF,
        round_int=[False] * len(DEF_PARAMS),
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
    ax_right.scatter(v1_def[:, 0], v1_def[:, 1], c='#FF6B6B', edgecolors='white', s=120)
    ax_right.scatter(v2_def[:, 0], v2_def[:, 1], c='#69DB7C', edgecolors='white', s=120)

    radar_def.draw_param_labels(ax=ax_right, fontsize=16, fontproperties=fm.prop, color="white")
    radar_def.draw_range_labels(ax=ax_right, fontsize=14, fontproperties=fm.prop, color="#CCCCCC")

    axs['radar'][1].text(
        0.5, 1.02, "DEFENSIVO",
        transform=axs['radar'][1].transAxes,
        ha="center", va="center",
        fontsize=22, color="#FF6B6B",
        fontproperties=fm.prop
    )

    # ======= Titles / Global Text =======
    axs['title'].text(
        0.5, 0.75,
        player_name,
        fontsize=30, fontproperties=fm.prop,
        color="white", ha="center"
    )

    axs['title'].text(
        0.5, 0.32,
        f"Comparado con promedio de la posición: {position_name}",
        fontsize=24, color="#69DB7C80",
        fontproperties=fm.prop, ha="center"
    )

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

    fig_of = make_cards_grid(df_1, df_2, OFENSIVO_STATS,
                            title="Máximos – Ofensivo")

    st.pyplot(fig_of, use_container_width=True)


    # ==========================
    # DEFENSIVO CARDS
    # ==========================

    fig_def = make_cards_grid(df_1, df_2, DEFENSIVO_STATS,
                            title="Máximos – Defensivo")

    st.pyplot(fig_def, use_container_width=True)

    # =============================
    # 1. SELECT PLAYER
    # =============================
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
            goles_recibidos=card_data["goles_recibidos"]
        )

        st.pyplot(fig_card, use_container_width=True)

        # =====================================================
        #   DUAL RADAR (OFENSIVO / DEFENSIVO)
        # =====================================================

        st.markdown("<h3 style='color:white;margin-top:25px;'>Radar Comparativo</h3>",
                    unsafe_allow_html=True)

        player_kpis = compute_player_kpis(df_team_original, df_tiempos, selected_player)
        pos_avg = compute_position_avg(df_team_original, df_tiempos, card_data["position"])
        team_max = compute_team_max(df_team_original, df_tiempos)

        fig_radar = plot_dual_radar_with_grid(
            selected_player,
            card_data["position"],
            player_kpis,
            pos_avg,
            team_max
        )

        st.pyplot(fig_radar, use_container_width=True)


        # =============================
        # 3. EVENT TYPE SELECTION
        # =============================
        st.markdown("<h3 style='color:white;'>Seleccionar Evento</h3>", unsafe_allow_html=True)

        modo = st.radio(
            "Categoría",
            ["Ofensivo", "Defensivo"],
            horizontal=True
        )

        # Offensive and defensive event lists
        offensive_events = [
            "Tiro Ferro",
            "1 VS 1 Ofensivo",
            "Pase Clave",
            "Falta Recibida",
            "Asistencia"
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
        fig_summary = plot_event_summary_matplotlib(
            df_player=df_player,
            df_team=df_team_original,
            event_name=selected_event,
            result_column="Resultado"
        )

        st.pyplot(fig_summary, use_container_width=True)
