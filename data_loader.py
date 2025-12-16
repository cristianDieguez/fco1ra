import os
import re
import pandas as pd
import unicodedata

def build_tournament_parquet(DF_CAMPEONATO, DF_COPA):
    """
    Builds a parquet summarizing Campeonato + Copa Plata with:
    - Fecha
    - Competicion
    - Local / Visitante
    - Goles Local / Visitante
    - Goleadores (from Goleador1..5)
    - Roja (string or list)
    - Position groups (list for each)
    """


    def to_list(x):
        """Converts comma or | separated strings into a list."""
        if pd.isna(x) or x is None:
            return []
        return [v.strip() for v in re.split(r"[|,]", str(x)) if v.strip()]

    def goleadores_from_columns(row):
        """Collects Goleador 1–5 into one clean list."""
        cols = [f"Goleador {i}" for i in range(1, 6)]
        players = []
        for c in cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip():
                players.append(str(row[c]).strip())
        return players

    def parse_roja(value):
        """
        Returns string if single entry.
        Returns list if multiple.
        Empty → None
        """
        if pd.isna(value) or value is None or str(value).strip() == "":
            return None

        parts = [v.strip() for v in re.split(r"[|,]", str(value)) if v.strip()]

        if len(parts) == 1:
            return parts[0]   # single red card → string
        return parts           # multiple → list

    def process_sheet(df, competicion_name):
        rows = []

        for _, row in df.iterrows():

            condicion = str(row["Condición"]).strip().lower()
            ferro = "Ferro"
            rival = str(row["Rival"]).strip()

            gf = int(row["Goles_Favor"])
            gr = int(row["Goles_Contra"])

            if condicion == "local":
                local = ferro
                visitante = rival
                goles_local = gf
                goles_visitante = gr
            else:
                local = rival
                visitante = ferro
                goles_local = gr
                goles_visitante = gf

            rows.append({
                "Fecha": row["Fecha"],
                "Competicion": competicion_name,

                "Local": local,
                "Visitante": visitante,
                "Goles Local": goles_local,
                "Goles Visitante": goles_visitante,

                "Goleadores": goleadores_from_columns(row),

                # NEW ROJA COLUMN (string OR list)
                "Roja": parse_roja(row.get("Roja", None)),

                # Positional lists
                "Arqueros": to_list(row.get("Arqueros", "")),
                "Cierres": to_list(row.get("Cierres", "")),
                "Alas Diestros": to_list(row.get("Alas Diestros", "")),
                "Alas Zurdos": to_list(row.get("Alas Zurdos", "")),
                "Pivots": to_list(row.get("Pivots", "")),
            })

        return rows

    # Process both competition sheets
    rows_campeonato = process_sheet(DF_CAMPEONATO, "Campeonato")
    rows_copa = process_sheet(DF_COPA, "Copa Plata")

    # Combine them
    final_rows = rows_campeonato + rows_copa

    return pd.DataFrame(final_rows)

def extract_dorsales_general(x):
    """
    Extracts dorsales based on these rules:
    - Split by '|'
    - For each chunk, take ONLY the number to the LEFT of a hyphen if it exists
    - If no hyphen, extract the first number found
    - Return list of integers (unique, sorted)
    """

    if pd.isna(x) or x is None:
        return []

    # Convert to string and split by '|'
    chunks = str(x).split("|")

    dorsales = []

    for c in chunks:
        c = c.strip()

        if "-" in c:
            # Everything before the first hyphen
            left = c.split("-", 1)[0].strip()

            # Extract first integer from the left side
            match = re.search(r"\d{1,2}", left)
            if match:
                dorsales.append(int(match.group()))
                continue

        # If no hyphen or left side failed: extract the first integer from the chunk
        match = re.search(r"\d{1,2}", c)
        if match:
            dorsales.append(int(match.group()))

    # return unique sorted dorsales
    return sorted(set(dorsales))

def extract_single_general(x):
    lst = extract_dorsales_general(x)
    return lst[0] if lst else None

    # --- Helper: Normalize text (lowercase + remove accents) ---

def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    return s

def assign_match_metadata(df, jornada, TOURNAMENT_DATA):
    """
    Safely matches Jornada from filename with Fecha in Campeonato:
    - Tries integer comparison first
    - Falls back to string comparison
    - Supports text-based Fecha like 'Cuartos'
    """

    if jornada is None:
        raise ValueError("❌ No jornada found in filename.")

    # jornada extracted from filename → convert to int
    try:
        jornada_int = int(jornada)
    except:
        jornada_int = None

    # Normalize Fecha column to allow int matching where possible
    fechas = TOURNAMENT_DATA["Fecha"]

    # Try matching numerically
    match = pd.DataFrame()
    if jornada_int is not None:
        # Convert only numeric fechas to int, ignore text fechas
        numeric_mask = fechas.apply(lambda x: str(x).isdigit())
        numeric_fechas = fechas[numeric_mask].astype(int)

        match = TOURNAMENT_DATA.loc[numeric_fechas == jornada_int]

    # If no numeric match, fall back to string matching
    if match.empty:
        jornada_str = str(jornada).strip()
        match = TOURNAMENT_DATA[fechas.astype(str).str.strip() == jornada_str]

    if match.empty:
        raise ValueError(f"❌ Jornada '{jornada}' not found in Campeonato sheet.")

    row = match.iloc[0]

    condicion = str(row["Condición"]).strip().lower()
    rival_team = str(row["Rival"]).strip()
    ferro_team = "Ferro"

    # Determine home/away mapping
    if condicion == "local":
        local_team = ferro_team
        visitor_team = rival_team
    else:
        local_team = rival_team
        visitor_team = ferro_team

    # Assign fields
    df["Jornada"] = str(jornada)
    df["Local"] = local_team
    df["Visitante"] = visitor_team

    return df


def format_mmss(seconds):
    if seconds is None or pd.isna(seconds):
        return None
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"


def assign_parte(df_partido, df_primer):
    """
    Adds a column 'Parte' to the PARTIDO EN JUEGO dataframe:
    - 'Primer Tiempo' if Fin <= max(Fin) of PRIMER TIEMPO
    - 'Segundo Tiempo' otherwise
    """

    fin_primer = df_primer["Fin"].max()  # timedelta

    df_partido = df_partido.copy()
    df_partido["Parte"] = df_partido["Fin"].apply(
        lambda f: 1 if f <= fin_primer else 2
    )

    return df_partido

def build_final_df(section_dfs):
    """
    Recombines sections into a single dataframe.
    Ensures all sections share columns: Evento, Tiempo, Inicio, Fin, Parte.
    Adds 'Seccion' for debugging/analysis.
    """

    final_list = []

    for sec_name, df in section_dfs.items():
        df2 = df.copy()

        # Add Parte if missing (only PARTIDO EN JUEGO has it)
        if "Parte" not in df2.columns:
            df2["Parte"] = None

        # Ensure all output columns exist
        keep_cols = ["Evento", "Tiempo", "Fin", "Parte"]
        for col in keep_cols:
            if col not in df2.columns:
                df2[col] = None

        df2 = df2[keep_cols]

        final_list.append(df2)

    df_final = pd.concat(final_list, ignore_index=True)
    return df_final




OUTPUT_DIR = "datasets"
TOURNAMENT_DATA = "datasets/Resultados_partidos.xlsx"
DF_CAMPEONATO = pd.read_excel(TOURNAMENT_DATA, sheet_name="CAMPEONATO")
DF_COPA = pd.read_excel(TOURNAMENT_DATA, sheet_name="COPA PLATA")

tournament_df = build_tournament_parquet(DF_CAMPEONATO, DF_COPA)
tournament_df.to_parquet("datasets/tournament_summary.parquet", index=False)



def extract_matchday_from_filename(file_name: str):
    """
    Extracts the matchday (Fecha/Jornada) number from filenames like:
    'Fecha 25 - Ferro vs Newbery - Botonera 3.xlsx'
    Returns a string (e.g. '25') or None.
    """
    m = re.match(
        r"Fecha\s*(\d+)\s*-\s*(.*?)\s*vs\s*(.*?)(?:\s*-|\.xlsx|$)",
        file_name,
        re.IGNORECASE,
    )
    return m.group(1) if m else None

# =============== 3️⃣ PARSE EXCEL ===============

def correct_sentido_tiros(df1: pd.DataFrame, matches_list: list):
    """
    Fixes wrong Sentido for Tiro Rival and Tiro Ferro based on FieldXto logic.
    
    Rules:
    - Tiro Rival: if FieldXto > 0.5 → flip all X/Y coords
    - Tiro Ferro: if FieldXto < 0.5 → flip all X/Y coords
    
    Flipping = 1 - coordinate
    """

    # Columns to flip
    coord_cols = ["FieldXfrom", "FieldYfrom", "FieldXto", "FieldYto"]

    # Work on a copy
    df = df1.copy()

    # Filter only selected matches
    mask_matches = df["Jornada"].isin(matches_list)

    # -----------------------------
    # 1) FIX TIRO RIVAL ORIENTATION
    # -----------------------------
    mask_rival = (
        mask_matches &
        (df["Acción"] == "Tiro Rival") &
        (pd.to_numeric(df["FieldXto"], errors="coerce") > 0.5)
    )

    df.loc[mask_rival, coord_cols] = 1 - df.loc[mask_rival, coord_cols]

    # -----------------------------
    # 2) FIX TIRO FERRO ORIENTATION
    # -----------------------------
    mask_ferro = (
        mask_matches &
        (df["Acción"] == "Tiro Ferro") &
        (pd.to_numeric(df["FieldXto"], errors="coerce") < 0.5)
    )

    df.loc[mask_ferro, coord_cols] = 1 - df.loc[mask_ferro, coord_cols]

    return df

def parse_longomatch_excel_botonera_1(file_bytes, file_name):
    # --- Section reference list ---
    sections_ref = [
        "Tiro Ferro", "Asistencia", "1 VS 1 Ofensivo", "Recuperación", "Presión", "Arquero", "Tiro Rival",
        "Pase Clave", "1 VS 1 Defensivo", "Pérdida", "Falta Cometida", "Falta Recibida",
        "Sanción Ferro", "Sanción Rival", "Sustituciones"
    ]

    # --- Extract match info from filename ---
    match_info = re.match(r"Fecha\s*(\d+)\s*-\s*(.*?)\s*vs\s*(.*?)(?:\s*-|\.xlsx|$)", file_name, re.IGNORECASE)
    matchday = match_info.group(1) if match_info else None
    
    # --- Read Excel ---
    df_raw = pd.read_excel(file_bytes, sheet_name="Estadísticas del proyecto", header=None)
    section_dfs = {}
    i = 0
    n = len(df_raw)

    while i < n:
        row_value = str(df_raw.iloc[i, 0]).strip() if pd.notna(df_raw.iloc[i, 0]) else ""
        if row_value in sections_ref:
            section_name = row_value
            if i + 1 < n:
                headers = df_raw.iloc[i + 1].dropna().tolist()
                data_start = i + 2

                # find end of section
                data_end = data_start
                while data_end < n:
                    next_val = str(df_raw.iloc[data_end, 0]).strip() if pd.notna(df_raw.iloc[data_end, 0]) else ""
                    if next_val in sections_ref:
                        break
                    data_end += 1

                section_df = df_raw.iloc[data_start:data_end, :len(headers)].copy()
                section_df.columns = headers

                # add metadata columns
                section_df.insert(0, "Acción", section_name)

                section_dfs[section_name] = section_df
                i = data_end
                continue
        i += 1

    df_all = pd.concat(section_dfs.values(), ignore_index=True)

        # === Correct coordinates using Campeonato sheet (B1_Sentido) ===

    try:
        # Match Fecha in CAMPEONATO sheet
        match_row = DF_CAMPEONATO[DF_CAMPEONATO["Fecha"].astype(str) == str(matchday)]

        if not match_row.empty:
            sentido = str(match_row.iloc[0]["B1_Sentido"]).strip().upper()

            if sentido == "I":   # invert
                coord_cols = [
                    "FieldXfrom", "FieldYfrom",
                    "FieldXto", "FieldYto",
                    "FieldX", "FieldY"
                ]

                for col in coord_cols:
                    if col in df_all.columns:
                        df_all[col] = 1 - df_all[col]

    except Exception as e:
        print(f"⚠️ Could not apply B1_Sentido correction for match {file_name}: {e}")

    # --- Clean-up ---
    # Clean Jugadores → list[int]
    df_all = assign_match_metadata(df_all, matchday, DF_CAMPEONATO)
    df_all["Jugadores"] = df_all["Jugadores"].apply(extract_dorsales_general)
    df_all = df_all.drop(columns=["Acercar", "Alejar", "Equipo"], errors='ignore')

    df_all = df_all[~df_all["Evento"].isna()]
    df_all = df_all.explode("Jugadores")
    df_all[["Jugadores", "Jornada"]] = df_all[["Jugadores", "Jornada"]].astype("Int64")
    df_all = df_all[df_all["Acción"] != "Sustituciones"]


    df_all = df_all.drop(columns=["Inicio", "Fin"])
    df_all['Tiempo'] = df_all['Tiempo'].apply(convert_to_seconds)

    matches_to_correct = [1, 3, 6, 10, 13, 14, 16, 17, 19, 22, 28, 31]

    df_all = correct_sentido_tiros(df_all, matches_list = matches_to_correct) 






    return df_all


def parse_longomatch_excel_botonera_2(file_bytes, file_name):
    """
    Parses the Botonera 2 Excel file containing transition and substitution events.
    Cleans blank rows, preserves Jugadores/Acercar/Alejar, and adds metadata.
    """

    # --- Official canonical names ---
    canonical_sections = {
        "transicion ofensiva": "Transición Ofensiva",
        "mt ofensivo": "MT Ofensivo",
        "mt defensivo": "MT Defensivo",
        "transicion defensiva": "Transición Defensiva",
        "sustituciones": "Sustituciones",
    }

    # --- Extract match info from filename ---
    match_info = re.match(
        r"Fecha\s*(\d+)\s*-\s*(.*?)\s*vs\s*(.*?)(?:\s*-|\.xlsx|$)",
        file_name, re.IGNORECASE
    )
    matchday = match_info.group(1) if match_info else None

    # --- Read raw Excel ---
    df_raw = pd.read_excel(file_bytes, sheet_name="Estadísticas del proyecto", header=None)
    section_dfs = {}
    i = 0
    n = len(df_raw)

    # --- 4️⃣ Loop through rows to detect sections ---
    while i < n:
        raw_value = df_raw.iloc[i, 0]
        row_value = str(raw_value).strip() if pd.notna(raw_value) else ""
        normalized = normalize_text(row_value)

        # If this row is a section title
        if normalized in canonical_sections:
            section_name = canonical_sections[normalized]

            # Extract headers
            headers = df_raw.iloc[i + 1].dropna().tolist()
            data_start = i + 2

            # Find end of section
            data_end = data_start
            while data_end < n:
                next_raw = df_raw.iloc[data_end, 0]
                next_val = str(next_raw).strip() if pd.notna(next_raw) else ""
                normalized_next = normalize_text(next_val)

                if normalized_next in canonical_sections:
                    break

                data_end += 1

            # Extract section rows
            section_df = df_raw.iloc[data_start:data_end, :len(headers)].copy()
            section_df.columns = headers

            # --- CLEAN: remove fully empty rows ---
            section_df = section_df.dropna(how="all")

            # Remove rows that are only empty strings/spaces
            section_df = section_df[
                ~section_df.apply(lambda row: row.astype(str).str.strip().eq("").all(), axis=1)
            ].reset_index(drop=True)

            # --- Ensure required columns exist (do NOT overwrite real values) ---
            for col in ["Jugadores", "Acercar", "Alejar"]:
                if col not in section_df.columns:
                    section_df[col] = None

            # --- Add metadata ---
            section_df.insert(0, "Acción", section_name)

            # Store it
            section_dfs[section_name] = section_df

            i = data_end
            continue

        i += 1

    # --- 5️⃣ Combine all parsed sections ---
    if not section_dfs:
        raise ValueError("❌ No recognizable sections found in this Botonera 2 file.")

    df_all = pd.concat(section_dfs.values(), ignore_index=True)

    # assign Jornada, Local, Visitante from Campeonato sheet
    df_all = assign_match_metadata(df_all, matchday, DF_CAMPEONATO)

    # --- Drop completely empty columns ---
    df_all = df_all.dropna(axis=1, how="all")

    # --- Ensure required columns still exist ---
    for col in ["Jugadores", "Acercar", "Alejar"]:
        if col not in df_all.columns:
            df_all[col] = None

    # --- 6️⃣ Normalize dorsales ---
    df_all["Jugadores"] = df_all["Jugadores"].apply(extract_dorsales_general)
    df_all["Acercar"] = df_all["Acercar"].apply(extract_single_general)
    df_all["Alejar"] = df_all["Alejar"].apply(extract_single_general)

    # --- 7️⃣ Consolidate Botonera 2 action types ---
    action_cols = ["Transición Ofensiva", "MT Ofensivo", "MT Defensivo", "Tipo_de_Accion"]

    def consolidate_action(row):
        for col in action_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                return row[col]
        return None

    df_all["Tipo_de_Accion"] = df_all.apply(consolidate_action, axis=1)

    df_all = df_all.drop(columns=["Equipo", "FieldX", "FieldY", "FieldXfrom", "FieldYfrom", "FieldXto", "FieldYto", "Transición Ofensiva", "MT Ofensivo", "MT Defensivo"], errors='ignore')

    df_all = df_all.explode("Jugadores")
    df_all[["Jugadores", "Jornada"]] = df_all[["Jugadores", "Jornada"]].astype("Int64")

    df_all['Inicio'] = df_all['Inicio'].apply(convert_to_seconds)
    df_all['Fin'] = df_all['Fin'].apply(convert_to_seconds)
    df_all['Tiempo'] = df_all['Tiempo'].apply(convert_to_seconds)
    
    
    return df_all

def parse_longomatch_excel_botonera_3(file_path: str, file_name: str):
    """
    Parses 'Botonera 3' Excel file (local file).
    - Sections '1' to '25' represent player sections.
    - In each section, rows look like: [ '8 001', Tiempo, Inicio, Fin ]
    - We split the first column at the space:
        - first number  → Jugador (e.g. 8)
        - (we ignore the second token for now)
    - Returns columns: Jugadores, Tiempo, Inicio, Fin, Jornada, Fecha, Local, Visitante, Match
    """

    df_raw = pd.read_excel(file_path, header=None)
    n = len(df_raw)

    player_sections = [str(i) for i in range(1, 26)]
    skip_sections = ["GOL A FAVOR EQ", "GOL RIVAL EQ", "ABP FERRO", "ABP RIVAL", "Sustituciones"]

    all_data = []
    i = 0

    while i < n:
        row_val = df_raw.iloc[i, 0]
        row_value = str(row_val).strip() if pd.notna(row_val) else ""

        # --- Player section (1–25) ---
        if row_value in player_sections:
            data_start = i + 2  # assume data starts 2 rows below section number
            data_end = data_start

            # Find where this section ends
            while data_end < n:
                next_val = df_raw.iloc[data_end, 0]
                next_str = str(next_val).strip() if pd.notna(next_val) else ""
                if next_str in player_sections or next_str in skip_sections:
                    break
                data_end += 1

            section_df = df_raw.iloc[data_start:data_end].copy()
            section_df = section_df.dropna(how="all")

            if not section_df.empty:
                # Expect columns: EventoRaw, Tiempo, Inicio, Fin
                section_df = section_df.iloc[:, :4]
                section_df.columns = ["EventoRaw", "Tiempo", "Inicio", "Fin"][: section_df.shape[1]]

                # --- Extract Jugador from EventoRaw (first number before space) ---
                def _extract_jugador(ev):
                    if pd.isna(ev):
                        return None
                    parts = str(ev).strip().split()
                    if not parts:
                        return None
                    try:
                        return int(parts[0])
                    except ValueError:
                        return None

                section_df["Jugadores"] = section_df["EventoRaw"].apply(_extract_jugador)

                # Keep only what we really need for game-time calculations
                keep_cols = ["Jugadores", "Tiempo", "Inicio", "Fin"]
                section_df = section_df[keep_cols]


                # Convert times to seconds
                for col in ["Tiempo", "Inicio", "Fin"]:
                    if col in section_df.columns:
                        section_df[col] = section_df[col].apply(convert_to_seconds)

                # Drop rows where Jugador is missing
                section_df = section_df[section_df["Jugadores"].notna()]

                all_data.append(section_df)

            i = data_end
            continue

        # --- Skip non-player sections entirely ---
        elif row_value in skip_sections:
            data_end = i + 1
            while data_end < n:
                next_val = df_raw.iloc[data_end, 0]
                next_str = str(next_val).strip() if pd.notna(next_val) else ""
                if next_str in player_sections or next_str in skip_sections:
                    break
                data_end += 1
            i = data_end
            continue

        else:
            i += 1

    if not all_data:
        df_final = pd.DataFrame(columns=["Jugadores", "Tiempo", "Inicio", "Fin"])
    else:
        df_final = pd.concat(all_data, ignore_index=True)

    # --- Map Jornada/Fecha + Local/Visitante from Campeonato ---
    jornada = extract_matchday_from_filename(file_name)
    df_final = assign_match_metadata(df_final, jornada, DF_CAMPEONATO)

    # Types
    df_final["Jugadores"] = df_final["Jugadores"].astype("Int64")
    df_final["Jornada"] = df_final["Jornada"].astype("Int64")

    return df_final

def convert_to_seconds(value):
    """
    Converts any Botonera time value into SECONDS (float).
    Output example: 455.2 seconds.
    """
    import datetime

    if pd.isna(value):
        return None

    # Case 1: Excel float (fraction of a day)
    if isinstance(value, (int, float)):
        return float(value) * 86400  # 24*60*60

    # Case 2: datetime.time
    if isinstance(value, datetime.time):
        return value.hour*3600 + value.minute*60 + value.second

    # Case 3: datetime.datetime (use the time part)
    if isinstance(value, datetime.datetime):
        t = value.time()
        return t.hour*3600 + t.minute*60 + t.second

    # Case 4: parse string
    s = str(value).strip()

    # Handle MM:SS
    if s.count(":") == 1:
        m, s = s.split(":")
        return int(m)*60 + float(s)

    # Handle HH:MM:SS
    if s.count(":") == 2:
        h, m, s = s.split(":")
        return int(h)*3600 + int(m)*60 + float(s)

    return None


def parse_botonera4_sections(file_path):
    """
    Reads the Botonera Excel file and extracts sections.
    Normalises section names to lowercase.
    Converts Tiempo, Inicio, Fin to seconds.
    """

    # SECTION NAMES normalized to lowercase
    SECTION_NAMES = {
        "primer tiempo": "primer tiempo",
        "segundo tiempo": "segundo tiempo",
        "partido en juego": "partido en juego",
        "sustituciones": "sustituciones"
    }

    df_raw = pd.read_excel(file_path, sheet_name="Estadísticas del proyecto", header=None)

    section_dfs = {}
    n = len(df_raw)
    i = 0

    while i < n:

        cell_value = df_raw.iloc[i, 0]
        title = str(cell_value).strip().lower() if pd.notna(cell_value) else ""

        if title in SECTION_NAMES:

            # Header row
            headers = df_raw.iloc[i + 1].dropna().tolist()
            num_cols = len(headers)
            data_start = i + 2

            # Find next section start
            data_end = data_start
            while data_end < n:
                next_val = df_raw.iloc[data_end, 0]
                next_title = str(next_val).strip().lower() if pd.notna(next_val) else ""
                if next_title in SECTION_NAMES:
                    break
                data_end += 1

            # Extract data
            sec = df_raw.iloc[data_start:data_end, :num_cols].copy()
            sec.columns = headers

            # Keep expected columns
            relevant = [c for c in ["Evento", "Tiempo", "Inicio", "Fin"] if c in sec.columns]
            sec = sec[relevant]

            # Convert time columns (ONE TIME ONLY — from Excel to seconds)
            for col in ["Tiempo", "Inicio", "Fin"]:
                if col in sec.columns:
                    sec[col] = sec[col].apply(convert_to_seconds)

            # Remove empty rows
            sec = sec.dropna(how="all")
            sec = sec[
                ~sec.apply(lambda r: r.astype(str).str.strip().eq("").all(), axis=1)
            ].reset_index(drop=True)

            # Store using NORMALIZED NAME
            normalized_name = SECTION_NAMES[title]
            section_dfs[normalized_name] = sec

            i = data_end
            continue

        i += 1

    df_all = pd.concat(section_dfs.values(), ignore_index=True)
    return section_dfs, df_all

def parse_single_botonera_4(file_path: str, file_name: str):
    """
    Parse Botonera 4, using normalized lowercase section names.
    - 'Tiempo', 'Inicio', 'Fin' are already in SECONDS (from parse_botonera4_sections).
    - Creates a new 'Tiempo_Efectivo' column from 'Tiempo'.
    - Keeps Inicio/Fin as video time in seconds.
    """

    section_dfs, _ = parse_botonera4_sections(file_path)

    # All keys now lowercase
    df_partido = section_dfs["partido en juego"]
    df_primer = section_dfs["primer tiempo"]

    # Assign Parte based on PRIMER TIEMPO
    df_partido = assign_parte(df_partido, df_primer)
    section_dfs["partido en juego"] = df_partido

    # Build final dataframe from all sections
    df_final = build_final_df(section_dfs)


    # Duration of each segment in seconds
    if "Tiempo" in df_final.columns and "Fin" in df_final.columns:
        df_final["Duracion"] = df_final["Fin"] - df_final["Tiempo"]
    else:
        df_final["Duracion"] = None

    # Keep only rows with Parte assigned
    df_final = df_final[df_final["Parte"].notna()].reset_index(drop=True)

    # Metadata from filename / campeonato
    jornada = extract_matchday_from_filename(file_name)
    df_final = assign_match_metadata(df_final, jornada, DF_CAMPEONATO)
    df_final["Jornada"] = df_final["Jornada"].astype("Int64")

    return df_final



# =============== 4️⃣ PROCESS ALL MATCHES ===============
def process_all_matches_local(root_partidos_dir: str):
    """
    root_partidos_dir should be the folder that contains:
        botonera_1/
        botonera_2/
        botonera_3/
        botonera_4/

    """

    b1_dir = os.path.join(root_partidos_dir, "botonera_1")
    b2_dir = os.path.join(root_partidos_dir, "botonera_2")
    b3_dir = os.path.join(root_partidos_dir, "botonera_3")
    b4_dir = os.path.join(root_partidos_dir, "botonera_4")

    dfs_b1 = []
    dfs_b2 = []
    dfs_b3 = []
    dfs_b4 = []

    # -------------- BOTONERA 1 --------------
    if os.path.isdir(b1_dir):
        for fname in os.listdir(b1_dir):
            if not fname.lower().endswith((".xlsx", ".xls")):
                continue
            fpath = os.path.join(b1_dir, fname)
            print(f"📄 [B1] Processing {fname}")
            df1 = parse_longomatch_excel_botonera_1(fpath, fname)
            dfs_b1.append(df1)
    else:
        print(f"⚠️ Botonera 1 directory not found: {b1_dir}")

    # -------------- BOTONERA 2 --------------
    if os.path.isdir(b2_dir):
        for fname in os.listdir(b2_dir):
            if not fname.lower().endswith((".xlsx", ".xls")):
                continue
            fpath = os.path.join(b2_dir, fname)
            print(f"📄 [B2] Processing {fname}")
            df2 = parse_longomatch_excel_botonera_2(fpath, fname)
            dfs_b2.append(df2)
    else:
        print(f"⚠️ Botonera 2 directory not found: {b2_dir}")

    # -------------- BOTONERA 3 --------------
    if os.path.isdir(b3_dir):
        for fname in os.listdir(b3_dir):
            if not fname.lower().endswith((".xlsx", ".xls")):
                continue
            fpath = os.path.join(b3_dir, fname)
            print(f"📄 [B3] Processing {fname}")
            df3 = parse_longomatch_excel_botonera_3(fpath, fname)
            dfs_b3.append(df3)
    else:
        print(f"⚠️ Botonera 3 directory not found: {b3_dir}")

    # -------------- BOTONERA 4 --------------
    if os.path.isdir(b4_dir):
        for fname in os.listdir(b4_dir):
            if not fname.lower().endswith((".xlsx", ".xls")):
                continue
            fpath = os.path.join(b4_dir, fname)
            print(f"📄 [B4] Processing {fname}")
            df4 = parse_single_botonera_4(fpath, fname)
            dfs_b4.append(df4)
    else:
        print(f"⚠️ Botonera 4 directory not found: {b4_dir}")

    # -------------- SAVE PARQUETS --------------
    os.makedirs("datasets", exist_ok=True)

    if dfs_b1:
        df_all_1 = pd.concat(dfs_b1, ignore_index=True)
        df_all_1.to_parquet("datasets/botonera_1.parquet", index=False)
        print("✅ Saved datasets/botonera_1.parquet")
    else:
        print("⚠️ No Botonera 1 files processed.")

    if dfs_b2:
        df_all_2 = pd.concat(dfs_b2, ignore_index=True)
        df_all_2.to_parquet("datasets/botonera_2.parquet", index=False)
        print("✅ Saved datasets/botonera_2.parquet")
    else:
        print("⚠️ No Botonera 2 files processed.")

    if dfs_b3:
        df_all_3 = pd.concat(dfs_b3, ignore_index=True)
        df_all_3.to_parquet("datasets/botonera_3.parquet", index=False)
        print("✅ Saved datasets/botonera_3.parquet")
    else:
        print("⚠️ No Botonera 3 files processed.")

    if dfs_b4:
        df_all_4 = pd.concat(dfs_b4, ignore_index=True)
        df_all_4.to_parquet("datasets/botonera_4.parquet", index=False)
        print("✅ Saved datasets/botonera_4.parquet")
    else:
        print("⚠️ No Botonera 4 files processed.")


# =============== 5️⃣ RUN SCRIPT ===============
if __name__ == "__main__":
    root_partidos_dir = r"C:\Users\Admin\Desktop\Streamlit\fco\partidos"

    process_all_matches_local(root_partidos_dir)
