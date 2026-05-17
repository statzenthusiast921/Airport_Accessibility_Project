"""Airport graph data loading and lookups."""

import pandas as pd
import numpy as np

#-----Read in and set up data
airport_df = pd.read_parquet('https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/master_air.parquet')

#-----One lookup for IATA to country (avoid huge merge joins on edge tables with 1000s of rows)
IATA_TO_COUNTRY = (
    airport_df.drop_duplicates(subset=["iata"])
    .set_index("iata")["country"]
)


def attach_country_columns_to_edges(edges_df):
    out = edges_df.copy()
    out["source_country"] = out["source"].map(IATA_TO_COUNTRY)
    out["target_country"] = out["target"].map(IATA_TO_COUNTRY)
    return out


graph1 = pd.read_parquet('https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_airline_airport.parquet')
airport_df1 = airport_df[['iata','country','display_name']].drop_duplicates()
airport_df1.rename(columns={'iata':'airport'}, inplace=True)
graph1_merged = pd.merge(graph1, airport_df1, on ='airport')


def build_airline_iata_to_name(airport_df):
    mapping = {}
    for carriers in airport_df["carriers"].dropna():
        if not isinstance(carriers, (list, tuple, np.ndarray)):
            continue
        for c in carriers:
            if not isinstance(c, dict):
                continue
            code = c.get("iata") or c.get("IATA") or c.get("airline_iata") or c.get("code")
            name = c.get("name")
            if code and name:
                code = str(code).strip().upper()
                if len(code) == 2:
                    mapping[code] = name
    return mapping


AIRLINE_IATA_TO_NAME = build_airline_iata_to_name(airport_df)
AIRPORT_IATA_META = (
    airport_df.drop_duplicates(subset=["iata"])
    .set_index("iata")[["display_name", "country"]]
    .to_dict("index")
)

def build_similarity_airport_profiles():
    ap = airport_df.drop_duplicates(subset=["iata"]).copy()
    ap["num_dests"] = pd.to_numeric(ap["num_dests"], errors="coerce").fillna(0).astype(int)
    ap["connectivity_index"] = pd.to_numeric(ap["connectivity_index"], errors="coerce").fillna(0.0)
    ap["redundancy_score"] = pd.to_numeric(ap["redundancy_score"], errors="coerce").fillna(0.0)
    ap["log1p_num_dests"] = np.log1p(ap["num_dests"].to_numpy(dtype=float))
    z_cols = ["connectivity_index", "log1p_num_dests", "redundancy_score"]
    mat = ap[z_cols].to_numpy(dtype=float)
    mu = mat.mean(axis=0)
    sig = mat.std(axis=0)
    sig[sig == 0] = 1.0
    z_mat = (mat - mu) / sig
    profiles = {}
    for i, iata in enumerate(ap["iata"].astype(str)):
        profiles[iata] = {
            "connectivity_index": float(mat[i, 0]),
            "redundancy_score": float(mat[i, 2]),
            "num_dests": int(ap["num_dests"].iloc[i]),
            "z_connectivity_index": float(z_mat[i, 0]),
            "z_log1p_num_dests": float(z_mat[i, 1]),
            "z_redundancy_score": float(z_mat[i, 2]),
        }
    return profiles


SIMILARITY_AIRPORT_PROFILE = build_similarity_airport_profiles()

SIMILARITY_Z_FEATURE_KEYS = (
    "z_connectivity_index",
    "z_log1p_num_dests",
    "z_redundancy_score",
)


graph2 = pd.read_parquet("https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_feature_similarity.parquet")
graph2_merged = attach_country_columns_to_edges(graph2)

graph3 = pd.read_parquet("https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_proximity.parquet")
graph3_merged = attach_country_columns_to_edges(graph3)

# Upper bound used when building edges_proximity.parquet (great-circle miles).
PROXIMITY_EDGE_MAX_MILES = 250.0

SHARED_DESTINATIONS_PARQUET_URL = (
    "https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/"
    "edges_shared_destinations.parquet"
)

SHARED_DESTINATIONS_COSINE_PARQUET_URL = (
    "https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/"
    "edges_shared_destinations_cosine.parquet"
)

#-----Loaded on first use only; keeps app startup responsive
merged_shared_destinations_edges = None
merged_shared_destinations_cosine_edges = None


def load_merged_shared_destinations_edges():
    """Load shared-destination edges once; subsequent calls reuse the same DataFrame."""
    global merged_shared_destinations_edges
    if merged_shared_destinations_edges is None:
        raw = pd.read_parquet(SHARED_DESTINATIONS_PARQUET_URL)
        merged_shared_destinations_edges = attach_country_columns_to_edges(raw)
    return merged_shared_destinations_edges


def load_merged_shared_destinations_cosine_edges():
    """Load cosine-weighted shared-destination edges once; subsequent calls reuse the same DataFrame."""
    global merged_shared_destinations_cosine_edges
    if merged_shared_destinations_cosine_edges is None:
        raw = pd.read_parquet(SHARED_DESTINATIONS_COSINE_PARQUET_URL)
        merged_shared_destinations_cosine_edges = attach_country_columns_to_edges(raw)
    return merged_shared_destinations_cosine_edges


dest_tbl = (
    airport_df[["display_name", "dest_name",'dest_iata', "connectivity_index", "redundancy_score"]].rename(
        columns={
            "dest_name": "Destination",
            "connectivity_index": "Connectivity Index",
            "redundancy_score": "Redundancy Index",
        }
    )
)

# ----- Set up choices for dropdown menus
country_choices = sorted(airport_df['country'].unique())
airport_choices = sorted(airport_df['display_name'].unique())
connection_type_choices = [
    "Carriers",
    "Statistical Similarity",
    "Proximity",
    "Shared Destinations",
    "Shared Destinations (Hub-Adjusted)",
]

#----- Label defintions for connection types
CONNECTION_TYPE_DEFINITIONS = [
    (
        "Carriers",
        "Airlines linked to airports they fly to within the selected country. "
        "Useful for seeing which carriers cluster at which hubs (after the top-carrier filter).",
    ),
    (
        "Statistical Similarity",
        "Airports linked when connectivity index, redundancy score, and destination count "
        "are alike (z-scored, then compared). Link scores are 0–1 (higher = more similar).",
    ),
    (
        "Proximity",
        "Airports linked by short distance (within the proximity edge cap in the dataset). "
        "Shows geographic neighbors, not airline overlap.",
    ),
    (
        "Shared Destinations",
        "Airports linked by how many destination cities they both serve (same destination IATA in both route lists). "
        "The weight is the raw count — large hubs usually share more destinations simply because they fly everywhere.",
    ),
    (
        "Shared Destinations (Hub-Adjusted)",
        "Same underlying overlap as the raw count, but each link is scaled down when either airport serves "
        "a very large number of destinations (cosine-style adjustment). "
        "Makes overlap between a mega-hub and a mid-size airport easier to interpret fairly.",
    ),
]

#----- Counry --> Airport Dictionary
df_for_dict = airport_df[['country','display_name']]
df_for_dict = df_for_dict.drop_duplicates(subset='display_name',keep='first')
country_airport_dict = df_for_dict.groupby('country')['display_name'].apply(list).to_dict()

HOVER_COLS = [
    'latitude', 'longitude', 'display_name',
    'num_dests', 'redundancy_score', 'connectivity_index',
]
