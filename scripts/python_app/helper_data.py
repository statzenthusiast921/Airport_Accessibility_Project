"""Airport graph data loading and lookups."""

import pandas as pd
import numpy as np

GITHUB_DATA_BASE = (
    "https://raw.githubusercontent.com/statzenthusiast921/"
    "Airport_Accessibility_Project/main/data/"
)

MASTER_AIR_PARQUET_URL = GITHUB_DATA_BASE + "master_air.parquet"
EDGES_AIRLINE_AIRPORT_PARQUET_URL = GITHUB_DATA_BASE + "edges_airline_airport.parquet"
EDGES_FEATURE_SIMILARITY_PARQUET_URL = GITHUB_DATA_BASE + "edges_feature_similarity.parquet"
EDGES_PROXIMITY_PARQUET_URL = GITHUB_DATA_BASE + "edges_proximity.parquet"
SHARED_DESTINATIONS_PARQUET_URL = GITHUB_DATA_BASE + "edges_shared_destinations.parquet"
SHARED_DESTINATIONS_COSINE_PARQUET_URL = (
    GITHUB_DATA_BASE + "edges_shared_destinations_cosine.parquet"
)

# Populated by app.py before initialize_derived_state() runs.
airport_df = None
graph1 = None
graph2 = None
graph3 = None
merged_shared_destinations_edges = None
merged_shared_destinations_cosine_edges = None

IATA_TO_COUNTRY = None
graph1_merged = None
AIRLINE_IATA_TO_NAME = None
AIRPORT_IATA_META = None
SIMILARITY_AIRPORT_PROFILE = None
graph2_merged = None
graph3_merged = None
dest_tbl = None
country_choices = None
airport_choices = None
country_airport_dict = None

# Upper bound used when building edges_proximity.parquet (great-circle miles).
PROXIMITY_EDGE_MAX_MILES = 250.0

SIMILARITY_Z_FEATURE_KEYS = (
    "z_connectivity_index",
    "z_log1p_num_dests",
    "z_redundancy_score",
    "z_elevation",
)

HOVER_COLS = [
    "latitude",
    "longitude",
    "display_name",
    "num_dests",
    "redundancy_score",
    "connectivity_index",
]

connection_type_choices = [
    "Carriers",
    "Statistical Similarity",
    "Proximity",
    "Shared Destinations",
    "Shared Destinations (Hub-Adjusted)",
]

CONNECTION_TYPE_DEFINITIONS = [
    (
        "Carriers",
        "Airlines linked to airports they fly to within the selected country. "
        "Useful for seeing which carriers cluster at which hubs (after the top-carrier filter).",
    ),
    (
        "Statistical Similarity",
        "Airports linked when connectivity index, redundancy score, destination count, "
        "and elevation are alike (z-scored, then compared). Link scores are 0–1 (higher = more similar).",
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


def attach_country_columns_to_edges(edges_df):
    out = edges_df.copy()
    out["source_country"] = out["source"].map(IATA_TO_COUNTRY)
    out["target_country"] = out["target"].map(IATA_TO_COUNTRY)
    return out


def build_airline_iata_to_name(airport_frame):
    mapping = {}
    for carriers in airport_frame["carriers"].dropna():
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


def build_similarity_airport_profiles():
    ap = airport_df.drop_duplicates(subset=["iata"]).copy()
    ap["num_dests"] = pd.to_numeric(ap["num_dests"], errors="coerce").fillna(0).astype(int)
    ap["connectivity_index"] = pd.to_numeric(ap["connectivity_index"], errors="coerce").fillna(0.0)
    ap["redundancy_score"] = pd.to_numeric(ap["redundancy_score"], errors="coerce").fillna(0.0)
    if "elevation" in ap.columns:
        ap["elevation"] = pd.to_numeric(ap["elevation"], errors="coerce").fillna(0.0)
    else:
        ap["elevation"] = 0.0
    ap["log1p_num_dests"] = np.log1p(ap["num_dests"].to_numpy(dtype=float))
    z_cols = ["connectivity_index", "log1p_num_dests", "redundancy_score", "elevation"]
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
            "elevation": float(mat[i, 3]),
            "z_connectivity_index": float(z_mat[i, 0]),
            "z_log1p_num_dests": float(z_mat[i, 1]),
            "z_redundancy_score": float(z_mat[i, 2]),
            "z_elevation": float(z_mat[i, 3]),
        }
    return profiles


def initialize_derived_state():
    """Build lookups and merged edge tables from raw frames loaded in app.py."""
    global IATA_TO_COUNTRY
    global graph1_merged
    global AIRLINE_IATA_TO_NAME
    global AIRPORT_IATA_META
    global SIMILARITY_AIRPORT_PROFILE
    global graph2_merged
    global graph3_merged
    global dest_tbl
    global country_choices
    global airport_choices
    global country_airport_dict

    IATA_TO_COUNTRY = (
        airport_df.drop_duplicates(subset=["iata"])
        .set_index("iata")["country"]
    )

    airport_df1 = airport_df[["iata", "country", "display_name"]].drop_duplicates()
    airport_df1.rename(columns={"iata": "airport"}, inplace=True)
    graph1_merged = pd.merge(graph1, airport_df1, on="airport")

    AIRLINE_IATA_TO_NAME = build_airline_iata_to_name(airport_df)
    AIRPORT_IATA_META = (
        airport_df.drop_duplicates(subset=["iata"])
        .set_index("iata")[["display_name", "country"]]
        .to_dict("index")
    )
    SIMILARITY_AIRPORT_PROFILE = build_similarity_airport_profiles()

    graph2_merged = attach_country_columns_to_edges(graph2)
    graph3_merged = attach_country_columns_to_edges(graph3)

    dest_tbl = (
        airport_df[
            ["display_name", "dest_name", "dest_iata", "connectivity_index", "redundancy_score"]
        ].rename(
            columns={
                "dest_name": "Destination",
                "connectivity_index": "Connectivity Index",
                "redundancy_score": "Redundancy Index",
            }
        )
    )

    country_choices = sorted(airport_df["country"].unique())
    airport_choices = sorted(airport_df["display_name"].unique())

    df_for_dict = airport_df[["country", "display_name"]]
    df_for_dict = df_for_dict.drop_duplicates(subset="display_name", keep="first")
    country_airport_dict = df_for_dict.groupby("country")["display_name"].apply(list).to_dict()


def load_merged_shared_destinations_edges():
    """Load on first Connections-tab use (~97 MB); not loaded at app startup."""
    global merged_shared_destinations_edges
    if merged_shared_destinations_edges is None:
        raw = pd.read_parquet(SHARED_DESTINATIONS_PARQUET_URL)
        merged_shared_destinations_edges = attach_country_columns_to_edges(raw)
    return merged_shared_destinations_edges


def load_merged_shared_destinations_cosine_edges():
    """Load on first Connections-tab use (~67 MB); not loaded at app startup."""
    global merged_shared_destinations_cosine_edges
    if merged_shared_destinations_cosine_edges is None:
        raw = pd.read_parquet(SHARED_DESTINATIONS_COSINE_PARQUET_URL)
        merged_shared_destinations_cosine_edges = attach_country_columns_to_edges(raw)
    return merged_shared_destinations_cosine_edges
