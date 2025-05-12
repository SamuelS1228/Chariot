
import math
from typing import List, Optional

EARTH_RADIUS_MILES = 3958.8

def haversine(lon1, lat1, lon2, lat2):
    """Great‑circle distance in miles (scalar)."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MILES * 2 * math.asin(math.sqrt(a))

def warehousing_cost(demand_lbs: float, sqft_per_lb: float, cost_per_sqft: float, fixed_cost: float) -> float:
    """Annual warehousing cost for a single site."""
    return fixed_cost + demand_lbs * sqft_per_lb * cost_per_sqft


# ──────────────────────────── OpenRouteService ────────────────────────────
try:
    import openrouteservice
except ImportError:  # allow utils to import even if ORS isn't installed yet
    openrouteservice = None

def get_drive_time_matrix(origins: List[List[float]], destinations: List[List[float]], api_key: str) -> Optional[List[List[float]]]:
    """Return a duration matrix (seconds) between origins and destinations using ORS.
    Returns None if ORS unavailable or an API error occurs."""
    if openrouteservice is None or not api_key:
        return None
    client = openrouteservice.Client(key=api_key)
    try:
        matrix = client.distance_matrix(
            locations=origins + destinations,
            profile="driving-car",
            metrics=["duration"],
            sources=list(range(len(origins))),
            destinations=list(range(len(origins), len(origins) + len(destinations))),
            units="m",
        )
        return matrix["durations"]
    except Exception as e:
        print("ORS error:", e)
        return None
