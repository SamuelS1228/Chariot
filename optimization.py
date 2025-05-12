
import numpy as np
from sklearn.cluster import KMeans
from utils import warehousing_cost, get_drive_time_matrix

EARTH_RADIUS_MILES = 3958.8

def _haversine_vec(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))

# ─────────────────────────────────────────────────────────────
# OpenRouteService helpers
# ─────────────────────────────────────────────────────────────
def _drive_time_matrix(orig, dest, api_key):
    if (api_key is None) or (api_key == ""):
        return None
    try:
        secs = get_drive_time_matrix(orig, dest, api_key)
        if secs is None:
            return None
        return np.array(secs) / 60.0  # seconds → minutes
    except Exception as e:
        print("drive‑time matrix error:", e)
        return None

def _drive_time_single(lon1, lat1, lon2, lat2, api_key):
    mat = _drive_time_matrix([[lon1, lat1]], [[lon2, lat2]], api_key)
    if mat is not None:
        return float(mat[0][0])
    miles = _haversine_vec([lon1], [lat1], [lon2], [lat2])[0]
    return miles / 50.0 * 60.0

# ─────────────────────────────────────────────────────────────
# Inbound + transfer helpers
def _inbound_cost(demand_per_wh, inbound_pts, centers, rate, api_key):
    if not inbound_pts or rate == 0:
        return 0.0
    centers_arr = np.asarray(centers)
    c_lon = centers_arr[:,0]; c_lat = centers_arr[:,1]
    demand_per_wh = np.asarray(demand_per_wh, dtype=float)
    cost = 0.0
    for lon, lat, pct in inbound_pts:
        times = _drive_time_matrix([[lon, lat]], centers, api_key)
        if times is None:
            dist = _haversine_vec(np.full_like(c_lon, lon), np.full_like(c_lat, lat), c_lon, c_lat)
            times = dist/50.0*60.0
        else:
            times = times[0]
        cost += (times * demand_per_wh * pct * rate).sum()
    return cost

# ─────────────────────────────────────────────────────────────
def _assign(df, centers, api_key):
    s_lat = df['Latitude'].values
    s_lon = df['Longitude'].values
    time_mat = _drive_time_matrix(np.column_stack([s_lon, s_lat]).tolist(), centers, api_key)
    if time_mat is None:
        # fallback
        dists = np.empty((len(df), len(centers)))
        for j,(lon,lat) in enumerate(centers):
            dists[:,j] = _haversine_vec(s_lon, s_lat, lon, lat)/50.0*60.0
    else:
        dists = time_mat
    idx = dists.argmin(axis=1)
    tmin = dists[np.arange(len(df)), idx]
    return idx, tmin

# ─────────────────────────────────────────────────────────────
def _total_cost(df, centers, rate_out_min, sqft_per_lb, cost_sqft, fixed_cost,
                inbound_pts, inbound_rate_min, ors_api_key):
    idx, tmin = _assign(df, centers, ors_api_key)
    assigned = df.copy()
    assigned['Warehouse'] = idx
    assigned['TimeMin'] = tmin

    out_cost = (assigned['TimeMin'] * assigned['DemandLbs'] * rate_out_min).sum()

    wh_cost = 0.0
    demand_per_wh = []
    for i in range(len(centers)):
        demand = assigned.loc[assigned['Warehouse']==i, 'DemandLbs'].sum()
        demand_per_wh.append(demand)
        wh_cost += warehousing_cost(demand, sqft_per_lb, cost_sqft, fixed_cost)

    in_cost = _inbound_cost(demand_per_wh, inbound_pts, centers, inbound_rate_min, ors_api_key)

    total = out_cost + wh_cost + in_cost
    return {
        'assigned': assigned,
        'out_cost': out_cost,
        'wh_cost': wh_cost,
        'in_cost': in_cost,
        'trans_cost': 0.0,
        'total_cost': total,
        'demand_per_wh': demand_per_wh
    }

# ─────────────────────────────────────────────────────────────
def _greedy_select(df, k, fixed_centers, candidate_centers,
                   *cost_args):
    selected = [list(fc) for fc in fixed_centers]
    remaining = [c for c in candidate_centers if c not in selected]
    while len(selected) < k and remaining:
        best_c = None
        best_cost = None
        for cand in remaining:
            trial = selected + [cand]
            cost = _total_cost(df, trial, *cost_args)['total_cost']
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_c = cand
        if best_c is None:
            break
        selected.append(best_c)
        remaining.remove(best_c)
    # if still fewer than k (e.g., not enough candidates), pad with remaining stores via kmeans
    if len(selected) < k:
        extra_needed = k - len(selected)
        store_coords = df[['Longitude','Latitude']].values
        km = KMeans(n_clusters=extra_needed, n_init=5, random_state=42)
        km.fit(store_coords)
        for c in km.cluster_centers_.tolist():
            selected.append([float(c[0]), float(c[1])])
            if len(selected) == k:
                break
    return selected
# ─────────────────────────────────────────────────────────────
def optimize(
    df,
    k_vals,
    rate_out_min,
    sqft_per_lb,
    cost_sqft,
    fixed_cost,
    consider_inbound=False,
    inbound_rate_min=0.0,
    inbound_pts=None,
    fixed_centers=None,
    rdc_list=None,  # preserved but not implemented in this minimal rebuild
    transfer_rate_min=0.0,  # for compatibility
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
    use_drive_times=False,
    ors_api_key="",
    restrict_candidates=False,
    candidate_centers=None,
):
    """
    Optimizes warehouse placement.
    If restrict_candidates=True, centers are chosen from candidate_centers (plus any fixed_centers).
    Otherwise k-means (with fixed overrides) is used.
    """
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    candidate_centers = candidate_centers or []

    best = None
    best_k = None
    best_centers = None

    for k in k_vals:
        k_eff = max(k, len(fixed_centers))
        if restrict_candidates and candidate_centers:
            centers = _greedy_select(
                df, k_eff, fixed_centers, candidate_centers,
                rate_out_min, sqft_per_lb, cost_sqft, fixed_cost,
                inbound_pts, inbound_rate_min,
                ors_api_key if use_drive_times else None
            )
        else:
            # k-means
            store_coords = df[['Longitude','Latitude']].values
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
            km.fit(store_coords)
            centers = [[float(c[0]), float(c[1])] for c in km.cluster_centers_.tolist()]
            # override with fixed centers
            for i,fc in enumerate(fixed_centers):
                centers[i] = [float(fc[0]), float(fc[1])]

        res = _total_cost(
            df, centers, rate_out_min, sqft_per_lb, cost_sqft, fixed_cost,
            inbound_pts, inbound_rate_min,
            ors_api_key if use_drive_times else None
        )
        if (best is None) or (res['total_cost'] < best['total_cost']):
            best = res
            best_centers = centers
            best_k = k_eff

    best['k'] = best_k
    best['centers'] = best_centers
    return best
