
import numpy as np
from sklearn.cluster import KMeans
from utils import haversine, warehousing_cost, get_drive_time_matrix

def _minutes_from_haversine(lon1, lat1, lon2, lat2):
    miles = haversine(lon1, lat1, lon2, lat2)
    return miles / 50.0 * 60.0  # assume 50 mph average

# -------------------------------------------------------------------------
def _drive_time_matrix(orig, dest, api_key):
    if not api_key or not orig or not dest:
        return None
    try:
        secs = get_drive_time_matrix(orig, dest, api_key)
        if secs is None:
            return None
        return np.array(secs) / 60.0  # minutes
    except Exception:
        return None

def _assign(df, centers, api_key):
    s_lon = df['Longitude'].values
    s_lat = df['Latitude'].values
    mat = _drive_time_matrix(np.column_stack([s_lon, s_lat]).tolist(),
                             centers, api_key)
    if mat is None:
        dists = np.empty((len(df), len(centers)))
        for j, (lon, lat) in enumerate(centers):
            dists[:, j] = _minutes_from_haversine(s_lon, s_lat, lon, lat)
    else:
        dists = mat
    idx = dists.argmin(axis=1)
    tmin = dists[np.arange(len(df)), idx]
    return idx, tmin

# -------------------------------------------------------------------------
def _greedy_candidate_select(df, k, fixed, sites, rate_out, api_key):
    selected = fixed.copy()
    remaining = [s for s in sites if s not in selected]
    while len(selected) < k and remaining:
        best_site, best_cost = None, None
        for cand in remaining:
            test = selected + [cand]
            cost = _compute_outbound(df, test, rate_out, api_key)[0]
            if best_cost is None or cost < best_cost:
                best_cost, best_site = cost, cand
        selected.append(best_site)
        remaining.remove(best_site)
    return selected

# -------------------------------------------------------------------------
def _compute_outbound(df, centers, rate_out, api_key):
    idx, tmin = _assign(df, centers, api_key)
    outbound_cost = (df['DemandLbs'] * tmin * rate_out).sum()
    return outbound_cost, idx, tmin

# -------------------------------------------------------------------------
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
    rdc_list=None,
    transfer_rate_min=0.0,
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
    use_drive_times=False,
    ors_api_key=None,
    candidate_sites=None,
):
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []
    best = None

    for k in k_vals:
        k_eff = max(k, len(fixed_centers))

        # ----- choose center locations -----------------------------------
        if candidate_sites and len(candidate_sites) >= k_eff:
            centers = _greedy_candidate_select(
                df, k_eff, fixed_centers, candidate_sites,
                rate_out_min, ors_api_key if use_drive_times else None
            )
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(df[['Longitude', 'Latitude']])
            centers = km.cluster_centers_.tolist()
            # override with fixed centers
            for i_fc, fc in enumerate(fixed_centers):
                centers[i_fc] = fc

        # assignment
        idx, tmin = _assign(df, centers, ors_api_key if use_drive_times else None)
        assigned = df.copy()
        assigned['Warehouse'] = idx
        assigned['TimeMin'] = tmin

        # outbound
        out_cost = (assigned['DemandLbs'] * tmin * rate_out_min).sum()

        # warehousing
        demand_per_wh = []
        wh_cost = 0.0
        for i in range(len(centers)):
            dem = assigned.loc[assigned['Warehouse'] == i, 'DemandLbs'].sum()
            demand_per_wh.append(dem)
            wh_cost += warehousing_cost(dem, sqft_per_lb, cost_sqft, fixed_cost)

        # inbound
        in_cost = 0.0
        if consider_inbound and inbound_pts:
            c_coords = centers
            for lon, lat, pct in inbound_pts:
                mat = _drive_time_matrix([[lon, lat]], c_coords,
                                         ors_api_key if use_drive_times else None)
                if mat is None:
                    mins = [_minutes_from_haversine(lon, lat, cx, cy) for cx, cy in c_coords]
                else:
                    mins = mat[0]
                in_cost += (np.array(mins) * np.array(demand_per_wh) * pct * inbound_rate_min).sum()

        # transfer (simple model)
        trans_cost = 0.0
        rdc_only = [r for r in rdc_list if not r['is_sdc']]
        if rdc_only:
            wh_coords = centers
            r_coords = [r['coords'] for r in rdc_only]
            mat = _drive_time_matrix(r_coords, wh_coords,
                                     ors_api_key if use_drive_times else None)
            if mat is None:
                mat = np.array([[ _minutes_from_haversine(rx, ry, wx, wy)
                                  for wx, wy in wh_coords]
                                 for rx, ry in r_coords])
            share = 1.0 / len(r_coords)
            for row in mat:
                trans_cost += (row * np.array(demand_per_wh) * share * transfer_rate_min).sum()

        total_cost = out_cost + wh_cost + in_cost + trans_cost

        if best is None or total_cost < best['total_cost']:
            best = dict(
                centers=centers,
                assigned=assigned,
                demand_per_wh=demand_per_wh,
                total_cost=total_cost,
                out_cost=out_cost,
                in_cost=in_cost,
                trans_cost=trans_cost,
                wh_cost=wh_cost,
            )
    return best
