
import numpy as np
from sklearn.cluster import KMeans
from utils import warehousing_cost, get_drive_time_matrix

EARTH_RADIUS_MILES = 3958.8

def _haversine_vec(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype=float); lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float); lat2 = np.asarray(lat2, dtype=float)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))

def _drive_time_matrix(orig, dest, api_key):
    if not api_key:
        return None
    secs = get_drive_time_matrix(orig, dest, api_key)
    if secs is None:
        return None
    return np.array(secs) / 60.0  # minutes

def _assign(df, centers, api_key=None):
    s_lat = df['Latitude'].values
    s_lon = df['Longitude'].values
    time_mat = _drive_time_matrix(np.column_stack([s_lon, s_lat]).tolist(), centers, api_key)
    dists = np.empty((len(df), len(centers)))
    if time_mat is not None:
        dists[:] = time_mat
    else:
        for j,(clon,clat) in enumerate(centers):
            dists[:,j] = _haversine_vec(s_lon, s_lat, clon, clat) / 50.0 * 60.0
    idx = dists.argmin(axis=1)
    dist_min = dists[np.arange(len(df)), idx]
    return idx, dist_min

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
    candidate_centers=None,
    rdc_list=None,
    transfer_rate_min=0.0,
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
    use_drive_times=False,
    ors_api_key=None,
):
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    candidate_centers = candidate_centers or []
    rdc_list = rdc_list or []

    store_coords = df[['Longitude','Latitude']].values
    best = None

    for k in k_vals:
        k_eff = max(k, len(fixed_centers))
        # choose centers
        if candidate_centers:
            # greedy pick among candidate centers
            available = [c for c in candidate_centers if c not in fixed_centers]
            centers = [list(fc) for fc in fixed_centers]
            while len(centers) < k_eff and available:
                # choose site that gives best incremental reduction
                best_site = None; best_cost = None
                for cand in available:
                    test_centers = centers + [cand]
                    idx, tmin = _assign(df, test_centers, ors_api_key if use_drive_times else None)
                    cost = (tmin * df['DemandLbs'] * rate_out_min).sum()
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_site = cand
                centers.append(best_site)
                available.remove(best_site)
            # ensure length
            while len(centers) < k_eff and available:
                centers.append(available.pop())
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
            km.fit(store_coords)
            centers = km.cluster_centers_.tolist()
            for i,fc in enumerate(fixed_centers):
                centers[i] = list(fc)

        idx, tmin = _assign(df, centers, ors_api_key if use_drive_times else None)
        assigned = df.copy()
        assigned['Warehouse'] = idx
        assigned['TimeMin'] = tmin

        out_cost = (tmin * assigned['DemandLbs'] * rate_out_min).sum()

        wh_cost = 0.0
        demand_per_wh = []
        for i in range(len(centers)):
            demand = assigned.loc[assigned['Warehouse']==i,'DemandLbs'].sum()
            demand_per_wh.append(demand)
            wh_cost += warehousing_cost(demand, sqft_per_lb, cost_sqft, fixed_cost)

        total_cost = out_cost + wh_cost  # inbound & transfer omitted in simplified version

        if best is None or total_cost < best['total_cost']:
            best = {
                'assigned': assigned,
                'centers': centers,
                'out_cost': out_cost,
                'wh_cost': wh_cost,
                'in_cost': 0.0,
                'trans_cost': 0.0,
                'total_cost': total_cost,
                'demand_per_wh': demand_per_wh
            }
    return best
