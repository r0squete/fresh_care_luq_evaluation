"""
Utility functions for LUQ calculation.
arosquete - 2026/02/13
"""

# Imports
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

from config import EPOCH, LAT_MAX, R_EARTH


# Functions
def to_epoch_seconds_datetime64(t64, epoch=EPOCH):
    """
    Convert datetime64 to seconds since epoch.
    """
    t64s = t64.astype("datetime64[s]")
    return (t64s - epoch).astype("timedelta64[s]").astype(np.int64)


def load_dataset_files(
    dataset_path, pattern="*.nc", eval_start=None, eval_end=None, tau_max_sec=None
):
    """
    Load and concatenate dataset files for specified time range.
    """
    files = sorted(glob.glob(str(Path(dataset_path) / pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {dataset_path}")

    # If no time filtering, load all files
    if eval_start is None and eval_end is None:
        return xr.open_mfdataset(files, combine="by_coords")

    # Calculate required time range
    if eval_start:
        start_dt = pd.to_datetime(eval_start, utc=True)
        start_year = start_dt.year
    else:
        start_year = None

    if eval_end:
        end_dt = pd.to_datetime(eval_end, utc=True)
        # Add tau_max buffer in days
        if tau_max_sec:
            end_dt += pd.Timedelta(seconds=tau_max_sec)
        end_year = end_dt.year
    else:
        end_year = None

    # Filter files by date if naming convention allows
    filtered_files = []
    for file in files:
        filename = Path(file).name

        # Try to extract date from filename (common patterns)
        # Pattern 1: YYYY or YYYYMM in filename

        year_match = re.search(r"(\d{4})", filename)
        if year_match:
            file_year = int(year_match.group(1))

            # Simple year-based filtering
            include_file = True
            if start_year and file_year < start_year:
                include_file = False
            if end_year and file_year > end_year:
                include_file = False

            if include_file:
                filtered_files.append(file)
        else:
            # If can't parse date, include file
            filtered_files.append(file)

    # Use filtered files or all if filtering didn't work
    files_to_load = filtered_files if filtered_files else files

    print(f"  Loading {len(files_to_load)} of {len(files)} files")
    return xr.open_mfdataset(files_to_load, combine="by_coords")


def load_drifter_file(drifter_file):
    """
    Load single drifter CSV file.
    """
    df = pd.read_csv(
        drifter_file,
        skiprows=6,
        sep=",",
        names=[
            "time",
            "latitude",
            "longitude",
            "ve",
            "vn",
            "err_ve",
            "err_vn",
            "sst",
            "err_sst",
            "flg_sst",
            "speed",
            "direction",
        ],
        header=0,
    )
    return df


def prep_dataset(
    ds,
    u_name="ug",
    v_name="vg",
    time_name="time",
    lat_name="lat",
    lon_name="lon",
    keep_uv=False,
):
    """
    Currents dataset preparation + angular conversion on the grid.

    Always returns:
      - time_sec (int64): seconds since epoch 1950-01-01
      - lat_rad, lon_rad (float32): radians
      - lon_dot, lat_dot (float32): rad/s on grid

    Optionally returns u,v (float32) if keep_uv=True.

    Output dict keys:
      time_sec, lat_rad, lon_rad, lon_dot, lat_dot, (optional) u, v
    """
    # coords
    time = ds[time_name].values
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # absolute seconds since epoch (int64)
    time_sec = to_epoch_seconds_datetime64(time)

    lat_rad = np.deg2rad(lat.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )
    lon_rad = np.deg2rad(lon.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )

    u = ds[u_name].values.astype(np.float32, copy=False)
    v = ds[v_name].values.astype(np.float32, copy=False)

    R = np.float32(6371000.0)
    coslat = np.cos(lat_rad)
    coslat = np.maximum(coslat, np.float32(1e-6))

    lon_dot = u / (R * coslat[None, :, None])  # rad/s
    lat_dot = v / R  # rad/s

    out = {
        "time_sec": time_sec,  # int64
        "lat_rad": lat_rad,
        "lon_rad": lon_rad,
        "lon_dot": lon_dot.astype(np.float32, copy=False),
        "lat_dot": lat_dot.astype(np.float32, copy=False),
    }
    if keep_uv:
        out["u"] = u
        out["v"] = v
    return out


def prep_drifter(df, time_col="time", lon_col="longitude", lat_col="latitude"):
    """
    Drifter prep:
      - time -> int64 seconds since epoch 1950-01-01
      - lon/lat (deg) -> float32 radians

    Returns dict:
      {
        "time_sec": int64 (n,)
        "lon_rad":  float32 (n,)
        "lat_rad":  float32 (n,)
      }
    """
    t_pd = pd.to_datetime(df[time_col], utc=True, errors="raise")
    t64 = t_pd.dt.tz_convert(None).to_numpy(dtype="datetime64[s]")

    time_sec = to_epoch_seconds_datetime64(t64)

    lon = df[lon_col].to_numpy(dtype=np.float32)
    lat = df[lat_col].to_numpy(dtype=np.float32)

    lon_rad = np.deg2rad(lon).astype(np.float32, copy=False)
    lat_rad = np.deg2rad(lat).astype(np.float32, copy=False)

    return {"time_sec": time_sec, "lon_rad": lon_rad, "lat_rad": lat_rad}


def validate_temporal_coverage(drifter_data, field_data, tau_max_sec):
    """
    Simple validation: check if drifter+tau_max fits within field data

    Returns:
    --------
    is_valid : bool
    message : str
    """
    drifter_start = drifter_data["time_sec"][0]
    drifter_end = drifter_data["time_sec"][-1]
    field_start = field_data["time_sec"][0]
    field_end = field_data["time_sec"][-1]

    # Need drifter to start after field starts
    if drifter_start < field_start:
        return False, "Drifter starts before field data"

    # Need drifter+tau_max to end before field ends
    required_end = drifter_end + tau_max_sec
    if required_end > field_end:
        return (
            False,
            f"Need field data until {required_end}, only have until {field_end}",
        )

    # Simple overlap check
    overlap_days = (field_end - drifter_start) / (24 * 3600)
    return True, f"OK: {overlap_days:.1f} days available"


def filter_by_temporal_range(drifter_data, eval_start=None, eval_end=None):
    """
    Filter drifter data by temporal range

    Parameters:
    -----------
    drifter_data : dict
        Prepared drifter data with 'time_sec' key
    eval_start : str, optional
        Start date "YYYY-MM-DD"
    eval_end : str, optional
        End date "YYYY-MM-DD"

    Returns:
    --------
    filtered_data : dict or None
        Filtered drifter data, None if no data in range
    """
    if eval_start is None and eval_end is None:
        return drifter_data  # No filtering needed

    time_sec = drifter_data["time_sec"]

    # Convert string dates to epoch seconds
    if eval_start:
        start_dt = pd.to_datetime(eval_start, utc=True)
        start_sec = to_epoch_seconds_datetime64(np.array([start_dt]))[0]
    else:
        start_sec = time_sec[0]

    if eval_end:
        end_dt = pd.to_datetime(eval_end, utc=True)
        end_sec = to_epoch_seconds_datetime64(np.array([end_dt]))[0]
    else:
        end_sec = time_sec[-1]

    # Filter indices
    mask = (time_sec >= start_sec) & (time_sec <= end_sec)

    if not mask.any():
        return None  # No data in temporal range

    return {
        "time_sec": time_sec[mask],
        "lon_rad": drifter_data["lon_rad"][mask],
        "lat_rad": drifter_data["lat_rad"][mask],
    }


def spherical_distance_km_acos(lon1, lat1, lon2, lat2):
    """
    Great-circle distance using spherical law of cosines.
    Inputs in radians. Works with arrays (lon1/lat1) and scalar target (lon2/lat2).
    Returns distance in km.
    """
    # cos(angle) = sinφ1 sinφ2 + cosφ1 cosφ2 cos(Δλ)
    cosang = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(
        lon1 - lon2
    )

    # numerical safety
    cosang = np.clip(cosang, -1.0, 1.0)

    return R_EARTH / 1000 * np.arccos(cosang)


def L_characteristic_km(lon_rad, lat_rad, i0=0, i1=None, dist_fn=None):
    """
    L = max great-circle distance between any two drifter positions
        within the index window [i0, i1).

    Inputs
    ------
    lon_rad, lat_rad : 1D arrays (radians)
    i0, i1           : window indices
    dist_fn          : distance function returning km
                       signature: dist_fn(lon_array, lat_array, lon_scalar, lat_scalar) -> km array

    Output
    ------
    L : float (km)
    """
    if i1 is None:
        i1 = len(lon_rad)

    if dist_fn is None:
        dist_fn = spherical_distance_km_acos  # must return km

    lon = lon_rad[i0:i1]
    lat = lat_rad[i0:i1]
    n = len(lon)

    Lmax = 0.0
    for i in range(n - 1):
        # vectorized distances from point i to points i+1..end
        d = dist_fn(lon[i + 1 :], lat[i + 1 :], lon[i], lat[i])
        m = np.nanmax(d)
        if m > Lmax:
            Lmax = float(m)

    return Lmax


def make_neighborhood(lon0_rad, lat0_rad, r_km, nx=9, ny=9):
    """
    Create a local square neighborhood centered at (lon0, lat0) with half-size r_km.

    Inputs
    ------
    lon0_rad, lat0_rad : float (radians)
    r_km               : float
    nx, ny             : int  (grid points)

    Output
    ------
    seed_lon_rad, seed_lat_rad : float32 arrays of shape (nx*ny,)
    """
    r_m = np.float32(r_km * 1000.0)

    # local metric factors (tangent plane approx)
    coslat = np.cos(np.float64(lat0_rad))
    if coslat < 1e-6:
        coslat = 1e-6

    dlon = (r_m / (R_EARTH * coslat)).astype(np.float32)  # radians
    dlat = (r_m / R_EARTH).astype(np.float32)  # radians

    # offsets in [-dlon, dlon] and [-dlat, dlat]
    xs = np.linspace(-dlon, dlon, nx, dtype=np.float32)
    ys = np.linspace(-dlat, dlat, ny, dtype=np.float32)

    # mesh (ny,nx) then flatten
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    seed_lon = np.float32(lon0_rad) + xg
    seed_lat = np.float32(lat0_rad) + yg

    return seed_lon.ravel(), seed_lat.ravel()


@njit
def _find_index_uniform(x, x0, dx, n):
    """
    Return i such that x is inside [x_i, x_{i+1}] for a uniform grid.
    If x is out-of-range (cannot bracket), return -1.
    """
    r = (x - x0) / dx  # continuous index
    i = int(np.floor(r))  # lower bracket index

    # out-of-range => invalid
    if i < 0:
        return -1
    if i >= n - 1:  # need i+1 to exist
        return -1
    return i


@njit(parallel=True)
def interp_trilinear_uv(time_sec, lat, lon, u, v, t, lat_q, lon_q):
    """
    Trilinear interpolation (linear in time, lat, lon).

    Inputs
    ------
    time_sec : 1D array [nt] (seconds since epoch 1950, int64)
    lat, lon : 1D arrays [nlat], [nlon] in radians (uniform spacing)
    u, v     : 3D arrays [nt, nlat, nlon] in m/s
    t        : scalar query time (seconds since epoch 1950, float64)
    lat_q,
    lon_q    : query positions (same shape), in radians

    Output
    ------
    u_q, v_q : interpolated velocities (same shape as lat_q/lon_q)
              out-of-domain -> NaN (handled later by mask logic).
    """
    nt = time_sec.size
    nlat = lat.size
    nlon = lon.size

    # grid start + spacing (assumes uniform grids) - convert to float64 for calculations
    t0 = np.float64(time_sec[0])
    dtg = np.float64(time_sec[1] - time_sec[0])
    lat0 = lat[0]
    dlat = lat[1] - lat[0]
    lon0 = lon[0]
    dlon = lon[1] - lon[0]

    # time bracket
    it = _find_index_uniform(np.float64(t), t0, dtg, nt)
    if it < 0:
        # out of time range -> return all NaNs
        u_out = np.empty_like(lat_q)
        v_out = np.empty_like(lat_q)
        for k in prange(lat_q.size):
            u_out.ravel()[k] = np.nan
            v_out.ravel()[k] = np.nan
        return u_out, v_out

    # time weight in [0,1] - use float64 for precision
    wt = (np.float64(t) - (t0 + np.float64(it) * dtg)) / dtg

    # outputs
    u_out = np.empty_like(lat_q)
    v_out = np.empty_like(lat_q)

    # flatten for a single fast loop
    lat_flat = lat_q.ravel()
    lon_flat = lon_q.ravel()
    u_flat = u_out.ravel()
    v_flat = v_out.ravel()

    # main loop (each query point)
    for k in prange(lat_flat.size):
        ph = lat_flat[k]  # query latitude (rad)
        la = lon_flat[k]  # query longitude (rad)

        # spatial brackets
        iy = _find_index_uniform(ph, lat0, dlat, nlat)
        ix = _find_index_uniform(la, lon0, dlon, nlon)

        if iy < 0 or ix < 0:
            # out of spatial range -> NaN
            u_flat[k] = np.nan
            v_flat[k] = np.nan
            continue

        # weights in [0,1]
        wy = (ph - (lat0 + iy * dlat)) / dlat
        wx = (la - (lon0 + ix * dlon)) / dlon

        # 8 corners of the (t,lat,lon) cube
        # time it (t0 slice)
        u000 = u[it, iy, ix]
        u001 = u[it, iy, ix + 1]
        u010 = u[it, iy + 1, ix]
        u011 = u[it, iy + 1, ix + 1]
        # time it+1 (t1 slice)
        u100 = u[it + 1, iy, ix]
        u101 = u[it + 1, iy, ix + 1]
        u110 = u[it + 1, iy + 1, ix]
        u111 = u[it + 1, iy + 1, ix + 1]

        v000 = v[it, iy, ix]
        v001 = v[it, iy, ix + 1]
        v010 = v[it, iy + 1, ix]
        v011 = v[it, iy + 1, ix + 1]

        v100 = v[it + 1, iy, ix]
        v101 = v[it + 1, iy, ix + 1]
        v110 = v[it + 1, iy + 1, ix]
        v111 = v[it + 1, iy + 1, ix + 1]

        # bilinear in (lat,lon) for each time slice
        u0 = (1 - wy) * ((1 - wx) * u000 + wx * u001) + wy * (
            (1 - wx) * u010 + wx * u011
        )
        u1 = (1 - wy) * ((1 - wx) * u100 + wx * u101) + wy * (
            (1 - wx) * u110 + wx * u111
        )

        v0 = (1 - wy) * ((1 - wx) * v000 + wx * v001) + wy * (
            (1 - wx) * v010 + wx * v011
        )
        v1 = (1 - wy) * ((1 - wx) * v100 + wx * v101) + wy * (
            (1 - wx) * v110 + wx * v111
        )

        # linear in time - use float64 weight
        u_flat[k] = (np.float64(1.0) - wt) * u0 + wt * u1
        v_flat[k] = (np.float64(1.0) - wt) * v0 + wt * v1

    return u_out, v_out


@njit
def advect_seeds_rk4(
    time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t0, tau_sec, dt_sec, lon0, lat0
):
    """
    Advect ns seeds from (lon0,lat0) at t0 for tau_sec using fixed-step RK4.

    Inputs
    ------
    time_sec : 1D array [nt] (uniform spacing, seconds since epoch 1950, int64)
    lat_grid, lon_grid : 1D float32 arrays [ny], [nx] in radians (uniform)
    lon_dot, lat_dot : float32 arrays [nt,ny,nx] in rad/s
    t0 : scalar seconds since epoch 1950 (int64 or float64)
    tau_sec : scalar seconds to integrate forward
    dt_sec : scalar time step (e.g. 3600, 7200, 10800)
    lon0, lat0 : float32 arrays [ns] initial seed positions (radians)

    Returns
    -------
    lonT, latT : float32 arrays [ns]
    valid : uint8 array [ns] (1 valid, 0 invalid)
    """
    ns = lon0.size
    lon = lon0.copy()
    lat = lat0.copy()
    valid = np.ones(ns, dtype=np.uint8)

    # reusable temporaries
    lon_a = np.empty(ns, dtype=np.float32)
    lat_a = np.empty(ns, dtype=np.float32)
    lon_b = np.empty(ns, dtype=np.float32)
    lat_b = np.empty(ns, dtype=np.float32)
    lon_c = np.empty(ns, dtype=np.float32)
    lat_c = np.empty(ns, dtype=np.float32)

    nsteps = int(np.ceil(tau_sec / dt_sec))
    t = np.float64(t0)  # convert to float64 for internal calculations

    for step in range(nsteps):
        # shorter last step if needed
        dt = np.float64(dt_sec)
        if step == nsteps - 1:
            rem = np.float64(tau_sec - (nsteps - 1) * dt_sec)
            if rem > 0:
                dt = rem

        half = np.float64(0.5) * dt
        sixth = dt / np.float64(6.0)

        # k1 at (t, lon, lat)
        k1_lon, k1_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t, lat, lon
        )

        # A = y + dt/2*k1
        for i in range(ns):
            if valid[i] == 0:
                lon_a[i] = lon[i]
                lat_a[i] = lat[i]
                continue
            if np.isnan(k1_lon[i]) or np.isnan(k1_lat[i]):
                valid[i] = 0
                lon_a[i] = lon[i]
                lat_a[i] = lat[i]
                continue
            la = lat[i] + np.float32(half * k1_lat[i])
            if la > LAT_MAX or la < -LAT_MAX:
                valid[i] = 0
                lon_a[i] = lon[i]
                lat_a[i] = lat[i]
                continue
            lon_a[i] = lon[i] + np.float32(half * k1_lon[i])
            lat_a[i] = la

        # k2 at (t+dt/2, A)
        k2_lon, k2_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t + half, lat_a, lon_a
        )

        # B = y + dt/2*k2
        for i in range(ns):
            if valid[i] == 0:
                lon_b[i] = lon[i]
                lat_b[i] = lat[i]
                continue
            if np.isnan(k2_lon[i]) or np.isnan(k2_lat[i]):
                valid[i] = 0
                lon_b[i] = lon[i]
                lat_b[i] = lat[i]
                continue
            lb = lat[i] + np.float32(half * k2_lat[i])
            if lb > LAT_MAX or lb < -LAT_MAX:
                valid[i] = 0
                lon_b[i] = lon[i]
                lat_b[i] = lat[i]
                continue
            lon_b[i] = lon[i] + np.float32(half * k2_lon[i])
            lat_b[i] = lb

        # k3 at (t+dt/2, B)
        k3_lon, k3_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t + half, lat_b, lon_b
        )

        # C = y + dt*k3
        for i in range(ns):
            if valid[i] == 0:
                lon_c[i] = lon[i]
                lat_c[i] = lat[i]
                continue
            if np.isnan(k3_lon[i]) or np.isnan(k3_lat[i]):
                valid[i] = 0
                lon_c[i] = lon[i]
                lat_c[i] = lat[i]
                continue
            lc = lat[i] + np.float32(dt * k3_lat[i])
            if lc > LAT_MAX or lc < -LAT_MAX:
                valid[i] = 0
                lon_c[i] = lon[i]
                lat_c[i] = lat[i]
                continue
            lon_c[i] = lon[i] + np.float32(dt * k3_lon[i])
            lat_c[i] = lc

        # k4 at (t+dt, C)
        k4_lon, k4_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t + dt, lat_c, lon_c
        )

        # update
        for i in range(ns):
            if valid[i] == 0:
                continue
            if np.isnan(k4_lon[i]) or np.isnan(k4_lat[i]):
                valid[i] = 0
                continue

            dlon = np.float32(
                sixth * (k1_lon[i] + 2.0 * k2_lon[i] + 2.0 * k3_lon[i] + k4_lon[i])
            )
            dlat = np.float32(
                sixth * (k1_lat[i] + 2.0 * k2_lat[i] + 2.0 * k3_lat[i] + k4_lat[i])
            )

            lat_new = lat[i] + dlat
            if lat_new > LAT_MAX or lat_new < -LAT_MAX:
                valid[i] = 0
                continue

            lon[i] = lon[i] + dlon
            lat[i] = lat_new

        t = t + dt

    return lon, lat, valid


def luq_case(
    field,
    t0_sec,
    tau_sec,
    dt_sec,
    lon0_rad,
    lat0_rad,
    lon_target_rad,
    lat_target_rad,
    r_km,
    nx=100,
    ny=100,
    min_valid_frac=0.8,
    return_map=False,
    fill_invalid=np.nan,
):
    """
    One LUQ case:
      neighborhood -> advect -> great-circle distance to target -> mean.

    Inputs
    ------
    field: dict from prep_dataset
    keys: time_sec, lat_rad, lon_rad, lon_dot, lat_dot
    t0_sec: int64 seconds since epoch 1950
    tau_sec: int/float seconds forward
    dt_sec: int/float integration internal step (e.g. 7200)
    lon0_rad, lat0_rad: float32 center (drifter at t0), radians
    lon_target_rad, lat_target_rad: float32 target (drifter at t0+tau), radians
    r_km: neighborhood half-size in km
    nx, ny: neighborhood grid size
    min_valid_frac: discard if too many seeds invalid
    return_map: bool, if True returns full spatial map
    fill_invalid: value for invalid grid points when return_map=True

    Returns
    -------
    If return_map=False (default):
        luq_mean_km: float32 (NaN if not enough valid)
        valid_frac: float32

    If return_map=True:
        luq_mean_km: float32 (NaN if not enough valid)
        valid_frac: float32
        seed_lon_grid: (ny,nx) array in radians
        seed_lat_grid: (ny,nx) array in radians
        LUQ_grid_km: (ny,nx) array LUQ values in km
        valid_grid: (ny,nx) bool array
    """
    # 1) neighborhood seeds (float32 arrays)
    seed_lon, seed_lat = make_neighborhood(lon0_rad, lat0_rad, r_km, nx=nx, ny=ny)

    # 2) advect seeds
    lonT, latT, valid = advect_seeds_rk4(
        field["time_sec"],
        field["lat_rad"],
        field["lon_rad"],
        field["lon_dot"],
        field["lat_dot"],
        t0_sec,
        tau_sec,
        dt_sec,
        seed_lon,
        seed_lat,
    )

    valid_mask = valid == 1
    valid_frac = valid_mask.mean().astype(np.float32)

    # 3) distances to target
    d = np.full(seed_lon.shape, fill_invalid, dtype=np.float32)
    if valid_mask.sum() > 0:
        d[valid_mask] = spherical_distance_km_acos(
            lonT[valid_mask], latT[valid_mask], lon_target_rad, lat_target_rad
        ).astype(np.float32)

    # Mean calculation (only valid points)
    if valid_mask.sum() == 0 or valid_frac < min_valid_frac:
        luq_mean_km = np.float32(np.nan)
    else:
        luq_mean_km = np.nanmean(d[valid_mask]).astype(np.float32)

    # Return based on flag
    if not return_map:
        return luq_mean_km, np.float32(valid_frac)
    else:
        # Reshape to grids for plotting
        seed_lon_grid = seed_lon.reshape(ny, nx)
        seed_lat_grid = seed_lat.reshape(ny, nx)
        LUQ_grid = d.reshape(ny, nx)
        valid_grid = valid_mask.reshape(ny, nx)

        return (
            luq_mean_km,
            np.float32(valid_frac),
            seed_lon_grid,
            seed_lat_grid,
            LUQ_grid,
            valid_grid,
        )


def save_results_csv(results, output_dir, tau_days, r_kms):
    """
    Save results as CSV - one per drifter.

    Inputs:
    -------
    results: dict
        The evaluation results to save.
    output_dir: str
        The directory to save the CSV files.
    tau_days: list
        The list of tau values in days.
    r_kms: list
        The list of radius values in kilometers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for drifter_idx, drifter_results in results.items():
        rows = []
        filename = drifter_results["filename"]
        L_char = drifter_results["L_characteristic_km"]

        for dataset_name in drifter_results.keys():
            if dataset_name in ["filename", "L_characteristic_km"]:
                continue

            for r_km in r_kms:
                for tau_day in tau_days:
                    tau_sec = np.int64(tau_day * 24 * 3600)

                    if tau_sec in drifter_results[dataset_name][r_km]:
                        data = drifter_results[dataset_name][r_km][tau_sec]
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "tau_days": tau_day,
                                "r_km": r_km,
                                "luq_mean_km": data["luq_mean_km"],
                                "luq_normalized": data["luq_normalized"],
                                "valid_frac": data["valid_frac"],
                                "n_cases": data["n_cases"],
                                "L_characteristic_km": L_char,
                            }
                        )

        if rows:
            df = pd.DataFrame(rows)
            output_file = (
                output_path
                / f"drifter_{drifter_idx:03d}_{filename.replace('.csv', '')}.csv"
            )
            df.to_csv(output_file, index=False)
            print(f"  ✓ {output_file.name}")
