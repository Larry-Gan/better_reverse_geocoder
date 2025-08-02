# processing.py
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
from geopy.distance import geodesic
from tqdm import tqdm
import numpy as np
import config # Import config variables
from kgcpy import lookupCZ 
import mappings 
import rasterio
from rasterio.windows import Window

def join_administrative_regions(points_gdf, countries_gdf, admin1_gdf, land_geom):
    """
    Spatially joins points with country and admin1 polygons.
    - Performs initial 'within' joins.
    - For points missing country data after 'within', checks distance to land
      and performs 'sjoin_nearest' for countries if close.
    - For points missing admin1 data after 'within', checks distance to land
      and performs 'sjoin_nearest' for admin1 if close.

    Args:
        points_gdf: GeoDataFrame of input points for the current chunk.
        countries_gdf: GeoDataFrame of countries.
        admin1_gdf: GeoDataFrame of Admin 1 regions.
        land_geom: Shapely geometry representing unified land area.

    Returns:
        pandas.DataFrame: DataFrame with added country and admin1 attribute
                          columns, indexed the same as points_gdf.
                          Values will be None if no join occurred.
    """
    print("Performing spatial joins for administrative regions...")

    # --- Get column names ---
    country_cols = countries_gdf.columns.difference(['geometry']).tolist()
    admin1_cols = admin1_gdf.columns.difference(['geometry']).tolist()
    all_admin_cols = sorted(list(set(country_cols + admin1_cols)))

    # --- Initialize results ---
    results_df = pd.DataFrame(index=points_gdf.index, columns=all_admin_cols).astype(object)

    # --- 1. Initial 'within' Joins ---
    print("Performing initial 'within' joins...")
    if country_cols:
        joined_country = gpd.sjoin(points_gdf[['geometry']], countries_gdf[country_cols + ['geometry']], how='left', predicate='within')
        joined_country = joined_country[~joined_country.index.duplicated(keep='first')]
        results_df.update(joined_country.dropna(subset=['index_right'])[country_cols])
    if admin1_cols:
        joined_admin1 = gpd.sjoin(points_gdf[['geometry']], admin1_gdf[admin1_cols + ['geometry']], how='left', predicate='within')
        joined_admin1 = joined_admin1[~joined_admin1.index.duplicated(keep='first')]
        results_df.update(joined_admin1.dropna(subset=['index_right'])[admin1_cols])

    # --- 2. Identify Points Missing Data After Initial Joins ---
    # Mask for points missing *all* country columns
    failed_initial_country_join_mask = results_df[country_cols].isnull().all(axis=1) if country_cols else pd.Series(False, index=results_df.index)
    failed_country_indices = results_df[failed_initial_country_join_mask].index

    # Mask for points missing *all* admin1 columns
    failed_initial_admin_join_mask = results_df[admin1_cols].isnull().all(axis=1) if admin1_cols else pd.Series(False, index=results_df.index)
    failed_admin_indices = results_df[failed_initial_admin_join_mask].index

    # Combine indices that failed *either* join - these need distance check
    indices_needing_distance_check = failed_country_indices.union(failed_admin_indices)

    # --- 3. Calculate Distance to Land (Only for points missing initial data) ---
    distance_series = pd.Series(dtype=float) # Initialize empty series
    if not indices_needing_distance_check.empty and land_geom is not None:
        print(f"Calculating land distance for {len(indices_needing_distance_check)} points potentially near boundaries...")
        points_to_check_gdf = points_gdf.loc[indices_needing_distance_check]
        distances = []
        # for idx, point_row in tqdm(points_to_check_gdf.iterrows(), total=len(points_to_check_gdf), desc="Checking Land Distance"):
        for idx, point_row in points_to_check_gdf.iterrows():
            try:
                point_geom = point_row.geometry
                if not land_geom.is_valid:
                     land_geom = land_geom.buffer(0)
                     if not land_geom.is_valid: raise ValueError("Land geometry invalid.")
                nearest_land_pt, _ = nearest_points(land_geom, point_geom)
                dist_km = geodesic((point_geom.y, point_geom.x), (nearest_land_pt.y, nearest_land_pt.x)).km
                distances.append(dist_km)
            except Exception as e: distances.append(float('inf'))
        # Create series with distance, indexed by the points checked
        distance_series = pd.Series(distances, index=indices_needing_distance_check)

    # --- 4. Nearest Country Join for Close Points Missing Country Data ---
    if not failed_country_indices.empty and country_cols:
        # Identify points that failed country join AND are close based on COUNTRY distance threshold
        country_dist_check = distance_series.reindex(failed_country_indices) <= config.MAX_OCEAN_LAND_DISTANCE_COUNTRY_KM
        country_nearest_indices = country_dist_check[country_dist_check].index

        if not country_nearest_indices.empty:
            print(f"Performing nearest country join for {len(country_nearest_indices)} points...")
            points_country_near_gdf = points_gdf.loc[country_nearest_indices, ['geometry']]
            try:
                # Removed max_distance for robustness
                nearest_country = gpd.sjoin_nearest(points_country_near_gdf, countries_gdf[country_cols + ['geometry']], how='left')
                nearest_country = nearest_country[~nearest_country.index.duplicated(keep='first')]
                # Prepare results for update (only country columns for these indices)
                nearest_country_results = nearest_country[country_cols]
                # Update main results - only affects rows in country_nearest_indices and only country_cols
                print(f"Updating results with nearest country data for {len(country_nearest_indices)} points.")
                results_df.update(nearest_country_results)
            except Exception as e:
                print(f"Warning: sjoin_nearest for countries failed: {e}")

    # --- 5. Nearest Admin1 Join for Close Points Missing Admin1 Data ---
    if not failed_admin_indices.empty and admin1_cols:
        # Identify points that failed admin1 join AND are close based on ADMIN1 distance threshold
        admin1_dist_check = distance_series.reindex(failed_admin_indices) <= config.MAX_OCEAN_LAND_DISTANCE_ADMIN1_KM
        admin1_nearest_indices = admin1_dist_check[admin1_dist_check].index

        if not admin1_nearest_indices.empty:
            print(f"Performing nearest admin1 join for {len(admin1_nearest_indices)} points...")
            points_admin1_near_gdf = points_gdf.loc[admin1_nearest_indices, ['geometry']]
            try:
                # Removed max_distance for robustness
                nearest_admin1 = gpd.sjoin_nearest(points_admin1_near_gdf, admin1_gdf[admin1_cols + ['geometry']], how='left')
                nearest_admin1 = nearest_admin1[~nearest_admin1.index.duplicated(keep='first')]
                # Prepare results for update (only admin1 columns for these indices)
                nearest_admin1_results = nearest_admin1[admin1_cols]
                # Update main results - only affects rows in admin1_nearest_indices and only admin1_cols
                print(f"Updating results with nearest admin1 data for {len(admin1_nearest_indices)} points.")
                results_df.update(nearest_admin1_results)
            except Exception as e:
                print(f"Warning: sjoin_nearest for admin1 failed: {e}")

    # --- 6. Final Cleanup ---
    # Note: Hierarchy correction step is removed as per request.
    # Potential inconsistencies might remain if nearest joins fail differently.
    results_df = results_df.fillna(np.nan).replace([np.nan, pd.NaT], [None, None])
    return results_df

def join_oceans(points_gdf, oceans_gdf, current_results_df):
    """
    Spatially joins points with ocean polygons, including all attribute columns,
    ONLY for points that do not have valid country/admin1 information.

    Args:
        points_gdf: GeoDataFrame of input points for the current chunk.
        oceans_gdf: GeoDataFrame of oceans.
        current_results_df: DataFrame containing results from previous steps.

    Returns:
        pandas.DataFrame: DataFrame with added ocean attribute columns,
                          indexed ONLY for the points joined with oceans.
                          Returns empty DataFrame if no points need ocean join or
                          oceans_gdf has no attribute columns.
    """
    # Identify points that are candidates for ocean join (all admin columns are null)
    admin_cols_in_results = [col for col in current_results_df.columns if col not in ['latitude', 'longitude', 'geometry']] # Rough check
    if not admin_cols_in_results:
        # If no admin columns were added in the first place, assume all points are candidates
        ocean_candidate_mask = pd.Series(True, index=current_results_df.index)
    else:
        ocean_candidate_mask = current_results_df[admin_cols_in_results].isnull().all(axis=1)

    ocean_candidate_indices = current_results_df[ocean_candidate_mask].index

    # Dynamically get all non-geometry columns from oceans_gdf
    ocean_cols = oceans_gdf.columns.difference(['geometry']).tolist()

    if ocean_candidate_indices.empty or not ocean_cols:
        if not ocean_cols:
             print("Warning: No attribute columns found in oceans_gdf. Skipping ocean join.")
        else:
             print("No points require ocean joining.")
        # Return empty DF, but include expected column names if they exist
        return pd.DataFrame(columns=ocean_cols, index=pd.Index([]))

    print(f"Performing spatial join for oceans on {len(ocean_candidate_indices)} points...")
    points_for_ocean_join = points_gdf.loc[ocean_candidate_indices, ['geometry']]

    # Perform the join
    joined_ocean = gpd.sjoin(points_for_ocean_join, oceans_gdf[ocean_cols + ['geometry']], how='left', predicate='within')

    joined_ocean = joined_ocean[~joined_ocean.index.duplicated(keep='first')]
    ocean_results = joined_ocean[ocean_cols] # Select dynamic columns

    ocean_results = ocean_results.fillna(np.nan).replace([np.nan], [None])

    return ocean_results


def find_nearest_cities(points_gdf, cities_gdf, city_kdtree):
    """
    Finds the nearest city within the configured distance for all points,
    including all city attribute columns.

    Args:
        points_gdf: GeoDataFrame of input points for the current chunk.
        cities_gdf: GeoDataFrame of filtered cities (with all desired columns).
        city_kdtree: SciPy KDTree for city coordinates.

    Returns:
        pandas.DataFrame: DataFrame with nearest city information (all attribute
                          columns) and distance, indexed the same as points_gdf.
                          Values will be None if no city found within distance.
    """
    num_points = len(points_gdf)
    # Dynamically get city attribute columns
    city_info_cols = cities_gdf.columns.difference(['geometry']).tolist()

    # Initialize results dictionary with dynamic keys + distance
    city_result_data = {col: [None] * num_points for col in city_info_cols}
    city_result_data['city_distance_km'] = [None] * num_points

    if city_kdtree is None or cities_gdf.empty or points_gdf.empty:
        print("Skipping city search: No KDTree, cities, or points provided.")
        return pd.DataFrame(city_result_data, index=points_gdf.index)

    print(f"Finding nearest cities for {len(points_gdf)} points...")
    max_dist_km = config.MAX_CITY_DISTANCE_KM
    max_dist_deg_approx = max_dist_km / 111.0 * 1.1

    point_coords_array = np.array(points_gdf.geometry.apply(lambda p: (p.y, p.x)).tolist())

    if point_coords_array.size == 0:
         return pd.DataFrame(city_result_data, index=points_gdf.index)

    try:
        distances_approx, indices = city_kdtree.query(
            point_coords_array, k=5, distance_upper_bound=max_dist_deg_approx
        )
    except Exception as e:
        print(f"KDTree query failed: {e}. Skipping city search for this chunk.")
        return pd.DataFrame(city_result_data, index=points_gdf.index)

    # Iterate through points and their candidate neighbors
    # for i, (dist_list, idx_list) in tqdm(enumerate(zip(distances_approx, indices)), total=num_points, desc="Calculating City Distances"):
    for i, (dist_list, idx_list) in enumerate(zip(distances_approx, indices)):
        point_geom = points_gdf.geometry.iloc[i]
        point_coords = (point_geom.y, point_geom.x)
        point_index = points_gdf.index[i]

        best_city_series = None
        min_dist_km = float('inf')

        valid_indices = [idx for idx in idx_list if idx < len(cities_gdf)]

        if not valid_indices:
            continue

        for city_idx in valid_indices:
            city_series = cities_gdf.iloc[city_idx]
            city_coords = (city_series.geometry.y, city_series.geometry.x)

            try:
                dist_km = geodesic(point_coords, city_coords).km
            except ValueError:
                continue

            if dist_km <= max_dist_km and dist_km < min_dist_km:
                min_dist_km = dist_km
                best_city_series = city_series # Store the whole series

        if best_city_series is not None:
            # Assign values from the best city's series to the result dict
            for col in city_info_cols:
                 city_result_data[col][i] = best_city_series[col]
            city_result_data['city_distance_km'][i] = round(min_dist_km, 3)

    city_results_df = pd.DataFrame(city_result_data, index=points_gdf.index)
    city_results_df = city_results_df.fillna(np.nan).replace([np.nan], [None])

    return city_results_df

def add_climate_zones(points_gdf):
    """
    Determines the Köppen climate zone for each point.

    Args:
        points_gdf: GeoDataFrame of input points with 'latitude' and 'longitude'.

    Returns:
        pandas.Series: Series containing climate zone descriptions,
                       indexed the same as points_gdf. Returns empty string
                       if lookup fails.
    """
    print("Determining climate zones...")
    climate_zones = []
        # for idx, row in tqdm(points_gdf.iterrows(), total=len(points_gdf), desc="Calculating Climate Zones"):
    for idx, row in points_gdf.iterrows():
        try:
            lat = row['latitude']
            lon = row['longitude']
            climate_code = lookupCZ(lat, lon)
            climate_zone = mappings.koppen_climates.get(climate_code, "") # Get description from map
            climate_zones.append(climate_zone)
        except Exception as e:
            # print(f"Warning: Climate lookup failed for index {idx} (Lat: {lat}, Lon: {lon}): {e}")
            climate_zones.append("") # Append empty string on error

    return pd.Series(climate_zones, index=points_gdf.index)

def add_land_cover(points_gdf, landcover_datasets):
    """
    Determines the dominant land cover class and its probability for each point
    using pre-computed rasters.

    Args:
        points_gdf: GeoDataFrame of input points with 'latitude' and 'longitude'.
        landcover_datasets (tuple): Tuple containing (ds_class, ds_prob)
                                    rasterio dataset objects.

    Returns:
        pandas.DataFrame: DataFrame with 'land_cover_class' and
                            'land_cover_probability' columns, indexed the same
                            as points_gdf.
    """
    if landcover_datasets is None or landcover_datasets == (None, None) or points_gdf.empty:
        print("Skipping land cover: No datasets or points provided.")
        return pd.DataFrame(columns=['land_cover_class', 'land_cover_probability'], index=points_gdf.index)

    ds_class, ds_prob = landcover_datasets # Unpack the tuple

    if ds_class is None or ds_prob is None:
            print("Skipping land cover: One or both precomputed datasets are missing.")
            return pd.DataFrame(columns=['land_cover_class', 'land_cover_probability'], index=points_gdf.index)

    print("Determining land cover classes from pre-computed rasters...")

    # Assume points_gdf CRS matches raster CRS (or add reprojection if needed)
    if ds_class.crs != points_gdf.crs:
            print(f"Warning: Point CRS ({points_gdf.crs}) differs from raster CRS ({ds_class.crs}). Land cover results may be inaccurate.")

    land_cover_classes = []
    land_cover_probabilities = []

    class_nodata = ds_class.nodata
    prob_nodata = ds_prob.nodata

    # Use iterrows for simplicity here, but for extreme performance,
    # extracting all coords and using ds.sample might be faster.
    for idx, row in points_gdf.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        class_name = None
        probability = None

        try:
            # Get pixel indices - use ds_class as reference
            row_idx, col_idx = ds_class.index(lon, lat)

            # Check bounds
            if not (0 <= row_idx < ds_class.height and 0 <= col_idx < ds_class.width):
                raise IndexError("Point coordinates outside raster bounds.")

            # Read the single pixel value from each precomputed raster
            window = Window(col_idx, row_idx, 1, 1)
            class_val = ds_class.read(1, window=window)[0, 0]
            prob_val = ds_prob.read(1, window=window)[0, 0]

            # Check against nodata values
            # if class_nodata is not None and class_val == class_nodata:
            #     # If class is nodata, treat both as nodata/None
            #     pass # Keep class_name and probability as None
            # else:
            class_name = mappings.LANDCOVER_CLASSES.get(int(class_val), "Unknown")
            # Check probability nodata separately
            if prob_nodata is not None and prob_val == prob_nodata:
                    class_name = "Unknown"
                    probability = None # Or np.nan
            else:
                    # Convert probability (might need scaling depending on precompute step)
                    probability = float(prob_val) # Adjust scaling if needed (e.g. / 10.0)

        except IndexError: # Point outside bounds
            pass # Keep class_name and probability as None
        except Exception as e:
            # print(f"Warning: Land cover lookup failed for index {idx} (Lat: {lat}, Lon: {lon}): {e}")
            pass # Keep class_name and probability as None

        land_cover_classes.append(class_name)
        land_cover_probabilities.append(probability)

    results_df = pd.DataFrame({
        'land_cover_class': land_cover_classes,
        'land_cover_probability': land_cover_probabilities
    }, index=points_gdf.index)

    return results_df