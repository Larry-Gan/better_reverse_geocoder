# data_loader.py
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import KDTree
import os
import config # Import config variables
import topojson
import mappings
import rasterio

# --- Helper Function ---
def _load_shapefile_helper(filepath, usecols=None, bbox=None):
    """Internal helper to load shapefile with pyogrio fallback and CRS check."""
    if not os.path.exists(filepath):
         raise FileNotFoundError(f"Shapefile not found: {filepath}")
    try:
        gdf = gpd.read_file(filepath, engine='pyogrio', use_arrow=True, columns=usecols, bbox=bbox)
        # print(f"Loaded {os.path.basename(filepath)} using pyogrio.")
    except ImportError:
        gdf = gpd.read_file(filepath, columns=usecols, bbox=bbox)
        # print(f"Loaded {os.path.basename(filepath)} using fiona.")
    except Exception as e:
        print(f"Pyogrio failed ({e}), falling back to fiona for {os.path.basename(filepath)}.")
        gdf = gpd.read_file(filepath, columns=usecols, bbox=bbox)

    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    return gdf

# --- Main Data Loading Functions ---

def load_countries():
    """Loads and prepares the country polygons."""
    print("Loading country data...")
    # filepath = os.path.join(config.DATA_DIR, 'ne_50m_admin_0_countries.shp')
    filepath = "../TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp"
    usecols = config.COUNTRY_NAME_COLS + ['geometry']
    countries = _load_shapefile_helper(filepath, usecols=usecols)
    countries['REGION_NAME'] = countries['REGION'].map(mappings.REGION_MAPPING).fillna('')
    countries['SUBREGION_NAME'] = countries['SUBREGION'].map(mappings.REGION_MAPPING).fillna('')
    countries.drop(columns=['REGION', 'SUBREGION'], inplace=True)
    # Rename for consistency
    countries.rename(
        columns={
            "ISO2": 'country_iso2',
            "REGION_NAME": 'country_region', 
            "SUBREGION_NAME": 'country_subregion'
            }, 
        inplace=True
    )
    print(f"Loaded {len(countries)} countries.")
    return countries

def load_admin1():
    """Loads and prepares the Admin 1 polygons."""
    print("Loading Admin 1 data...")
    filepath = os.path.join(config.DATA_DIR, 'ne_10m_admin_1_states_provinces.shp')
    # Load potential name columns + geometry
    usecols = config.ADMIN1_NAME_COLS + ['geometry'] # Filter out None/empty strings
    admin1 = _load_shapefile_helper(filepath, usecols=usecols)

    admin1.drop(columns=['name_alt'], inplace=True)

    # Rename for consistency
    admin1.rename(
        columns={
            "name": 'admin1_name',
            "name_local": 'admin1_local_name',
            "iso_a2": 'admin1_iso2',
            "region": 'admin1_region'
            }, 
        inplace=True
    )

    print(f"Loaded {len(admin1)} Admin 1 regions.")
    return admin1

def load_oceans():
    """Loads the ocean polygons."""
    print("Loading ocean data...")
    filepath = os.path.join(config.DATA_DIR, 'oceans_simplified.shp')
    usecols = [config.OCEAN_NAME_COL, 'geometry']
    oceans = _load_shapefile_helper(filepath, usecols=usecols)
    print(f"Loaded {len(oceans)} ocean/sea polygons.")
    return oceans

def create_unified_land(countries_gdf):
    """Creates a unified MultiPolygon geometry of all land areas."""
    print("Creating unified land geometry...")
    if countries_gdf is None or countries_gdf.empty:
        print("Warning: Cannot create unified land geometry without country data.")
        return None
    # Dissolve can be faster than unary_union for simple cases
    # land_polygons = countries_gdf.dissolve().geometry.iloc[0]
    # unary_union is generally more robust for complex overlaps/gaps
    land_polygons = countries_gdf.unary_union
    print("Unified land geometry created.")
    return land_polygons

def load_cities_from_geojson_and_build_kdtree():
    """
    Loads city data from a pre-processed GeoJSON file, prepares columns,
    filters by population, and builds a KDTree for efficient searching.

    Returns:
        tuple: A tuple containing (GeoDataFrame, KDTree).
               The GeoDataFrame contains the filtered city data, and the
               KDTree is built from its coordinates.
    """
    print("Loading city data from pre-processed GeoJSON...")
    filepath = config.CITY_GEOJSON_FILE

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"City GeoJSON file not found: {filepath}. "
            f"Please ensure your combined city GeoJSON is at this location."
        )

    # --- Load the GeoDataFrame ---
    try:
        # Use pyogrio engine for performance if available
        cities_gdf = gpd.read_file(filepath, engine='pyogrio', use_arrow=True)
        print(f"Loaded {len(cities_gdf)} cities from GeoJSON using pyogrio.")
    except Exception:
        cities_gdf = gpd.read_file(filepath)
        print(f"Loaded {len(cities_gdf)} cities from GeoJSON using fiona.")


    # --- Standardize Column Names ---
    # Create a mapping for columns that don't start with 'city_' and are not 'geometry'
    rename_map = {
        col: f'city_{col}'
        for col in cities_gdf.columns
        if not col.startswith('city_') and col != 'geometry'
    }
    if rename_map:
        print(f"Renaming columns for consistency: {rename_map}")
        cities_gdf.rename(columns=rename_map, inplace=True)
    
    # --- Filter by Population ---
    # Ensure the population column exists before filtering
    if config.CITY_POPULATION_COL in cities_gdf.columns:
        # Convert population to numeric, coercing errors to NaN, then fill NaN with 0
        cities_gdf[config.CITY_POPULATION_COL] = pd.to_numeric(
            cities_gdf[config.CITY_POPULATION_COL], errors='coerce'
        ).fillna(0)
        
        initial_count = len(cities_gdf)
        cities_filtered = cities_gdf[cities_gdf[config.CITY_POPULATION_COL] >= config.CITY_POPULATION_THRESHOLD].copy()
        print(f"Filtered cities: {len(cities_filtered)} of {initial_count} (Population >= {config.CITY_POPULATION_THRESHOLD})")
    else:
        print(f"Warning: Population column '{config.CITY_POPULATION_COL}' not found. Skipping population filter.")
        cities_filtered = cities_gdf.copy()


    # --- Build KDTree ---
    kdtree = None
    if not cities_filtered.empty:
        print("Building KDTree for cities...")
        # Use (latitude, longitude) from the geometry for KDTree query consistency
        city_coordinates = np.array(list(zip(cities_filtered.geometry.y, cities_filtered.geometry.x)))
        try:
             kdtree = KDTree(city_coordinates)
             print("KDTree built successfully.")
        except Exception as e:
             print(f"Error building KDTree: {e}. Proceeding without KDTree.")
             kdtree = None
             cities_filtered = cities_filtered.iloc[0:0] # Empty GDF if tree fails
    else:
        print("No cities found after filtering. No KDTree built.")

    # The rest of the application will use the new columns dynamically.
    # We just need to return the final GeoDataFrame and the tree.
    return cities_filtered, kdtree


def load_cities_and_build_kdtree_from_geonames():
    """Loads GeoNames cities500.txt, filters, creates GeoDataFrame, and builds KDTree."""
    print("Loading city data from GeoNames cities500.txt...")
    geonames_txt_filename = os.path.join(config.DATA_DIR, 'cities500.txt')

    if not os.path.exists(geonames_txt_filename):
        raise FileNotFoundError(
            f"GeoNames file not found: {geonames_txt_filename}. "
            f"Download from http://download.geonames.org/export/dump/cities500.zip "
            f"and place it in '{config.DATA_DIR}'"
        )

    # Define column names based on GeoNames readme.txt
    # http://download.geonames.org/export/dump/readme.txt
    colnames = [
        'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
        'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code',
        'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation',
        'dem', 'timezone', 'modification_date'
    ]

    print(f"Reading {geonames_txt_filename} ...")
    try:
        # Use pandas read_csv directly on the zip file
        cities_df = pd.read_csv(
            geonames_txt_filename,
            sep='\t',
            header=None,
            names=colnames,
            low_memory=False, # Suppresses DtypeWarning
            encoding='utf-8',
            keep_default_na=False,    # ← don’t use pandas’ default list of NA strings
            na_values=['']        
        )
    except Exception as e:
        raise IOError(f"Failed to read or unzip {geonames_txt_filename}: {e}")


    print(f"Loaded {len(cities_df)} entries from GeoNames.")

    # Select necessary columns
    cities_filtered = cities_df[['name', 'latitude', 'longitude', 'admin1_code',
        'admin2_code', 'country_code', 'population', 'alternatenames']].copy()
    
    # Mapping ids to names for admin areas
    cities_filtered[['admin1_code', 'admin2_code']] = cities_filtered[['admin1_code', 'admin2_code']].fillna('')

    # --- Load Admin Name Mappings ---
    print("\n--- Loading Admin Name Mappings ---")
    admin1_names_map = load_admin1_names()
    admin2_names_map = load_admin2_names()

    # --- Map Admin Names ---
    print("\n--- Mapping Admin Codes to Names ---")
    if not admin1_names_map:
        print("Warning: Admin1 name map is empty. Cannot map Admin1 names.")
        cities_filtered['admin1_name'] = ""
    else:
        # Create tuple keys for mapping
        admin1_keys = list(zip(cities_filtered['country_code'], cities_filtered['admin1_code']))
        # Map names, fill missing ones with 'N/A'
        cities_filtered['admin1_name'] = pd.Series(admin1_keys).map(admin1_names_map).fillna('')
        print("Admin1 names mapped.")

    if not admin2_names_map:
        print("Warning: Admin2 name map is empty. Cannot map Admin2 names.")
        cities_filtered['admin2_name'] = ""
    else:
        # Create tuple keys for mapping
        admin2_keys = list(zip(cities_filtered['country_code'], cities_filtered['admin1_code'], cities_filtered['admin2_code']))
        # Map names, fill missing ones with 'N/A'
        cities_filtered['admin2_name'] = pd.Series(admin2_keys).map(admin2_names_map).fillna('')
        print("Admin2 names mapped.")

    # --- Convert to GeoDataFrame ---
    print("Converting GeoNames data to GeoDataFrame...")
    try:
        geometry = [Point(xy) for xy in zip(cities_filtered['longitude'], cities_filtered['latitude'])]
        cities_gdf = gpd.GeoDataFrame(
            cities_filtered,
            geometry=geometry,
            crs='EPSG:4326' # GeoNames uses WGS84
        )
    except Exception as e:
        raise ValueError(f"Error creating GeoDataFrame from GeoNames data: {e}")

    # Rename for consistency with the rest of the code
    cities_gdf.rename(columns={'name': 'city_name'}, inplace=True)


    # --- Prepare City KDTree ---
    kdtree = None
    if not cities_gdf.empty:
        print("Building KDTree for GeoNames cities...")
        # Use (latitude, longitude) for KDTree query consistency
        city_coordinates = np.array(list(zip(cities_gdf.geometry.y, cities_gdf.geometry.x)))
        try:
             kdtree = KDTree(city_coordinates)
             print("KDTree built successfully.")
        except Exception as e:
             print(f"Error building KDTree: {e}. Proceeding without KDTree.")
             kdtree = None
             cities_gdf = cities_gdf.iloc[0:0] # Empty GDF if tree fails
    else:
        print("No cities found after loading/filtering GeoNames. No KDTree built.")

    

    cities_gdf.rename(
        columns={
            'name': 'city_name', 
            'latitude': 'city_latitude', 
            'longitude': 'city_longitude', 
            'admin1_code': 'city_admin1_code',
            'admin2_code': 'city_admin2_code',
            'admin1_name': 'city_admin1_name',
            'admin2_name': 'city_admin2_name',
            'country_code': 'city_country_code',
            'population': 'city_population'
            }, 
        inplace=True
    )

    cities_gdf = cities_gdf.drop(columns=['city_admin1_code', 'city_admin2_code'])

    # Keep only necessary columns for processing function
    return cities_gdf, kdtree

def load_admin1_names():
    """Loads admin1 codes and names from GeoNames admin1CodesASCII.txt."""
    filename = os.path.join(config.DATA_DIR, 'admin1CodesASCII.txt')

    print(f"Loading Admin1 names from {filename}...")
    try:
        admin1_df = pd.read_csv(
            filename,
            sep='\t',
            header=None,
            names=['code', 'name', 'asciiname', 'geonameid'],
            encoding='utf-8',
            dtype={'geonameid': str} # Keep geonameid as string if needed later
        )
        # Create unique keys: (country_code, admin1_code)
        admin1_df[['country_code', 'admin1_code']] = admin1_df['code'].str.split('.', expand=True, n=1)
        # Create a mapping dictionary: key = (country_code, admin1_code), value = name
        admin1_map = dict(zip(zip(admin1_df['country_code'], admin1_df['admin1_code']), admin1_df['name']))
        print(f"Loaded {len(admin1_map)} Admin1 name mappings.")
        return admin1_map
    except Exception as e:
        print(f"Error loading or processing {filename}: {e}")
        return {} # Return empty dict on error

def load_admin2_names():
    """Loads admin2 codes and names from GeoNames admin2Codes.txt."""
    filename = os.path.join(config.DATA_DIR, 'admin2Codes.txt')


    print(f"Loading Admin2 names from {filename}...")
    try:
        admin2_df = pd.read_csv(
            filename,
            sep='\t',
            header=None,
            names=['code', 'name', 'asciiname', 'geonameid'],
            encoding='utf-8',
            dtype={'geonameid': str}
        )
        # Create unique keys: (country_code, admin1_code, admin2_code)
        # Handle cases where admin2_code might be missing or format is unexpected
        split_codes = admin2_df['code'].str.split('.', expand=True, n=2)
        admin2_df['country_code'] = split_codes[0]
        admin2_df['admin1_code'] = split_codes[1]
        admin2_df['admin2_code'] = split_codes[2]

        # Filter out rows where splitting didn't yield all 3 parts (if any)
        admin2_df = admin2_df.dropna(subset=['country_code', 'admin1_code', 'admin2_code'])

        # Create a mapping dictionary: key = (country, admin1, admin2), value = name
        admin2_map = dict(zip(zip(admin2_df['country_code'], admin2_df['admin1_code'], admin2_df['admin2_code']), admin2_df['name']))
        print(f"Loaded {len(admin2_map)} Admin2 name mappings.")
        return admin2_map
    except Exception as e:
        print(f"Error loading or processing {filename}: {e}")
        return {} # Return empty dict on error
    
def load_landcover_datasets():
    """
    Opens the pre-computed dominant land cover class and probability rasters.

    Returns:
        tuple: A tuple containing (dataset_class, dataset_probability) rasterio
                dataset objects, or (None, None) if loading fails or is disabled.
                Returns None if config.PROCESS_LANDCOVER is False.
    """
    if not config.PROCESS_LANDCOVER:
        print("Land cover processing is disabled in config.")
        return None # Return None explicitly if disabled

    print("Opening pre-computed Land Cover GeoTIFF datasets...")
    ds_class = None
    ds_prob = None
    try:
        # Check existence again (belt-and-suspenders)
        if not os.path.exists(config.LANDCOVER_MAX_CLASS_FILE):
                raise FileNotFoundError(f"Max class file not found: {config.LANDCOVER_MAX_CLASS_FILE}")
        if not os.path.exists(config.LANDCOVER_MAX_PROB_FILE):
                raise FileNotFoundError(f"Max probability file not found: {config.LANDCOVER_MAX_PROB_FILE}")

        ds_class = rasterio.open(config.LANDCOVER_MAX_CLASS_FILE)
        ds_prob = rasterio.open(config.LANDCOVER_MAX_PROB_FILE)

        # Basic validation (optional)
        if ds_class.crs != ds_prob.crs or ds_class.transform != ds_prob.transform or ds_class.shape != ds_prob.shape:
            print("Warning: Precomputed class and probability rasters have different CRS, transform, or shape.")
            # Decide how to handle: raise error, proceed with warning, etc.
            # For now, just warn.

        print(f"Opened Max Class: {os.path.basename(config.LANDCOVER_MAX_CLASS_FILE)} (CRS: {ds_class.crs}, Shape: {ds_class.shape})")
        print(f"Opened Max Prob: {os.path.basename(config.LANDCOVER_MAX_PROB_FILE)} (CRS: {ds_prob.crs}, Shape: {ds_prob.shape})")
        return ds_class, ds_prob # Return as a tuple

    except FileNotFoundError as e:
        print(f"Error: {e}")
        if ds_class: ds_class.close()
        if ds_prob: ds_prob.close()
        return None, None # Return tuple of Nones on error
    except rasterio.RasterioIOError as e:
        print(f"Error opening precomputed raster file with rasterio: {e}")
        if ds_class: ds_class.close()
        if ds_prob: ds_prob.close()
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading precomputed land cover data: {e}")
        if ds_class: ds_class.close()
        if ds_prob: ds_prob.close()
        return None, None


def load_cities_and_build_kdtree():
    """Loads populated places, filters by population, and builds KDTree."""
    print("Loading city data...")
    filepath = os.path.join(config.DATA_DIR, 'ne_50m_populated_places.shp')
    cities = _load_shapefile_helper(filepath)

    # Check for population column
    if config.CITY_POPULATION_COL not in cities.columns:
        raise ValueError(f"Population column '{config.CITY_POPULATION_COL}' not found in cities shapefile.")
    # Check for name column
    if config.CITY_NAME_COL not in cities.columns:
        raise ValueError(f"City name column '{config.CITY_NAME_COL}' not found in cities shapefile.")

    # Filter by population
    cities_filtered = cities[cities[config.CITY_POPULATION_COL] > config.CITY_POPULATION_THRESHOLD].copy()
    print(f"Filtered cities: {len(cities_filtered)} (Population > {config.CITY_POPULATION_THRESHOLD})")

    # Build KDTree
    kdtree = None
    if not cities_filtered.empty:
        print("Building KDTree for cities...")
        # Use (latitude, longitude) for KDTree query consistency
        city_coordinates = np.array(list(zip(cities_filtered.geometry.y, cities_filtered.geometry.x)))
        try:
             kdtree = KDTree(city_coordinates)
             print("KDTree built successfully.")
        except Exception as e:
             print(f"Error building KDTree: {e}. Proceeding without KDTree.")
             kdtree = None # Ensure kdtree is None if building fails
             cities_filtered = cities_filtered.iloc[0:0] # Empty GDF if tree fails
    else:
        print("No cities found after filtering. No KDTree built.")

    # Keep only necessary columns
    cities_filtered = cities_filtered[[config.CITY_NAME_COL, 'geometry']].rename(columns={config.CITY_NAME_COL: 'city_name'})

    return cities_filtered, kdtree