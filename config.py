# config.py
import os

# --- File Paths ---
# Directory where Natural Earth shapefiles are unzipped
# Assumes subdirectories like 'ne_50m_admin_0_countries', 'ne_50m_populated_places', etc.
DATA_DIR = 'natural_earth_data/'
LANDCOVER_DIR = 'landcover stuff/earthenv_landcover_data/' # <--- ADD: Directory for EarthEnv GeoTIFFs

CITY_GEOJSON_FILE = os.path.join("outputs", 'combined_cities.geojson')
USECOLS = ['name', 'filename', 'source', 'latitude', 'longitude',]

# Input CSV file with 'latitude', 'longitude' columns
INPUT_FILENMAE = "combined_sampled_geodata.csv"
# INPUT_FILE = os.path.join("input", 'misc_coords.csv')
INPUT_FILE = os.path.join("input", INPUT_FILENMAE)

# Output CSV file path
# OUTPUT_FILE = os.path.join("results", "geocoded_results_modular_multi.csv")
OUTPUT_FILE = os.path.join("results", "geocoded_" + INPUT_FILENMAE)

# --- Processing Parameters ---
# Process points in batches to manage memory
CHUNK_SIZE = 10000

# Minimum population for a place to be considered a 'city'
CITY_POPULATION_THRESHOLD = 500

# Maximum distance (km) to search for the nearest city
MAX_CITY_DISTANCE_KM = 25.0

# If a point is in the ocean, how close (km) must it be to land
# to still get administrative/city information?
MAX_OCEAN_LAND_DISTANCE_ADMIN1_KM = 25.0

MAX_OCEAN_LAND_DISTANCE_COUNTRY_KM = 100.0

# --- Feature Flags ---
# Set to True to find the nearest city even for points in the ocean
# (but within MAX_OCEAN_LAND_DISTANCE_KM of land).
# Set to False to only find cities for points strictly on land.
FIND_CITY_FOR_OCEAN_POINTS = True

# --- Data Column Names (Adjust if necessary based on Natural Earth version) ---
# Check your shapefiles for the exact column names
COUNTRY_NAME_COLS =  ['ISO2', 'REGION', 'SUBREGION']
ADMIN1_NAME_COLS = ['name', 'name_alt', 'name_local', 'iso_a2', 'region'] # Potential columns for Admin1 name
CITY_POPULATION_COL = 'pop_max' # Or 'POP_MAX', 'POP2000' etc.
CITY_NAME_COL = 'name' # Or 'NAME'
OCEAN_NAME_COL = 'ocean_name'
OCEAN_VALS_TO_DROP = {"Asia", "Africa", "Antarctica", "Oceania", "Australia", "Europe", "North America", "South America", "Australia"}

PROCESS_LANDCOVER = True # Keep enabled
# --- Land Cover Specific --- # <--- UPDATE section
# LANDCOVER_FILENAME_PATTERN = "consensus_full_class_{}.tif" # No longer needed for lookup
# LANDCOVER_NUM_CLASSES = 12 # No longer needed for lookup
LANDCOVER_MAX_CLASS_FILE = os.path.join(LANDCOVER_DIR, 'consensus_max_class.tif')
LANDCOVER_MAX_PROB_FILE = os.path.join(LANDCOVER_DIR, 'consensus_max_probability.tif')

# Optional: Check if precomputed files exist
if PROCESS_LANDCOVER and (not os.path.exists(LANDCOVER_MAX_CLASS_FILE) or not os.path.exists(LANDCOVER_MAX_PROB_FILE)):
        print(f"Warning: Precomputed land cover files not found in '{LANDCOVER_DIR}'.")
        print("Run precompute_landcover.py first.")
        PROCESS_LANDCOVER = False


# Ensure data directory exists (optional check)
if not os.path.isdir(DATA_DIR):
    print(f"Warning: Data directory '{DATA_DIR}' not found. Please ensure Natural Earth data is downloaded and unzipped there.")