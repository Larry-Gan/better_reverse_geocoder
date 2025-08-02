# main.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import numpy as np
import gc # Garbage collector

# Import from our modules
import config
import data_loader
import processing
import rasterio # Ensure rasterio is imported

import time

# I think majority voting would be best
# From what I see tho, Country is better? Though theoretically admin is better
# For text, say border or majority voting
# For geocells, tend towards majority voting for country, but if doing admin1 will probably have ot go admin1 right?
    # Could check cities too...
# Add elevation (SRTM/ASTER) and Time Zone and maybe mountain range nearby, also landmarks if you can get them, distance to nearest coastlinne/major water body, prescence of roads/buildings
# try ot get dates if possible 
# alt train on precipitation, temperature, , pop density, political groupings, urban vs rural
def main():
    print("--- Starting Reverse Geocoding Process ---")

    landcover_datasets_tuple = None
    # --- Load Geospatial Data ---
    try:
        countries = data_loader.load_countries()
        admin1 = data_loader.load_admin1()
        oceans = data_loader.load_oceans()
        cities_filtered, city_kdtree = data_loader.load_cities_from_geojson_and_build_kdtree()
        land_geom = data_loader.create_unified_land(countries)

        if config.PROCESS_LANDCOVER:
            print("\n--- Loading Raster Data ---")
            # Returns a tuple (ds_class, ds_prob) or (None, None) or None
            landcover_datasets_tuple = data_loader.load_landcover_datasets()
            # If loading failed or disabled by loader, turn off processing
            if landcover_datasets_tuple is None or landcover_datasets_tuple == (None, None):
                print("Disabling land cover processing due to loading errors or config.")
                config.PROCESS_LANDCOVER = False
                landcover_datasets_tuple = None # Ensure it's None
    # ... (error handling remains the same) ...
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure required data files (Natural Earth, GeoNames) are downloaded and paths in config.py are correct.")
        return
    except ValueError as e:
         print(f"Error processing data: {e}")
         return
    except Exception as e:
         print(f"An unexpected error occurred during data loading: {e}")
         if landcover_datasets_tuple:
            print("Closing opened land cover datasets due to error...")
            ds_c, ds_p = landcover_datasets_tuple
            if ds_c:
                try: ds_c.close()
                except Exception as ce: print(f"Error closing class dataset: {ce}")
            if ds_p:
                try: ds_p.close()
                except Exception as ce: print(f"Error closing prob dataset: {ce}")
         return

    # --- Prepare Input and Output ---
    try:
        input_chunks = pd.read_csv(
            config.INPUT_FILE,
            chunksize=config.CHUNK_SIZE,
            usecols=config.USECOLS,
            encoding='utf-8'
        )
        with open(config.INPUT_FILE, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1
    # ... (error handling remains the same) ...
    except FileNotFoundError:
        print(f"Error: Input file not found at '{config.INPUT_FILE}'")
        return
    except UnicodeDecodeError as e:
        print(f"Error reading input file '{config.INPUT_FILE}': {e}")
        print("The file might not be UTF-8 encoded. Please check the file encoding.")
        return
    except Exception as e:
        print(f"Error reading input file '{config.INPUT_FILE}': {e}")
        if landcover_datasets_tuple: # Cleanup raster data
            ds_c, ds_p = landcover_datasets_tuple
            if ds_c: ds_c.close()
            if ds_p: ds_p.close()
        return


    print(f"Total points to process: {total_rows}")

    # Prepare output file
    first_chunk = True
    output_mode = 'w'

    # --- Define Final Output Columns Dynamically ---
    print("Defining final output columns...")
    final_columns = config.USECOLS[:]
    # Add columns from countries (excluding geometry)
    final_columns.extend(countries.columns.difference(['geometry']).tolist())
    # Add columns from admin1 (excluding geometry)
    final_columns.extend(admin1.columns.difference(['geometry']).tolist())
    # Add columns from oceans (excluding geometry)
    final_columns.extend(oceans.columns.difference(['geometry']).tolist())
    # Add columns from cities (excluding geometry)
    final_columns.extend(cities_filtered.columns.difference(['geometry']).tolist())
    # Add city distance column
    final_columns.append('city_distance_km')
    # add climate zones column
    final_columns.append('climate_zone')
    # Add land cover column
    if config.PROCESS_LANDCOVER:
        final_columns.extend(['land_cover_class', 'land_cover_probability'])

    # Remove duplicates while preserving order
    seen = set()
    final_columns = [x for x in final_columns if not (x in seen or seen.add(x))]
    print(f"Final output columns: {final_columns}")
    print("No dupes: ", len(final_columns) == len(seen))


    # --- Process Chunks ---
    for i, chunk in enumerate(tqdm(input_chunks, total=int(np.ceil(total_rows / config.CHUNK_SIZE)), desc="Processing Chunks")):
        print(f"\n--- Processing Chunk {i+1} ---")

        # --- Prepare Chunk GeoDataFrame ---
        try:
            chunk = chunk.dropna(subset=['latitude', 'longitude'])
            chunk = chunk[(chunk['latitude'].between(-90, 90)) & (chunk['longitude'].between(-180, 180))]
            if chunk.empty:
                print("Chunk is empty after validation, skipping.")
                continue

            geometry = [Point(xy) for xy in zip(chunk['longitude'], chunk['latitude'])]
            points_gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs='EPSG:4326')
            points_gdf['original_csv_index'] = chunk.index
            points_gdf = points_gdf.set_index('original_csv_index', drop=True)
            points_gdf.index.name = None

            # Initialize results DataFrame starting with input lat/lon
            results_df = points_gdf[config.USECOLS].copy()

            print(results_df.head())

        except Exception as e:
            print(f"Error creating GeoDataFrame for chunk {i+1}: {e}. Skipping chunk.")
            continue

        # --- Run Processing Steps ---
        # 1. Join Country and Admin1 info
        admin_results_df = processing.join_administrative_regions(
            points_gdf, countries, admin1, land_geom
        )
        results_df = results_df.merge(admin_results_df, left_index=True, right_index=True, how='left')

        # 2. Join Ocean info
        ocean_results_df = processing.join_oceans(
            points_gdf, oceans, results_df
        )
        # Dynamically get expected ocean columns for the merge
        ocean_cols_expected = oceans.columns.difference(['geometry']).tolist()
        if not ocean_results_df.empty or ocean_cols_expected: # Check if there are results or expected cols
             # Ensure the ocean results df has the expected columns even if empty
             for col in ocean_cols_expected:
                 if col not in ocean_results_df.columns:
                     ocean_results_df[col] = None
             results_df = results_df.merge(ocean_results_df[ocean_cols_expected], left_index=True, right_index=True, how='left')


        # 3. Find Nearest Cities
        city_results_df = processing.find_nearest_cities(
            points_gdf, cities_filtered, city_kdtree
        )
        results_df = results_df.merge(city_results_df, left_index=True, right_index=True, how='left')

        # 4. Add Climate Zones (Requires only lat/lon)
        climate_zone_series = processing.add_climate_zones(points_gdf)
        results_df['climate_zone'] = climate_zone_series

         # --- ADD: 5. Add Land Cover ---
        if config.PROCESS_LANDCOVER and landcover_datasets_tuple:
            landcover_results_df = processing.add_land_cover(
                points_gdf, landcover_datasets_tuple # Pass the tuple
            )
            results_df = results_df.merge(landcover_results_df, left_index=True, right_index=True, how='left')
        elif config.PROCESS_LANDCOVER: # If enabled but datasets are None
                results_df['land_cover_class'] = None
                results_df['land_cover_probability'] = None

        # --- Finalize and Save Chunk ---
        # Ensure all dynamically determined final columns exist
        for col in final_columns:
            if col not in results_df.columns:
                results_df[col] = None

        # Reorder and select final columns
        chunk_output_df = results_df[final_columns]

        # Replace any remaining NaNs with None
        chunk_output_df = chunk_output_df.fillna(np.nan).replace([np.nan], [None])

        try:
            chunk_output_df.to_csv(config.OUTPUT_FILE, mode=output_mode, header=first_chunk, index=False, encoding='utf-8')
            first_chunk = False
            output_mode = 'a'
            print(f"Chunk {i+1} processed and saved.")
        except Exception as e:
            print(f"Error writing chunk {i+1} to output file: {e}")
        finally: # --- UPDATE: Ensure datasets in tuple are closed ---
            if landcover_datasets_tuple:
                print("\nClosing land cover datasets...")
                ds_c, ds_p = landcover_datasets_tuple
                closed_c = closed_p = False
                if ds_c:
                    try:
                        ds_c.close()
                        closed_c = True
                    except Exception as e: print(f"Error closing land cover class dataset: {e}")
                if ds_p:
                    try:
                        ds_p.close()
                        closed_p = True
                    except Exception as e: print(f"Error closing land cover prob dataset: {e}")
                if closed_c and closed_p:
                        print("Land cover datasets closed.")


        # --- Memory Management ---
        del points_gdf, results_df, admin_results_df, ocean_results_df, city_results_df, chunk_output_df, geometry
        gc.collect()
        print(f"Memory cleaned after chunk {i+1}.")


    print(f"\n--- Processing Complete ---")
    print(f"Results saved to {config.OUTPUT_FILE}")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("\n\n")
    print(f"Total time taken: {end - start} seconds")