# main.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# from tqdm import tqdm
import numpy as np
import gc # Garbage collector
import time
import os # Added for cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial # To pass arguments to worker function

# Import from our modules
import config
import data_loader
import processing

def process_chunk(chunk, countries, admin1, oceans, cities_filtered, city_kdtree, land_geom, final_columns):
    """
    Processes a single chunk of data. This function will be run by worker processes.

    Args:
        chunk (pd.DataFrame): The input data chunk.
        countries (gpd.GeoDataFrame): Loaded country data.
        admin1 (gpd.GeoDataFrame): Loaded admin1 data.
        oceans (gpd.GeoDataFrame): Loaded ocean data.
        cities_filtered (gpd.GeoDataFrame): Loaded and filtered city data.
        city_kdtree (scipy.spatial.KDTree): KDTree for cities.
        land_geom (shapely.geometry.BaseGeometry): Unified land geometry.
        final_columns (list): List of expected output columns.

    Returns:
        pd.DataFrame or None: The processed DataFrame for the chunk, or None if processing fails.
    """
    try:
        # --- Prepare Chunk GeoDataFrame ---
        chunk = chunk.dropna(subset=['latitude', 'longitude'])
        chunk = chunk[(chunk['latitude'].between(-90, 90)) & (chunk['longitude'].between(-180, 180))]
        if chunk.empty:
            print("Chunk is empty after validation, skipping.")
            return None # Return None for empty chunks

        geometry = [Point(xy) for xy in zip(chunk['longitude'], chunk['latitude'])]
        points_gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs='EPSG:4326')
        # Use original index from read_csv chunk, important for potential debugging
        # points_gdf['original_csv_index'] = chunk.index
        # points_gdf = points_gdf.set_index('original_csv_index', drop=True)
        # points_gdf.index.name = None # Keep original index from chunk

        # Initialize results DataFrame starting with input lat/lon and original index
        results_df = points_gdf[['latitude', 'longitude']].copy()

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
             # Make sure index types match before merge if necessary
             if results_df.index.dtype != ocean_results_df.index.dtype:
                 ocean_results_df.index = ocean_results_df.index.astype(results_df.index.dtype)
             results_df = results_df.merge(ocean_results_df[ocean_cols_expected], left_index=True, right_index=True, how='left')


        # 3. Find Nearest Cities
        city_results_df = processing.find_nearest_cities(
            points_gdf, cities_filtered, city_kdtree
        )
        # Make sure index types match before merge if necessary
        if results_df.index.dtype != city_results_df.index.dtype:
            city_results_df.index = city_results_df.index.astype(results_df.index.dtype)
        results_df = results_df.merge(city_results_df, left_index=True, right_index=True, how='left')

        # 4. Add Climate Zones (Requires only lat/lon)
        climate_zone_series = processing.add_climate_zones(points_gdf)
        # Ensure index alignment
        results_df['climate_zone'] = climate_zone_series.reindex(results_df.index)


        # --- Finalize Chunk ---
        # Ensure all dynamically determined final columns exist
        for col in final_columns:
            if col not in results_df.columns:
                results_df[col] = None

        # Reorder and select final columns
        chunk_output_df = results_df[final_columns]

        # Replace any remaining NaNs/NaTs with None (more JSON friendly if needed later)
        chunk_output_df = chunk_output_df.fillna(np.nan).replace([np.nan, pd.NaT], [None, None])

        # Clean up memory for this chunk within the worker
        del points_gdf, admin_results_df, ocean_results_df, city_results_df, climate_zone_series, results_df, geometry
        gc.collect()

        return chunk_output_df

    except Exception as e:
        print(f"Error processing chunk (Index starts at {chunk.index[0]}): {e}")
        # Optionally: return an empty dataframe with correct columns or None
        # Returning None might be simpler to handle in the main loop
        return None


def main():
    print("--- Starting Reverse Geocoding Process ---")

    # --- Load Geospatial Data (Once in the main process) ---
    try:
        print("Loading shared geospatial data...")
        countries = data_loader.load_countries()
        admin1 = data_loader.load_admin1()
        oceans = data_loader.load_oceans()
        cities_filtered, city_kdtree = data_loader.load_cities_and_build_kdtree_from_geonames()
        land_geom = data_loader.create_unified_land(countries)
        print("Shared geospatial data loaded.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure required data files (Natural Earth, GeoNames) are downloaded and paths in config.py are correct.")
        return
    except ValueError as e:
         print(f"Error processing data: {e}")
         return
    except Exception as e:
         print(f"An unexpected error occurred during data loading: {e}")
         return

    # --- Prepare Input and Output ---
    try:
        # Get total rows first for tqdm
        with open(config.INPUT_FILE, 'r', encoding='utf-8') as f:
             # Simple line count, adjust if header logic is complex
             total_rows = sum(1 for _ in f) -1 # Subtract header row
        if total_rows < 0: total_rows = 0

        input_chunks = pd.read_csv(
            config.INPUT_FILE,
            chunksize=config.CHUNK_SIZE,
            usecols=['latitude', 'longitude'],
            encoding='utf-8',
            # Keep index to potentially help track chunks, though not strictly needed for processing logic now
            # index_col=False # Ensure default RangeIndex is created per chunk
        )
        print(f"Reading input file: {config.INPUT_FILE}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{config.INPUT_FILE}'")
        return
    except UnicodeDecodeError as e:
        print(f"Error reading input file '{config.INPUT_FILE}': {e}")
        print("The file might not be UTF-8 encoded. Please check the file encoding.")
        return
    except Exception as e:
        print(f"Error reading input file '{config.INPUT_FILE}': {e}")
        return

    print(f"Total points to process: {total_rows}")
    num_chunks = int(np.ceil(total_rows / config.CHUNK_SIZE)) if total_rows > 0 else 0
    print(f"Processing in {num_chunks} chunks of size {config.CHUNK_SIZE}")

    # Prepare output file
    first_chunk_written = False
    output_mode = 'w'

    # --- Define Final Output Columns Dynamically ---
    print("Defining final output columns...")
    final_columns = ['latitude', 'longitude']
    # Add columns from countries (excluding geometry)
    if countries is not None and not countries.empty:
        final_columns.extend(countries.columns.difference(['geometry']).tolist())
    # Add columns from admin1 (excluding geometry)
    if admin1 is not None and not admin1.empty:
        final_columns.extend(admin1.columns.difference(['geometry']).tolist())
    # Add columns from oceans (excluding geometry)
    if oceans is not None and not oceans.empty:
        final_columns.extend(oceans.columns.difference(['geometry']).tolist())
    # Add columns from cities (excluding geometry)
    if cities_filtered is not None and not cities_filtered.empty:
        final_columns.extend(cities_filtered.columns.difference(['geometry']).tolist())
    # Add city distance column
    final_columns.append('city_distance_km')
    # add climate zones column
    final_columns.append('climate_zone')

    # Remove duplicates while preserving order
    seen = set()
    final_columns = [x for x in final_columns if not (x in seen or seen.add(x))]
    print(f"Final output columns: {final_columns}")


    # --- Prepare Worker Function with Partial Arguments ---
    # This creates a function that only needs the 'chunk' argument,
    # as the other data is "baked in".
    partial_process_chunk = partial(process_chunk,
                                    countries=countries,
                                    admin1=admin1,
                                    oceans=oceans,
                                    cities_filtered=cities_filtered,
                                    city_kdtree=city_kdtree,
                                    land_geom=land_geom,
                                    final_columns=final_columns)

    # --- Process Chunks in Parallel ---
    # Determine number of workers (adjust as needed based on your system RAM/CPU)
    # Using cpu_count() is a common starting point, but might need reduction
    # if memory usage per process is high.
    max_workers = os.cpu_count()
    print(f"Using up to {max_workers} worker processes.")

    processed_chunks_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to process chunks in parallel.
        # Results will be yielded in the order chunks were submitted.
        results_iterator = executor.map(partial_process_chunk, input_chunks)

        # Wrap iterator with tqdm for progress tracking
        # results_with_progress = tqdm(results_iterator, total=num_chunks, desc="Processing Chunks")

        for chunk_output_df in results_iterator:#results_with_progress:
            if chunk_output_df is not None and not chunk_output_df.empty:
                try:
                    # Write header only for the very first non-empty chunk processed
                    write_header = not first_chunk_written
                    chunk_output_df.to_csv(
                        config.OUTPUT_FILE,
                        mode=output_mode,
                        header=write_header,
                        index=False,
                        encoding='utf-8'
                    )
                    if write_header:
                        first_chunk_written = True
                        output_mode = 'a' # Switch to append mode after header is written
                    processed_chunks_count += 1
                    # print(f"Chunk {processed_chunks_count} written to CSV.") # Optional: more verbose logging
                except Exception as e:
                    print(f"Error writing chunk to output file: {e}")
            elif chunk_output_df is None:
                 print("A chunk failed processing and was skipped.")
            # else: # Chunk was empty after validation
            #     print("An empty chunk was skipped.") # Optional logging

            # Explicitly delete the processed chunk DataFrame to potentially free memory sooner
            del chunk_output_df
            gc.collect() # Encourage garbage collection

    print(f"\n--- Processing Complete ---")
    print(f"Processed and saved results for {processed_chunks_count} non-empty chunks.")
    if processed_chunks_count > 0:
        print(f"Results saved to {config.OUTPUT_FILE}")
    else:
        print("No data was written to the output file.")

# --- Crucial for Multiprocessing ---
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("\n")
    print(f"Total time taken: {end - start:.2f} seconds")