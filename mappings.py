# === Mapping Dictionaries ===

# REGION_MAPPING: Maps M49 region codes to English names, for country level
REGION_MAPPING = {
    2: "Africa",
    5: "South America",
    9: "Oceania",
    11: "Western Africa",
    13: "Central America",
    14: "Eastern Africa",
    15: "Northern Africa",
    17: "Middle Africa",
    18: "Southern Africa",
    19: "Americas",
    21: "Northern America",
    29: "Caribbean",
    30: "Eastern Asia",
    34: "Southern Asia",
    35: "South-eastern Asia",
    39: "Southern Europe",
    53: "Australia and New Zealand",
    54: "Melanesia",
    57: "Micronesia",
    61: "Polynesia",
    142: "Asia",
    143: "Central Asia",
    145: "Western Asia",
    150: "Europe",
    151: "Eastern Europe",
    154: "Northern Europe",
    155: "Western Europe",
    202: "Sub-Saharan Africa",
    419: "Latin America and the Caribbean",
}

# koppen_climates: Maps Köppen climate classification codes to descriptions
koppen_climates = {
    # Group A: Tropical climates
    "Af": "Tropical rainforest climate",
    "Am": "Tropical monsoon climate",
    "Aw": "Tropical savanna climate (dry winter)",
    "As": "Tropical savanna climate (dry summer)",
    
    # Group B: Desert and semi-arid climates
    "BWh": "Hot desert climate",
    "BWk": "Cold desert climate",
    "BSh": "Hot semi-arid climate",
    "BSk": "Cold semi-arid climate",
    
    # Group C: Temperate climates
    # decided to shorten some
    "Cfa": "Humid subtropical climate",
    "Cfb": "Temperate oceanic climate",
    # "Cfb": "Temperate oceanic or subtropical highland climate",
    "Cfc": "Subpolar oceanic climate",
    "Cwa": "Monsoon-influenced humid subtropical climate",
    "Cwb": "Monsoon-influenced temperate oceanic climate",
    "Cwc": "Monsoon-influenced subpolar oceanic climate",
    # "Cwb": "Subtropical highland or Monsoon-influenced temperate oceanic climate",
    # "Cwc": "Cold subtropical highland or Monsoon-influenced subpolar oceanic climate",
    "Csa": "Hot-summer Mediterranean climate",
    "Csb": "Warm-summer Mediterranean climate",
    "Csc": "Cold-summer Mediterranean climate",
    
    # Group D: Continental climates
    "Dfa": "Hot-summer humid continental climate",
    "Dfb": "Warm-summer humid continental climate",
    "Dfc": "Subarctic climate",
    "Dfd": "Extremely cold subarctic climate",
    "Dwa": "Monsoon-influenced hot-summer humid continental climate",
    "Dwb": "Monsoon-influenced warm-summer humid continental climate",
    "Dwc": "Monsoon-influenced subarctic climate",
    "Dwd": "Monsoon-influenced extremely cold subarctic climate",
    "Dsa": "Mediterranean-influenced hot-summer humid continental climate",
    "Dsb": "Mediterranean-influenced warm-summer humid continental climate",
    "Dsc": "Mediterranean-influenced subarctic climate",
    "Dsd": "Mediterranean-influenced extremely cold subarctic climate",
    
    # Group E: Polar and alpine climates
    "ET": "Tundra climate",
    "EF": "Ice cap climate"
}

# --- ADD Land Cover Mapping ---
LANDCOVER_CLASSES = {
    1: "Evergreen/Deciduous Needleleaf Trees",
    2: "Evergreen Broadleaf Trees",
    3: "Deciduous Broadleaf Trees",
    4: "Mixed Trees",
    5: "Shrubs",
    6: "Herbaceous",
    7: "Cultivated and Managed Vegetation",
    8: "Regularly Flooded Vegetation",
    9: "Urban/Built-up",
    10: "Snow/Ice",
    11: "Barren",
    12: "Open Water",
}