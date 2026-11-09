import io
import os
import numpy as np
import pandas as pd
import requests

# hub mappings config
EIA_API_KEY = os.getenv("EIA_API_KEY", "")

#US hubs
HUBS = {
    "DC_Virginia": {"airport": "IAD", "rto": "PJM", "state": "VA"},
    "DC_Texas": {"airport": "DFW", "rto": "ERCOT", "state": "TX"},
    "DC_Arizona": {"airport": "PHX", "rto": "AZPS", "state": "AZ"},
    "DC_Washington": {"airport": "SEA", "rto": "BPAT", "state": "WA"},
}

# EPA Grid Carbon Emission Factors (gCO2e per kWh generated)
EMISSION_FACTORS = {
    "COL": 950.0,  # Coal
    "NG": 450.0,  # Natural Gas
    "OIL": 800.0,  # Petroleum
    "NUC": 0.0,  # Nuclear
    "WAT": 0.0,  # Hydroelectric
    "WND": 0.0,  # Wind
    "SUN": 0.0,  # Solar
    "OTH": 500.0,  # Other / Unknown
}


# weather data(IEM ASOS API)
def calculate_wet_bulb(T_c: pd.Series, RH: pd.Series) -> pd.Series:
  """Calculates ambient wet-bulb temperature (°C) using Stull's empirical psychrometric equation."""
  #relative humidity bounds between 1% and 100%
  RH = np.clip(RH, 1.0, 100.0)

  T_wb = (
      T_c * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
      + np.arctan(T_c + RH)
      - np.arctan(RH - 1.676331)
      + 0.00391838 * (RH**1.5) * np.arctan(0.023101 * RH)
      - 4.686035
  )
  return T_wb


def fetch_iem_weather(
    station: str, start_date: str, end_date: str
) -> pd.DataFrame:
  """Fetches hourly dry-bulb temp (°F) and relative humidity (%) from Iowa Environmental Mesonet.

  No API key required.
  """
  start = pd.to_datetime(start_date)
  end = pd.to_datetime(end_date)

  url = (
      f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
      f"station={station}&data=tmpf&data=relh&"
      f"year1={start.year}&month1={start.month}&day1={start.day}&"
      f"year2={end.year}&month2={end.month}&day2={end.day}&"
      f"tz=Etc%2FUTC&format=onlycomma&missing=M"
  )

  response = requests.get(url, timeout=30)
  response.raise_for_status()

  # IEM may return a plain-text error page (with HTTP 200) for invalid
  # parameters. Catch that before passing it to pandas, which otherwise
  # reports a confusing CSV tokenization error.
  if not response.text.lstrip().startswith("station,"):
    raise ValueError(f"Unexpected IEM response for {station}: {response.text[:500]!r}")

  # Parse CSV payload into Pandas
  df = pd.read_csv(io.StringIO(response.text))

  # Clean missing values and parse timestamps
  df["valid"] = pd.to_datetime(df["valid"])
  df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
  df["relh"] = pd.to_numeric(df["relh"], errors="coerce")
  df = df.dropna(subset=["tmpf", "relh"])

  # Convert Dry-Bulb from Fahrenheit to Celsius
  df["temp_c"] = (df["tmpf"] - 32.0) * (5.0 / 9.0)

  # Compute Wet-Bulb Temperature (°C)
  df["temp_wb"] = calculate_wet_bulb(df["temp_c"], df["relh"])

  # Resample to strict 1-hour frequency
  df = df.set_index("valid").resample("1h").mean(numeric_only=True).reset_index()
  df = df.rename(columns={"valid": "timestamp"})

  return df[["timestamp", "temp_c", "temp_wb"]]

# carbon intensity data (EIA API v2)
def fetch_eia_carbon_intensity(
    rto: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
  """Queries EIA v2 API for hourly electricity generation by fuel type and calculates hourly grid carbon intensity (gCO2e/kWh)."""
  url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"

  params = {
      "api_key": api_key,
      "frequency": "hourly",
      "data[0]": "value",
      "facets[respondent][]": rto,
      "start": f"{start_date}T00",
      "end": f"{end_date}T23",
      "sort[0][column]": "period",
      "sort[0][direction]": "asc",
      "length": 5000,
  }

  response = requests.get(url, params=params)
  response.raise_for_status()
  payload = response.json()
  data = payload.get("response", {}).get("data", [])

  if not data:
    raise ValueError(f"No EIA grid data returned for RTO: {rto}")

  df = pd.DataFrame(data)
  # The EIA RTO fuel-type endpoint identifies the fuel code as `fueltype`.
  # Validate the schema here so API errors or future schema changes are clear.
  if "fueltype" not in df.columns:
    raise ValueError(
        "EIA fuel-type response is missing 'fueltype'; "
        f"received columns: {list(df.columns)}"
    )
  if "period" not in df.columns or "value" not in df.columns:
    raise ValueError(
        "EIA fuel-type response is missing required 'period' or 'value'; "
        f"received columns: {list(df.columns)}"
    )
  df["timestamp"] = pd.to_datetime(df["period"])
  df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

  # Map EIA fuel codes to carbon emission factors
  df["emission_factor"] = (
      df["fueltype"].map(EMISSION_FACTORS).fillna(EMISSION_FACTORS["OTH"])
  )
  df["emissions_g"] = df["value"] * 1000.0 * df["emission_factor"]  # MWh to kWh

  # Aggregate generation and emissions per hour across all fuel types
  hourly_grid = (
      df.groupby("timestamp")
      .agg(total_mwh=("value", "sum"), total_emissions_g=("emissions_g", "sum"))
      .reset_index()
  )

  # Calculate Weighted Average Carbon Intensity (gCO2e/kWh)
  hourly_grid["carbon_intensity"] = np.where(
      hourly_grid["total_mwh"] > 0,
      hourly_grid["total_emissions_g"] / (hourly_grid["total_mwh"] * 1000.0),
      400.0,  # Fallback grid average if data is missing
  )

  return hourly_grid[["timestamp", "carbon_intensity"]]


# dataset merging
if __name__ == "__main__":
  START_DATE = "2025-07-01"
  END_DATE = "2025-07-07"

  print(f"--- Fetching Time-Series Data from {START_DATE} to {END_DATE} ---")

  all_hub_data = []

  for dc_name, config in HUBS.items():
    print(f"\nProcessing {dc_name} (Airport: {config['airport']}, RTO: {config['rto']})...")

    # 1.Fetch Weather
    df_weather = fetch_iem_weather(config["airport"], START_DATE, END_DATE)

    # 2. Fetch Grid Carbon (using synthetic fallback if API key is not set)
    if not EIA_API_KEY:
      print(
          "  [NOTE] Using local synthetic grid carbon fallback (Add EIA"
          " API key for live grid data)."
      )
      df_grid = df_weather[["timestamp"]].copy()
      # Generate realistic base carbon values per region
      base_carbon = {
          "DC_Virginia": 380,
          "DC_Texas": 320,
          "DC_Arizona": 410,
          "DC_Washington": 40,
      }
      df_grid["carbon_intensity"] = base_carbon[
          dc_name
      ] + 50 * np.sin(df_grid.index * 0.26)
    else:
      df_grid = fetch_eia_carbon_intensity(
          config["rto"], START_DATE, END_DATE, EIA_API_KEY
      )

    # 3. Merge Weather and Grid Data on Timestamp
    df_merged = pd.merge(df_weather, df_grid, on="timestamp", how="inner")
    df_merged["datacenter"] = dc_name

    all_hub_data.append(df_merged)
    df_merged = df_merged[["timestamp", "temp_c", "temp_wb", "carbon_intensity", "datacenter"]]

  # Combine all data centers into one DataFrame
  df_final = pd.concat(all_hub_data, ignore_index=True)

  # Display sample results
  print("\n=== SAMPLE COMBINED DATASET ===")
  print(df_final.head(10).to_string(index=False))

  # Save to CSV for the PuLP solver
  df_final.to_csv("us_datacenters_processed_input.csv", index=False)
  print("\nData pipeline complete! Saved to 'us_datacenters_processed_input.csv'.")
