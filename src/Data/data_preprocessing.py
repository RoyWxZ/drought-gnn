import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pathlib import Path

# Resolves to drought-gnn/Data/Data_tabular_raw/ regardless of where the script is run from
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Data_tabular_raw"

def load_data(url: str) -> pd.DataFrame:
    """Load raw tabular data for preprocessing."""
    try:
        data = pd.read_csv(url, index_col=0)
        print("Raw tabular data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading raw tabular data: {e}")
        return pd.DataFrame()

def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns, handle missing values, group by location and date,
    and convert units"""
    try:
        #drop NaNs in all columns except for soil_moisture_am_anomaly and soil_moisture_pm_anomaly, which have too many NaNs to drop
        dataset.dropna(subset=["Month", "Lat", "Lon", "temperature_2m", "temperature_2m_min", "temperature_2m_max", "dewpoint_temperature_2m", 
                               "total_precipitation_sum", "total_precipitation_min", "total_precipitation_max", "u_component_of_wind_10m", 
                               "v_component_of_wind_10m",'spei01', "surface_net_solar_radiation_sum", "total_evaporation_sum",
                               'next_month_precipitation', 'next_month_tempreature', 'EVI', 'NDVI', "soil_moisture_am", "soil_moisture_pm"], inplace=True)
        dataset = dataset.drop(["soil_moisture_am_anomaly", "soil_moisture_pm_anomaly"], axis = 1)
        dataset = dataset.drop(["next_month_tempreature", "next_month_precipitation" ], axis = 1)
        dataset = dataset.drop(["spei03"], axis = 1)

        dataset['datetime'] = pd.to_datetime(dataset['Month'], format='%Y%m')
        #Add year and month values as predictors
        dataset['year'] = dataset['datetime'].dt.year
        dataset['month2'] = dataset['datetime'].dt.month

        dataset.sort_values(by=['Lat', 'Lon', 'Month'], inplace=True) #for each location, list by month in chronological order

        #Add next month SPEI as a target variable for prediction
        dataset['next_month_spei01'] = dataset.groupby(['Lat', 'Lon'])['spei01'].shift(-1)
        dataset.dropna(subset=["next_month_spei01"], inplace=True)

        #Switch lats and lons
        Lat = dataset["Lat"]
        dataset["Lat"] = dataset["Lon"]
        dataset["Lon"] = Lat

        #Convert from m to mm
        dataset['total_precipitation_sum'] = dataset['total_precipitation_sum'] * 1000
        dataset['total_precipitation_min'] = dataset['total_precipitation_min'] * 1000
        dataset['total_precipitation_max'] = dataset['total_precipitation_max'] * 1000
        dataset['total_evaporation_sum'] = dataset['total_evaporation_sum'] * 1000

        #Convert from m/s to km/h
        dataset['u_component_of_wind_10m'] = dataset['u_component_of_wind_10m'] * 3.6
        dataset['v_component_of_wind_10m'] = dataset['v_component_of_wind_10m'] * 3.6

        dataset = dataset.reset_index(drop=True) #make the new ordering of the data the default ordering

        print("Data preprocessing completed successfully.")
        return dataset
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return pd.DataFrame()

def normalize_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize the data first using robust scaling to handle outliers (since precipitation and
      wind values naturally have long tails and outliers) and then apply min-max scaling"""
    try:
        min_max_scaler = MinMaxScaler().set_output(transform="pandas")
        robust_scaler = RobustScaler().set_output(transform="pandas")
        dataset_without_datetime = dataset.drop(["datetime"], axis=1) #drop datetime column for normalization since it is not a numerical feature
        dataset_normalized = robust_scaler.fit_transform(dataset_without_datetime)
        dataset_normalized = min_max_scaler.fit_transform(dataset_normalized)

        dataset_normalized = dataset_normalized.rename(columns={'Month': 'month_norm', 'Lat': 'lat_norm', 'Lon': 'lon_norm'})

        #horziontally concatenate the normalized data with the original Month, Lat, and Lon columns
        dataset = pd.concat([dataset[['Month','Lat', 'Lon', "datetime"]], dataset_normalized], axis=1)

        #create a column of row indices that will be utilized downstream
        dataset["row index"] = dataset.index

        print("Data normalization completed successfully.")
        return dataset
    except Exception as e:
        print(f"Data normalization failed: {e}")
        return pd.DataFrame()
    
def save_preprocessed_data(dataset: pd.DataFrame, filename: str) -> None:
    """Save the preprocessed data to a CSV file."""
    try:
        dataset.to_csv(filename, index=False)
        print(f"Preprocessed data saved successfully to {filename}.")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

if __name__ == "__main__":
    url = RAW_DATA_DIR / "201505-202309.csv"
    raw_data = load_data(url)
    preprocessed_data = preprocess_data(raw_data)
    normalized_data = normalize_data(preprocessed_data)
    save_preprocessed_data(normalized_data, PROJECT_ROOT / "Data" / "Data_preprocessed" / "preprocessed.csv")