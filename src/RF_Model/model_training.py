import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

train_months = [201505, 201506, 201507, 201508, 201509, 201510, 201511, 201512,
201601, 201602, 201603, 201604, 201605, 201606, 201607, 201608,
201609, 201610, 201611, 201612, 201701, 201702, 201703, 201704,
201705, 201706, 201707, 201708, 201709, 201710, 201711, 201712,
201801, 201802, 201803, 201804, 201805, 201806, 201807, 201808,
201809, 201810, 201811, 201812, 201901, 201902, 201903, 201904,
201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912,
202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008,
202009, 202010, 202011, 202012, 202101, 202102, 202103, 202104, 
202105, 202106, 202107, 202108, 202109, 202110, 202111, 202112, 
202201, 202202, 202203, 202204, 202205]

test_months = [202206, 202207, 202208, 202209, 202210, 202211, 
               202212, 202301, 202302, 202303, 202304, 202305, 
               202306, 202307, 202308]

def load_data(url: str) -> pd.DataFrame:
    """Load preprocessed dataset for training."""
    try:
        data = pd.read_csv(url)
        print("Preprocessed data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return pd.DataFrame()

def get_train_test_data(data: pd.DataFrame) -> tuple:
    """Split data into training and testing sets based on month."""
    train_data = data[data['month'].isin(train_months)]
    test_data = data[data['month'].isin(test_months)]
    
    X_train = train_data.drop(columns=['next_month_spei01'])
    y_train = train_data['next_month_spei01']
    
    X_test = test_data.drop(columns=['next_month_spei01'])
    y_test = test_data['next_month_spei01']
    
    return X_train, X_test, y_train, y_test