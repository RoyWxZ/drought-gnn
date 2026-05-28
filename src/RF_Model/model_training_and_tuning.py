import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
from pathlib import Path

#85 months
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

#15 months
test_months = [202206, 202207, 202208, 202209, 202210, 202211, 
               202212, 202301, 202302, 202303, 202304, 202305, 
               202306, 202307, 202308]

PROJECT_ROOT = Path(__file__).parent.parent.parent

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

    #drop non-normalized features and target variable
    X_train = train_data.drop(columns=['next_month_spei01', 'Month', 'Lat', 'Lon', 'datetime'])
    y_train = train_data['next_month_spei01']
    
    X_test = test_data.drop(columns=['next_month_spei01', 'Month', 'Lat', 'Lon', 'datetime'])
    y_test = test_data['next_month_spei01']
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> RandomForestRegressor:
    """Train a Random Forest Regressor on the training data."""
    base_params = {"n_estimators": 500, "random_state": 42, "verbose": 2, "n_jobs": -1}
    if params:
        base_params.update(params)
    #to use RF for time series forecasting, no oob score + dataset contains time based and lag features
    rf_model = RandomForestRegressor(**base_params)
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")
    return rf_model

def get_temporal_cv_splits(data: pd.DataFrame, n_splits: int = 5) -> list:
    """
    Get arrays of row positions for each fold of cross validation (expanding window), used to
    tune hyperparameters without overfitting to a particular test set
    
    split folds over months so all spatial locations at a given time step
    stay in the same fold and prevents temporal data leakage.
    """
    sorted_months = sorted(data['Month'].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_month_idx, val_month_idx in tscv.split(sorted_months):
        train_months = [sorted_months[i] for i in train_month_idx]
        val_months   = [sorted_months[i] for i in val_month_idx]
        train_pos = np.where(data['Month'].isin(train_months))[0]
        val_pos   = np.where(data['Month'].isin(val_months))[0]
        splits.append((train_pos, val_pos))
    return splits

def tune_hyperparameters(data: pd.DataFrame) -> dict:
    feature_cols = [c for c in data.columns
                    if c not in ['next_month_spei01', 'Month', 'Lat', 'Lon', 'datetime']]
    X = data[feature_cols]
    y = data['next_month_spei01']

    param_grid = {
        'max_depth':         [None, 10, 20], #smaller max depth = less overfitting and faster training time
        'min_samples_leaf':  [10, 20, 50], #larger min samples leaf = makes more general rules and less overfitting
        'max_features':      ['sqrt', 1.0], #smaller max features = greater forest diversity for more correlated features
    }

    cv_splits = get_temporal_cv_splits(data)

    print("Starting hyperparameter tuning with GridSearchCV...")

    search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=cv_splits,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2,
    )
    search.fit(X, y)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    return search.best_params_

if __name__ == "__main__":
    data_url = PROJECT_ROOT / "data/preprocessed_data.csv"
    dataset = load_data(data_url)
    best_params = tune_hyperparameters(dataset)
    X_train, X_test, y_train, y_test = get_train_test_data(dataset)

    print("Training final model with best hyperparameters...")
    rf_model = train_random_forest(X_train, y_train, params=best_params)

    model_save_path = PROJECT_ROOT / "Data/Models/rf_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Trained Random Forest model saved to {model_save_path}.")