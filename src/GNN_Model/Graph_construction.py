import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString

def load_data(url: str) -> pd.DataFrame:
    """Load preprocessed dataset."""
    try:
        data = pd.read_csv(url)
        print("Preprocessed data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return pd.DataFrame()
    
def make_correlation_df(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculate SPEI correlations between every pair of locations to use for edge construction."""
    try:
        loc_arr = dataset[["Lat", "Lon"]].drop_duplicates().to_numpy()
        output_df = pd.DataFrame(columns=["lat1", "lon1", "lat2", "lon2", "spei_correlation"])
        for i in range(len(loc_arr)):
            for j in range(i, len(loc_arr)):
                loc_i = loc_arr[i]
                loc_j = loc_arr[j]
                spei_i = dataset[(dataset["Lat"] == loc_i[0]) & (dataset["Lon"] == loc_i[1])]["spei01"]
                spei_j = dataset[(dataset["Lat"] == loc_j[0]) & (dataset["Lon"] == loc_j[1])]["spei01"]
                corr = spei_i.corr(spei_j)
                output_df.loc[len(output_df)] = [loc_i[0], loc_i[1], loc_j[0], loc_j[1], corr]
        print("Correlation DataFrame created successfully.")
        return output_df
    except Exception as e:
        print(f"Error creating correlation DataFrame: {e}")
        return pd.DataFrame()
    
def construct_graph(dataset: pd.DataFrame, corr_df: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    """
    Spatial edges: connect locations with correlation above a certain threshold
    Temporal edges: connect each location to itself in the next time step (directed edge)
    """
    try:
        spatio_temporal_graph = nx.DiGraph() 
        #create node features
        print("Creating nodes and adding attributes...")
        for index, row in dataset.iterrows():
            node_attributes = row[["Month", "Lat", "Lon", "datetime", "month_norm", "lat_norm", "lon_norm", 
                                "temperature_2m", "temperature_2m_min", "temperature_2m_max", "dewpoint_temperature_2m", 
                                "total_precipitation_sum", "total_precipitation_min", "total_precipitation_max", 
                                "u_component_of_wind_10m", "v_component_of_wind_10m", "surface_net_solar_radiation_sum", 
                                "total_evaporation_sum", "soil_moisture_am", "soil_moisture_pm", "NDVI", "EVI", 
                                "spei01", "year", "month2", "next_month_spei01"]].to_dict()  
            spatio_temporal_graph.add_node(row["row_index"], **node_attributes)
        
        #create spatial edges based on correlation threshold
        print("creating spatial edges")
        for month in dataset["Month"].unique():
            month_data = dataset[dataset["Month"] == month]
            for _, corr_row in corr_df.iterrows():
                if corr_row["spei_correlation"] >= threshold:
                    node_i = month_data[
                        (month_data["Lat"] == corr_row["lat1"]) & 
                        (month_data["Lon"] == corr_row["lon1"])
                    ]["row_index"]
                    node_j = month_data[
                        (month_data["Lat"] == corr_row["lat2"]) & 
                        (month_data["Lon"] == corr_row["lon2"])
                    ]["row_index"]
                    
                    if not node_i.empty and not node_j.empty:
                        spatio_temporal_graph.add_edge(node_i.iloc[0], node_j.iloc[0])
                        spatio_temporal_graph.add_edge(node_j.iloc[0], node_i.iloc[0])
            print("Month {}: Spatial edges added.".format(month))

        #Create temporal edges
        print("Creating temporal edges...")
        for _, row in dataset.iterrows():
            next_timestep_month = row["Month"] + 1 if row["Month"] < 12 else 1
            next_timestep_year = row["year"] if row["Month"] < 12 else row["year"] + 1

            next_node = dataset[
                (dataset["Lat"] == row["Lat"]) &
                (dataset["Lon"] == row["Lon"]) &
                (dataset["year"] == next_timestep_year) &
                (dataset["Month"] == next_timestep_month)
            ]["row_index"]

            if not next_node.empty:
                spatio_temporal_graph.add_edge(row["row_index"], next_node.iloc[0])

        print("Graph constructed successfully.")
        return spatio_temporal_graph
    except Exception as e:
        print(f"Error constructing graph: {e}")
        return nx.Graph()
    
def generate_graph_visualization(graph: nx.DiGraph) -> plt.figure:
    pass