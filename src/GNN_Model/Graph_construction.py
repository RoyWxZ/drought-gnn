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
    
def construct_graph(dataset: pd.DataFrame, corr_df: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    """
    Spatial edges: connect locations with correlation above a certain threshold
    Temporal edges: connect each location to itself in the next time step (directed edge)
    """
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
    
def generate_graph_visualization(graph: nx.DiGraph, world_data: gpd.GeoDataFrame, month: int, dataset: pd.DataFrame) -> plt.figure:
    african_countries = [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
        "Congo (Congo-Brazzaville)", "Djibouti", "Egypt", "Equatorial Guinea",
        "Eritrea", "Eswatini (fmr. Swaziland)", "Ethiopia", "Gabon", "Gambia",
        "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho",
        "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania",
        "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
        "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
        "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
        "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    ]

    african_countries_data = world_data[world_data['name'].isin(african_countries)]
    fig, ax = plt.subplots(figsize=(10, 8))
    african_countries_data.plot(ax=ax, color='lightgray', edgecolor='black')
    # Adjust the plot to focus on africa
    ax.set_xlim([-20, 60])
    ax.set_ylim([-40, 40])

    df_specific_month = dataset[dataset["Month"] == month]

    #Convert dataset of a specific month to a GeoDataFrame for plotting
    geometry = [Point(lon, lat) for lon, lat in zip(df_specific_month['Lon'], df_specific_month['Lat'])]
    gdf_specific_month = gpd.GeoDataFrame(df_specific_month, geometry=geometry)

    #plot nodes
    gdf_specific_month.plot(ax=ax, marker='o', color='#000099', markersize=40)

    #plot edges
    specific_month_nodes = df_specific_month["row_index"].tolist()
    subgraph = graph.subgraph(specific_month_nodes)
    for source, target in subgraph.edges():
        point1 = gdf_specific_month.geometry[source]
        point2 = gdf_specific_month.geometry[target]
        ax.annotate("", 
                    xy=(point2.x, point2.y), 
                    xytext=(point1.x, point1.y),
                    arrowprops=dict(arrowstyle="->", color='#660000', lw=1.5))

    return fig