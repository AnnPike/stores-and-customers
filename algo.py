import numpy as np
import osmnx as ox
from sklearn.neighbors import NearestNeighbors
import pickle
import os


def get_route_distance(orig_point, dest_point, G, epsg_place=9821):
    """
    Input:
    G - networkx graph of a city
    orig_lon - float: origin longitude
    orig_lat - float: origin latitude
    dest_lon - float: destination longitude
    dest_lat - float: destination latitude
    epsg_place - EPSG code of city = 9821 # crs for Kyiv
    output: distance of shortest path in meters """
    # fetch the nearest node w.r.t coordinates
    orig_node = get_network_node(orig_point, G)
    dest_node = get_network_node(dest_point, G)
    # find the shortest path
    route_nodes = ox.routing.shortest_path(G, orig_node, dest_node, weight="length")
    if route_nodes is None:
        route_nodes = ox.routing.shortest_path(G, dest_node, orig_node, weight="length")
    gdf = ox.graph_to_gdfs(G, edges=False).loc[route_nodes].to_crs(epsg=epsg_place)
    # calculate distance of the route
    distance_m = round(gdf.distance(gdf.shift(1)).sum(), 2)
    return distance_m


def get_network_node(orig_point, G):
    orig_lon, orig_lat = orig_point[0], orig_point[1]
    orig_node = ox.nearest_nodes(G, orig_lon, orig_lat)
    return orig_node


def iterate_new_store(new_store_store_long, new_store_lat, name, G_map, stores, population):
    """
    Input:
    new_store_store_long - float, longitude of a new store
    new_store_lat - float, latitude of a new store
    name - string, name of the file for the new store
    G_map -networkx graph of a city
    stores - pd.DataFrame of store locations
    population - pd.DataFrame of customers locations
    Output:
    dictionary with keys as customer id and values as store locations and route distance to them
    """

    stores_new = np.concatenate([stores, np.array([new_store_store_long, new_store_lat]).reshape((1, -1))])

    # define 3 closest stores for each customer
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(stores_new)
    lab = neigh.kneighbors(population)
    euclidian_dist_based_neighbours = lab[1]

    if name not in os.listdir('saved_data'):
        route_neighbours_dict = {}
        for i in range(len(population)):
            close_stores = euclidian_dist_based_neighbours[i]
            dict_stores_dict = {j: get_route_distance(population[i], stores_new[j], G_map) for j in close_stores}
            closest_store = min(dict_stores_dict, key=dict_stores_dict.get)
            route_neighbours_dict[i] = (closest_store, dict_stores_dict[closest_store])
        pickle.dump(route_neighbours_dict, open(f"saved_data/{name}", 'wb'))
    else:
        route_neighbours_dict = pickle.load(open(f"saved_data/{name}", 'rb'))
    return route_neighbours_dict
