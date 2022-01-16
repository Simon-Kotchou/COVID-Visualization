import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt
import numpy as np
import math
from sklearn import linear_model
import joblib


# useful for filtering data from a certain state or province
def filter_data_by_state(data, state):
    data = data[data['Province_State'] == state]
    return data[data['Admin2'] != 'Unassigned']


# makes nodes in graph for each row in data
def create_nodes_for_data(data):
    G = nx.Graph()
    for index, row in data.iterrows():
        G.add_node(str(row.iloc[5]))
    return G


# fills nodes with data and assign relative_time
def populate_data_into_nodes_state(G, data, logistic_curve=False, pop_data=None, load_curves=False,
                                   max_iterations=10000):
    already_infected = set()
    logistic_curves = dict()
    time = 0
    start_date = dt.datetime(2020, 1, 22)
    current_date = dt.datetime(2020, 1, 22)
    end_date = dt.datetime(2020, 5, 27)  # change as more data comes in
    while current_date <= end_date:
        day_data = data[data[current_date.strftime('%#m/%#d/%y')] > 0]
        was_added = False
        for index, row in day_data.iterrows():
            county = str(row.iloc[5])
            if county not in already_infected:
                G.nodes[county]['relative_time'] = time
                G.nodes[county]['real_time'] = current_date.strftime('%#m/%#d/%y')
                G.nodes[county]['lat'] = row.iloc[8]
                G.nodes[county]['long'] = row.iloc[9]
                G.nodes[county]['data'] = row.iloc[12:]
                if pop_data is not None:
                    pop = pop_data[pop_data['County'] == county]
                    if not pop.empty:
                        G.nodes[county]['pop'] = pop['Population']
                    else:
                        G.nodes[county]['pop'] = 2000  # give unknown counties 2000 population
                if logistic_curve:
                    if load_curves:
                        load_logistic_curve(county, logistic_curves)
                    else:
                        logistic_curves[county] = fit_logistic_curve(G.nodes[county], max_iterations)
                already_infected.add(county)
                was_added = True
        if was_added:
            time += 1
        current_date += dt.timedelta(days=1)
    return time, logistic_curves


# uses relative_time field to determine where to place edges, distance can be limited but by default is not
def create_edges_for_graph_first_infection(G, max_relative_time, distance_limit):
    time = 0
    prev_nodes = []
    temp = []
    while time <= max_relative_time:
        for node in G.nodes(data=True):
            if 'relative_time' in node[1] and node[1]['relative_time'] == time:
                for prev_node in prev_nodes:
                    distance = calculate_distance(prev_node, node)
                    if distance <= distance_limit:
                        G.add_edge(prev_node[0], node[0], weight=distance)
                temp.append(node)
        prev_nodes = temp.copy()
        temp.clear()
        time += 1


# uses real_time and distance to determine where to place edges threshold and distance_limit must be provided
def create_edges_for_graph_threshold_distance(G, threshold, distance_limit):
    start_date = dt.datetime(2020, 1, 22)
    current_date = dt.datetime(2020, 1, 22)
    end_date = dt.datetime(2020, 5, 27)
    prev_nodes = []
    while current_date <= end_date:
        for node in G.nodes(data=True):
            if 'real_time' in node[1] and node[1]['real_time'] == current_date.strftime('%#m/%#d/%y'):
                for prev_node in prev_nodes:
                    distance = calculate_distance(prev_node, node)
                    day = int(current_date.strftime('%#j')) - int(start_date.strftime('%#j'))
                    weight = prev_node[1]['data'][int(day)]/distance
                    if weight >= threshold and distance <= distance_limit:
                        G.add_edge(prev_node[0], node[0], weight=weight)
                prev_nodes.append(node)
        current_date += dt.timedelta(days=1)


# uses regression to fit a logistic growth curve to the each county and then run a simulation to create edges
def create_edges_for_graph_logistic_simulation(G, logistic_curves, time_limit, radius_weight):
    time = np.zeros((1, 1), dtype=int)
    infected = []
    found_first = False
    while time[0][0] <= time_limit:
        if not found_first:
            for node in G.nodes(data=True):
                if node[0] in logistic_curves and logistic_curves[node[0]].predict(time)[0] > 1:
                    infected.append(node)
                    found_first = True
                    break
        time[0][0] += 1
        if found_first:
            break
    while time[0][0] <= time_limit:
        for node1 in infected:
            radius = radius_weight * logistic_curves[node1[0]].predict(time)[0]  # / float(node1[1]['pop'])
            for node2 in G.nodes(data=True):
                if node2 in infected:
                    continue
                if 'lat' in node2[1]:
                    distance = calculate_distance(node1, node2)
                    if distance < radius:
                        G.add_edge(node1[0], node2[0], weight=distance)
                        infected.append(node2)

        time[0][0] += 1


# uses sklearn to fit a logistic growth curve to the data in the node
def fit_logistic_curve(node, max_iterations):
    x = np.linspace(0, len(node['data'])-1, len(node['data']), dtype=int)
    y = np.zeros(len(x), dtype=int)
    for i in range(0, len(y)):
        y[i] = int(node['data'][i])
    x = x[:, np.newaxis]
    clf = linear_model.LogisticRegression(C=1e5, max_iter=max_iterations)
    clf.fit(x, y)
    print('Curve Fitted')
    print('Score : ', clf.score(x, y))
    # # used to plot logistic curves
    # plt.figure(1, figsize=(4, 3))
    # # plt.clf()
    # x = np.linspace(0, 250, 251)
    # x = x[:, np.newaxis]
    # plt.plot(x, clf.predict(x))
    # plt.show()
    return clf


# saves logistic_curves to a directory named logistic_curves and a modifier can be added for different save locations
def save_logistic_curves(logistic_curves, modifier=''):
    if modifier is not '':
        modifier = '_' + modifier
    for key in logistic_curves:
        joblib.dump(logistic_curves[key], 'logistic_curves%s/%s.pkl' % (modifier, key))


# loads logistic curves from a directory named logistic_curves and a modifier can be added for different load locations
def load_logistic_curve(county, logistic_curves, modifier=''):
    if modifier is not '':
        modifier = '_' + modifier
    logistic_curves[county] = joblib.load('logistic_curves%s/%s.pkl' % (modifier, county))


# runs all methods in order to create a networkx graph of the spread of Covid-19 based on first infection in a location
# and connects location which were infected next
def create_graph_first_infected(data, distance_limit=30000):
    G = create_nodes_for_data(data)
    max_time = populate_data_into_nodes_state(G, data)[0]
    create_edges_for_graph_first_infection(G, max_time, distance_limit)
    return G


# runs all methods in order to create a networkx graph of the spread Covid-19 based on a threshold and distance_limit
# this method may be better on larger sample sizes so far locations are not infecting each other
def create_graph_infected_distance(data, threshold=2.5, distance_limit=75):
    G = create_nodes_for_data(data)
    populate_data_into_nodes_state(G, data)
    create_edges_for_graph_threshold_distance(G, threshold=threshold, distance_limit=distance_limit)
    return G


# runs all methods in order to create a networkx graph of the spread Covid-19 based on a Logistic simulation
def create_graph_logistic_simulation(data, time_limit=250, radius_weight=.5):
    G = create_nodes_for_data(data)
    pop_data = pd.read_csv('pop_ny_data_2020.csv')
    max_relative_time, logistic_curves = populate_data_into_nodes_state(G, data, logistic_curve=True, pop_data=pop_data,
                                                                        max_iterations=50000)
    create_edges_for_graph_logistic_simulation(G, logistic_curves, time_limit, radius_weight)
    save_logistic_curves(logistic_curves)
    return G


# runs all methods in order to create a networkx graph of the spread Covid-19 based on a logistic simulation loaded from
# saved logistic curves
def create_graph_logistic_simulation_load_lc(data, time_limit=250, radius_weight=.5):
    G = create_nodes_for_data(data)
    pop_data = pd.read_csv('pop_ny_data_2020.csv')
    max_relative_time, logistic_curves = populate_data_into_nodes_state(G, data, logistic_curve=True, pop_data=pop_data,
                                                                        load_curves=True)
    create_edges_for_graph_logistic_simulation(G, logistic_curves, time_limit, radius_weight)
    return G


# converts the networkx graph created to two csv files which can be imported into cytoscape
def convert_to_csv(G):
    output_node = 'county,relative_time,real_time,lat,long,degree,degree_centrality,closeness_centrality\n'
    output_edge = 'source,target,distance\n'
    for node in G.nodes(data=True):
        if 'relative_time' in node[1]:
            output_node += '%s,%s,%s,%s,%s,%s,%s,%s\n' % (node[0], node[1]['relative_time'], node[1]['real_time'],
                                                          node[1]['lat'], node[1]['long'], node[1]['degree'],
                                                          node[1]['degree_centrality'], node[1]['closeness_centrality'])
    for edge in G.edges(data=True):
        output_edge += '%s,%s,%s\n' % (edge[0], edge[1], edge[2]['weight'])

    return output_node, output_edge


# converts the cytoscape data and network csv files into csv files usable for the cytoscape coordinateLayout plugin
def convert_to_cyto_layout(node_file_in, edge_file_in, node_file_out, edge_file_out, scale=50):
    nodes_out_text = ''
    edges_out_text = ''
    nodes_in = pd.read_csv(node_file_in)
    cities = dict()
    i = 1
    for index, row in nodes_in.iterrows():
        cities[row.iloc[0]] = i
        nodes_out_text += '%s %s %s %s %s\n' % (i, str(row.iloc[0]).replace(' ', '_'), '1', float(row.iloc[3])/scale,
                                                float(row.iloc[4])/scale)
        i += 1
    edges_in = pd.read_csv(edge_file_in)
    for index, row in edges_in.iterrows():
        edges_out_text += '%s %s %s\n' % (cities[row.iloc[0]], cities[row.iloc[1]], row.iloc[2])

    with open(node_file_out, 'w') as f:
        f.write(nodes_out_text)
        f.close()

    with open(edge_file_out, 'w') as f:
        f.write(edges_out_text)
        f.close()


# calculates the distance between two nodes
def calculate_distance(node1, node2):
    lat1 = node1[1]['lat']
    lat2 = node2[1]['lat']
    lon1 = node1[1]['long']
    lon2 = node2[1]['long']
    R = 3958.5  # radius of earth in miles
    dlat = deg_to_rad(lat2 - lat1)
    dlon = deg_to_rad(lon2 - lon1)
    a = (math.sin(dlat/2))**2 + math.cos(deg_to_rad(lat1)) * math.cos(deg_to_rad(lat2)) * (math.sin(dlon/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return math.floor(R * c)


# helper function for calculate distance
def deg_to_rad(deg):
    return deg * (math.pi / 180)


def calculate_node_stats(G):
    for item in nx.degree_centrality(G).items():
        G.nodes[item[0]]['degree_centrality'] = item[1]
    for item in nx.closeness_centrality(G).items():
        G.nodes[item[0]]['closeness_centrality'] = item[1]
    for item in G.degree():
        G.nodes[item[0]]['degree'] = item[1]


def write_csv(filename, G):
    csv = convert_to_csv(G)
    with open('%s_network.csv' % filename, 'w') as f:
        f.write(csv[1])
        f.close()

    with open('%s_data.csv' % filename, 'w') as f:
        f.write(csv[0])
        f.close()


def main():
    data = pd.read_csv('time_series_covid19_confirmed_US.csv')

    new_york = filter_data_by_state(data, 'New York')

    # visualizes data in graph vs time
    dates = list(range(0, len(data.columns[12:].values)))
    plt.figure()
    for index, row in new_york.iterrows():
        plt.plot(dates, row.iloc[12:], label=row.iloc[5])
    plt.title(str.title('NY Counties cases vs. time'))
    plt.legend(framealpha=2, frameon=True, ncol=3, loc='upper left')
    plt.show()

    # creates graph of first infection
    G = create_graph_first_infected(new_york)
    calculate_node_stats(G)
    write_csv('first_infection', G)

    # creates graph of threshold distance
    H = create_graph_infected_distance(new_york)
    calculate_node_stats(H)
    write_csv('threshold_distance', H)

    # creates graph of logistic simulation
    # J = create_graph_logistic_simulation(new_york)
    J = create_graph_logistic_simulation_load_lc(new_york)
    calculate_node_stats(J)
    write_csv('logistic_simulation', J)

    # this will make files for the coordinateLayout plugin for cytoscape for each graph
    convert_to_cyto_layout('first_infection_data.csv', 'first_infection_network.csv',
                           'first_infection_data_cyto_layout.csv', 'first_infection_network_cyto_layout.csv')
    convert_to_cyto_layout('threshold_distance_data.csv', 'threshold_distance_network.csv',
                           'threshold_distance_data_cyto_layout.csv', 'threshold_distance_network_cyto_layout.csv')
    convert_to_cyto_layout('logistic_simulation_data.csv', 'logistic_simulation_network.csv',
                           'logistic_simulation_data_cyto_layout.csv', 'logistic_simulation_network_cyto_layout.csv')


if __name__ == '__main__':
    main()
