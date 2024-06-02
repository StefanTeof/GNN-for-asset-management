import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_data(df, graph, time_shift=26):
    
    # Normalize the data
    scaler = MinMaxScaler()
    adj_close_values = df['Adj Close'].values.reshape(-1, 1)
    scaled_adj_close = scaler.fit_transform(adj_close_values)
    df['Adj Close'] = scaled_adj_close

    # Shift future prices
    df['future_price'] = df.groupby('symbol')['Adj Close'].shift(-time_shift)

    # Remove rows with NaN values
    df.dropna(axis=0, inplace=True)

    # Add data to the graph
    for company in graph.nodes:
        company_data = df[df['symbol'] == company]
        adj_close_dict = dict(zip(company_data.index, company_data['Adj Close']))
        nx.set_node_attributes(graph, {company: adj_close_dict}, name='Adj Close')

        future_price_dict = dict(zip(company_data.index, company_data['future_price']))
        nx.set_node_attributes(graph, {company: future_price_dict}, name='Future Price')

    # Remove nodes and edges based on the specified condition
    nodes_to_remove = [node for node in graph.nodes(data=True) if len(node[1].get('Future Price', [])) < 496]
    edges_to_remove = [(edge[0], edge[1]) for edge in graph.edges() if edge[0] in [node[0] for node in nodes_to_remove] or edge[1] in [node[0] for node in nodes_to_remove]]
    graph.remove_nodes_from([node[0] for node in nodes_to_remove])
    graph.remove_edges_from(edges_to_remove)

    # Delete the 'Name' attribute from the nodes
    for node in graph.nodes(data=True):
        del node[1]['Name']

    # Label encode the 'Sector' attribute
    encoder = LabelEncoder()
    sectors = [node[1]['Sector'] for node in graph.nodes(data=True)]
    sector_encoded = encoder.fit_transform(sectors)
    sector_dict = dict(zip(sectors, sector_encoded))
    for node in graph.nodes(data=True):
        node[1]['Sector'] = sector_dict[node[1]['Sector']]

    return graph


def create_data_object(graph, node_to_index):

    node_index_list = [node_to_index[node[0]] for node in graph.nodes(data=True)]
    node_sector_list = [node[1]['Sector'] for node in graph.nodes(data=True)]

    node_index = torch.tensor(node_index_list, dtype=torch.long)
    node_sectors = torch.tensor(node_sector_list, dtype=torch.float)

    edge_index_list = [[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in graph.edges(data=True)]
    edge_weight_list = [edge[2]['weight'] for edge in graph.edges(data=True)]

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    current_prices = []
    future_prices = []

    for node in graph.nodes(data=True):
        adj_close_dict = node[1].get('Adj Close', {})
        prices = [adj_close_dict[date] for date in sorted(adj_close_dict.keys())]

        future_price_dict = node[1].get('Future Price', {})
        fprices = [future_price_dict[date] for date in sorted(future_price_dict.keys())]

        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        fprices_tensor = torch.tensor(fprices, dtype=torch.float32)

        current_prices.append(prices_tensor)
        future_prices.append(fprices_tensor)

    x = torch.stack(current_prices)
    y = torch.stack(future_prices)

    data = Data(
        x=x,
        node_sectors=node_sectors,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )

    return data
