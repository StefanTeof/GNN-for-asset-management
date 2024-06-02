import torch
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_train_test_data():

    graph = nx.read_graphml('../Graphs/pearson_correlation_threshold_graph.graphml')
    df = pd.read_csv('D:\MANU\Datasets\yfinance_weekly_data.csv')
    df.set_index('Date', inplace=True)

    
    adj_close_values = df['Adj Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_adj_close = scaler.fit_transform(adj_close_values)
    df['Adj Close'] = scaled_adj_close
    # Add the data to the graph
    for company in graph.nodes:
        company_data = df[df['symbol'] == company]
        adj_close_dict = dict(zip(company_data.index, company_data['Adj Close']))
        nx.set_node_attributes(graph, {company: adj_close_dict}, name='Adj Close')

    nodes_to_remove = [node for node, attrs in graph.nodes(data=True) if len(attrs.get('Adj Close', [])) != 522]

    # Remove these nodes from the graph
    for node in nodes_to_remove:
        graph.remove_node(node)

    window_size = 26  # 6 months
    horizon = 26  

    windows = []
    targets = []
    symbols = []
    window_dates = []
    target_dates = []

    filtered_df = df[~df['symbol'].isin(list(nodes_to_remove))]


    # Group by company symbol
    grouped = filtered_df.groupby('symbol')



    for name, group in grouped:
        group = group.sort_index()
        for i in range(len(group) - window_size - horizon + 1):
            window = group.iloc[i:i+window_size]
            target = group.iloc[i+window_size+horizon-1]
            
            windows.append(window['Adj Close'].values)
            targets.append(target['Adj Close'])
            symbols.append(name)
            window_dates.append(window.index.min())
            target_dates.append(target.name)

    windows_df = pd.DataFrame({
        'Symbol': symbols,
        'WindowStartDate': window_dates,
        'TargetDate': target_dates,
        'Window': windows,
        'Target': targets
    })
   # Group by 'Symbol' and count the number of entries for each symbol
    symbol_counts = windows_df.groupby('Symbol').size()

    # Check if all symbols have the same number of rows
    equal_rows_for_all_symbols = symbol_counts.nunique() == 1

    if equal_rows_for_all_symbols:
        print(f"All symbols have the same number of rows: {symbol_counts.iloc[0]}")
    else:
        print("Not all symbols have the same number of rows. Counts per symbol:\n", symbol_counts[symbol_counts != 471])
        print(len(symbol_counts[symbol_counts != 471]))
    # Check the number of unique companies
    unique_companies = windows_df['Symbol'].nunique()

    # Group by 'TargetDate' and count unique symbols for each date
    date_counts = windows_df.groupby('TargetDate')['Symbol'].nunique()

    # Check if all dates have entries for all companies
    all_dates_full = date_counts.eq(unique_companies).all()

    if all_dates_full:
        print(f"Each of the {len(date_counts)} dates has data for all {unique_companies} companies.")
    else:
        missing_data_dates = date_counts[date_counts != unique_companies]
        print(f"Missing data on {len(missing_data_dates)} dates. Details:\n{missing_data_dates}")

    # Length of grouped DataFrame by 'TargetDate'
    grouped_length = len(windows_df.groupby('TargetDate'))
    print(f"Length of DataFrame grouped by 'TargetDate': {grouped_length}")


    windows_df.sort_values(by=['TargetDate', 'Symbol'], inplace=True)

    # windows_df
    # Group by TargetDate
    grouped_by_date = windows_df.groupby('TargetDate')

    inputs = []
    outputs = []
    dates = []

    for date, group in grouped_by_date:
        if len(group) == 442:  # Ensures we have data for all companies
            # Reshape windows into [442, 26]
            input_windows = np.stack(group['Window'].values)  # Stacks into [442, 26]
        
            # Targets
            output_targets = group['Target'].values.reshape(-1, 1)  # Reshape into [1, 442]
            
            inputs.append(input_windows)
            outputs.append(output_targets)
            dates.append(date)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    encoder = LabelEncoder()

    sectors = []
    for node in graph.nodes(data=True):
        sector = node[1]['Sector']
        if sector not in sectors:
            sectors.append(sector)


    sector_encoded = encoder.fit_transform(sectors)
    sector_dict = dict(zip(sectors, sector_encoded))

    for node in graph.nodes(data=True):
        node[1]['Sector'] = sector_dict[node[1]['Sector']]
    node_to_index = {label: index for index, label in enumerate(graph.nodes)}
    node_to_index
    node_index_list = []
    node_sector_list = []

    for node in graph.nodes(data=True):
        node_index_list.append(node_to_index[node[0]])
        node_sector_list.append(node[1]['Sector'])
    node_index = torch.tensor(node_index_list, dtype=torch.long)
    node_sectors = torch.tensor(node_sector_list, dtype=torch.float)
    edge_index_list = []
    edge_weight_list = []

    for edge in graph.edges(data=True):
        src = edge[0]
        dst = edge[1]    
        edge_index_list.append([node_to_index[src], node_to_index[dst]])
        edge_weight_list.append(edge[2]['weight'])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    data_list = []

    for i in range(inputs_tensor.shape[0]):  # Loop through each date (471 in total)
        x = inputs[i]  # Select input data for the current date, shape: [26, 442]
        x = torch.tensor(x, dtype=torch.float32)
        y = outputs[i]  
        y = torch.tensor(y, dtype=torch.float32)
        y = y.view(-1)
        
        # Create a Data object for the current date
        data = Data(x=x, node_sectors=node_sectors, edge_index=edge_index, edge_weight=edge_weight, y=y)
        data_list.append(data)
    
    # Split the data
    split_idx = int(len(data_list) * 0.8)  # 80% for training, 20% for testing
    train_data_list = data_list[:split_idx]
    test_data_list = data_list[split_idx:]
    
    
    return train_data_list, test_data_list