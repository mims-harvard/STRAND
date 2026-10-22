import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data
from zipfile import ZipFile
import tarfile
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation
from multiprocessing import Pool

def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]

def combine_res(res_1, res_2):
    res_out = {}
    for key in res_1:
        res_out[key] = np.concatenate([res_1[key], res_2[key]])
    return res_out

def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        
def zip_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for zip file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset
    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.zip')
        print_sys('Extracting zip file...')
        with ZipFile((save_path + '.zip'), 'r') as zip:
            zip.extractall(path = data_path)
        print_sys("Done!")  
        
def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset

    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.tar.gz')
        print_sys('Extracting tar file...')
        with tarfile.open(save_path  + '.tar.gz') as tar:
            tar.extractall(path= data_path)
        print_sys("Done!")  
        
def get_go_auto(gene_list, data_path, data_name):
    """
    Get gene ontology data

    Args:
        gene_list (list): list of gene names
        data_path (str): the path to save the extracted dataset
        data_name (str): the name of the dataset

    Returns:
        df_edge_list (pd.DataFrame): gene ontology edge list
    """
    go_path = os.path.join(data_path, data_name, 'go.csv')
    
    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        ## download gene2go.pkl
        if not os.path.exists(os.path.join(data_path, 'gene2go.pkl')):
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
            dataverse_download(server_path, os.path.join(data_path, 'gene2go.pkl'))
        with open(os.path.join(data_path, 'gene2go.pkl'), 'rb') as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1],
                   gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

        edge_list_filter = [i for i in edge_list if i[2] > 0]
        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1',
                                                                      1: 'gene2',
                                                                      2: 'score'})

        df_edge_list = df_edge_list.rename(columns = {'gene1': 'source',
                                                      'gene2': 'target',
                                                      'score': 'importance'})
        df_edge_list.to_csv(go_path, index = False)        
        return df_edge_list

def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list
        
def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})
    
    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

def get_similarity_network(network_type, adata, threshold, k,
                           data_path, data_name, split, seed, train_gene_set_size,
                           set2conditions, default_pert_graph=True, pert_list=None):
    
    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(adata, threshold, k,
                                                     data_path, data_name, split,
                                                     seed, train_gene_set_size,
                                                     set2conditions)
    elif network_type == 'go':
        if default_pert_graph:
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
            tar_data_download_wrapper(server_path, 
                                     os.path.join(data_path, 'go_essential_all'),
                                     data_path)
            df_jaccard = pd.read_csv(os.path.join(data_path, 
                                     'go_essential_all/go_essential_all.csv'))

        else:
            df_jaccard = make_GO(data_path, pert_list, data_name)

        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                                    ['importance'])).reset_index(drop = True)

    return df_out

def get_coexpression_network_from_train(adata, threshold, k, data_path,
                                        data_name, split, seed, train_gene_set_size,
                                        set2conditions, control_label='ctrl'):  # ADD control_label param
    """
    Infer co-expression network from training data
    """
    
    fname = os.path.join(os.path.join(data_path, data_name), split + '_'  +
                         str(seed) + '_' + str(train_gene_set_size) + '_' +
                         str(threshold) + '_' + str(k) +
                         '_co_expression_network.csv')
    
    # Check if valid cached file exists
    if os.path.exists(fname):
        if os.path.getsize(fname) > 0:
            df = pd.read_csv(fname)
            if len(df) > 0:
                return df
        # If file is empty/corrupt, remove it
        print_sys(f"Removing corrupt/empty file: {fname}")
        os.remove(fname)
    
    # COMPUTE NETWORK (runs if file doesn't exist OR was just deleted)
    gene_list = [f for f in adata.var.gene_name.values]
    idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
    X = adata.X
    train_perts = set2conditions['train']
    
    # FIX: Use control_label parameter instead of hardcoded 'ctrl'
    ctrl_conditions = [i for i in train_perts if control_label.lower() in i.lower()]
    if not ctrl_conditions:
        print_sys(f"Warning: No conditions found containing '{control_label}'. Using all training data.")
        ctrl_conditions = train_perts
    
    X_tr = X[np.isin(adata.obs.condition, ctrl_conditions)]
    gene_list = adata.var['gene_name'].values

    X_tr = X_tr.toarray()
    out = np_pearson_cor(X_tr, X_tr)
    out[np.isnan(out)] = 0
    out = np.abs(out)

    out_sort_idx = np.argsort(out)[:, -(k + 1):]
    out_sort_val = np.sort(out)[:, -(k + 1):]

    df_g = []
    for i in range(out_sort_idx.shape[0]):
        target = idx2gene[i]
        for j in range(out_sort_idx.shape[1]):
            df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

    df_g = [i for i in df_g if i[2] > threshold]
    df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source',
                                                            1: 'target',
                                                            2: 'importance'})
    df_co_expression.to_csv(fname, index = False)
    return df_co_expression
    
def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): whether condition is 'ctrl' or not
        pert_names (list): list of perturbations
    """

    if condition == 'ctrl':
        return True
    else:
        cond1 = condition.split('+')[0]
        cond2 = condition.split('+')[1]
        num_ctrl = (cond1 == 'ctrl') + (cond2 == 'ctrl')
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False
        
def uncertainty_loss_fct(pred, logvar, y, perts, reg = 0.1, ctrl = None,
                         direction_lambda = 1e-3, dict_filter = None):
    """
    Uncertainty loss function

    Args:
        pred (torch.tensor): predicted values
        logvar (torch.tensor): log variance
        y (torch.tensor): true values
        perts (list): list of perturbations
        reg (float): regularization parameter
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2                     
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
            y_p = y[np.where(perts==p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts==p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts==p)[0]]
            y_p = y[np.where(perts==p)[0]]
            logvar_p = logvar[np.where(perts==p)[0]]
                         
        # uncertainty based loss
        losses += torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(
            -logvar_p)  * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
        # direction loss                 
        if p!= 'ctrl':
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl) -
                                 torch.sign(pred_p - ctrl))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
            
    return losses/(len(set(perts)))


def loss_fct(pred, y, perts, ctrl = None, direction_lambda = 1e-3, dict_filter = None):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        
        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
        losses = losses + torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
        ## direction loss
        if (p!= 'ctrl'):
            losses = losses + torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
                                                torch.sign(pred_p - ctrl))**2)/\
                                                pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))


def print_sys(s):
    print(s, flush = True, file = sys.stderr)
    
def create_cell_graph_for_prediction(X, pert_idx, pert_gene):
    """
    Create a perturbation specific cell graph for inference

    Args:
        X (np.array): gene expression matrix
        pert_idx (list): list of perturbation indices
        pert_gene (list): list of perturbations

    """

    if pert_idx is None:
        pert_idx = [-1]
    return Data(x=torch.Tensor(X).T, pert_idx = pert_idx, pert=pert_gene)
    

def create_cell_graph_dataset_for_prediction(pert_gene, ctrl_adata, gene_names,
                                             device, num_samples = 300):
    """
    Create a perturbation specific cell graph dataset for inference

    Args:
        pert_gene (list): list of perturbations
        ctrl_adata (anndata): control anndata
        gene_names (list): list of gene names
        device (torch.device): device to use
        num_samples (int): number of samples to use for inference (default: 300)

    """

    # Get the indices (and signs) of applied perturbation
    pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

    Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
    # Create cell graphs
    cell_graphs = [create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device) for X in Xs]
    return cell_graphs

def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))