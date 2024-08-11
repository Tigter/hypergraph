import numpy as np
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
from  tqdm import tqdm
seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


class SequenceDataset(Dataset):
    def __init__(self, act_df, prot_len=2500):
        self.act_df = act_df
        self.prot_len = prot_len

    def __len__(self):
        return len(self.act_df)

    def __getitem__(self, idx):
        seq = self.act_df.iloc[idx]['encoded_sequence']
        seq_padded = pad_sequence([seq], batch_first=True, padding_value=0)
        mol = self.act_df.iloc[idx]['mol_feature']
        label = self.act_df.iloc[idx]['label']
        return seq_padded, mol, label


def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return torch.tensor([0])
    else:
        return torch.tensor([seq_dic[aa] for aa in seq], dtype=torch.long)

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i, w in enumerate(seq_rdic)}

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def parse_data(act_dir, mol_dir, protein_dir, with_label=True, prot_len=2500, prot_vec="Convolution", mol_vec="Convolution", mol_len=2048):
    print(f"Parsing {act_dir}, {mol_dir}, {protein_dir} with length {prot_len}, type {prot_vec}")

    protein_col = "Protein_ID"
    mol_col = "Compound_ID"
    col_names = [protein_col, mol_col]
    if with_label: 
        label_col = "Label"
        col_names.append(label_col)
    print(act_dir, mol_dir, protein_dir)
    act_df = pd.read_csv(act_dir)
    mol_df = pd.read_csv(mol_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")

    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df["Sequence"].map(lambda a: encodeSeq(a, seq_dic))

    act_df = pd.merge(act_df, protein_df, left_on=protein_col, right_index=True)
    act_df = pd.merge(act_df, mol_df, left_on=mol_col, right_index=True)

    mol_feature = np.stack(act_df[mol_vec].map(lambda fp: np.array(fp.split("\t"), dtype=float)))

    if prot_vec == "Convolution":
        protein_sequences = act_df["encoded_sequence"].values
        protein_feature = [torch.tensor(seq, dtype=torch.long) for seq in protein_sequences]
        protein_feature = pad_sequence(protein_feature, batch_first=True, padding_value=0)
        if protein_feature.size(1) > prot_len:
            protein_feature = protein_feature[:, :prot_len]
        else:
            padding = prot_len - protein_feature.size(1)
            protein_feature = F.pad(protein_feature, (0, padding), "constant", 0)
    else:
        protein_feature = np.stack(act_df[prot_vec].map(lambda fp: np.array(fp.split("\t"), dtype=float)))

    if with_label:
        label = act_df[label_col].values
        print(f"\tPositive data: {sum(act_df[label_col])}")
        print(f"\tNegative data: {act_df.shape[0] - sum(act_df[label_col])}")
        return {"protein_feature": protein_feature, "mol_feature": torch.tensor(mol_feature, dtype=torch.float32), 'Protein_ID': act_df['Protein_ID'].values, "Compound_ID": act_df['Compound_ID'].values ,"label": torch.tensor(label, dtype=torch.float32)}
    else:
        return {"protein_feature": protein_feature, "mol_feature": torch.tensor(mol_feature, dtype=torch.float32)}


def parse_data_test(act_dir, mol_dir, protein_dir, with_label=True, prot_len=2500, prot_vec="Convolution", mol_vec="Convolution", mol_len=2048):
    print(f"Parsing {act_dir}, {mol_dir}, {protein_dir} with length {prot_len}, type {prot_vec}")

    protein_col = "Protein_ID"
    mol_col = "Compound_ID"
    col_names = [protein_col, mol_col]
    if with_label: 
        label_col = "Label"
        col_names.append(label_col)
    print(act_dir, mol_dir, protein_dir)
    act_df = pd.read_csv(act_dir)
    # 筛选label等于1的
    act_df = act_df[act_df['Label']==1]

    mol_df = pd.read_csv(mol_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")

    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df["Sequence"].map(lambda a: encodeSeq(a, seq_dic))
        

    act_df = pd.merge(act_df, protein_df, left_on=protein_col, right_index=True)
    act_df = pd.merge(act_df, mol_df, left_on=mol_col, right_index=True)

    mol_feature = np.stack(act_df[mol_vec].map(lambda fp: np.array(fp.split("\t"), dtype=float)))

    if prot_vec == "Convolution":
        protein_sequences = act_df["encoded_sequence"].values
        protein_feature = [torch.tensor(seq, dtype=torch.long) for seq in protein_sequences]
        protein_feature = pad_sequence(protein_feature, batch_first=True, padding_value=0)
        if protein_feature.size(1) > prot_len:
            protein_feature = protein_feature[:, :prot_len]
        else:
            padding = prot_len - protein_feature.size(1)
            protein_feature = F.pad(protein_feature, (0, padding), "constant", 0)

        protein_feature_all = protein_df['encoded_sequence'].values
        protein_feature_all = [torch.tensor(seq, dtype=torch.long) for seq in protein_feature_all]
        protein_feature_all = pad_sequence(protein_feature_all, batch_first=True, padding_value=0)
        if protein_feature_all.size(1) > prot_len:
            protein_feature_all = protein_feature_all[:, :prot_len]
        else:
            padding = prot_len - protein_feature_all.size(1)
            protein_feature_all = F.pad(protein_feature_all, (0, padding), "constant", 0)
        protein_dict = {protein_df.index[i]: protein_feature_all[i] for i in range(len(protein_df))}
    else:
        protein_feature = np.stack(act_df[prot_vec].map(lambda fp: np.array(fp.split("\t"), dtype=float)))

    if with_label:
        label = act_df[label_col].values
        print(f"\tPositive data: {sum(act_df[label_col])}")
        print(f"\tNegative data: {act_df.shape[0] - sum(act_df[label_col])}")
        
        return {"protein_feature": protein_feature, "mol_feature": torch.tensor(mol_feature, dtype=torch.float32), 'Protein_ID': act_df['Protein_ID'].values, "Compound_ID": act_df['Compound_ID'].values ,"label": torch.tensor(label, dtype=torch.float32),'Protein_dict':protein_dict}
    else:
        return {"protein_feature": protein_feature, "mol_feature": torch.tensor(mol_feature, dtype=torch.float32)}

class PLayer(nn.Module):
    def __init__(self, size, filters, activation, initializer, regularizer_param):
        super(PLayer, self).__init__()
        self.conv1d = nn.Conv1d(1, filters, kernel_size=size, padding=size//2, bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.activation = activation
        self.l2_reg = nn.Parameter(torch.tensor(regularizer_param, dtype=torch.float))

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.activation(x)
        return nn.functional.max_pool1d(x, x.size(2)).squeeze(2)


class MolEnzPrediction(nn.Module):
    
    def __init__(self, dropout=0.2, mol_layers=(1024, 512), protein_strides=(10, 15, 20, 25), filters=64,
                 learning_rate=1e-3, decay=0.0, fc_layers=None, prot_vec=None, prot_len=2500, activation="relu",
                 mol_len=2048, mol_vec="ECFP4", protein_layers=None):
        super(MolEnzPrediction, self).__init__()
        
        self.dropout = dropout
        self.mol_layers = mol_layers
        self.protein_strides = protein_strides
        self.filters = filters
        self.fc_layers = fc_layers
        self.learning_rate = learning_rate
        self.prot_vec = prot_vec
        self.prot_len = prot_len
        self.mol_vec = mol_vec
        self.mol_len = mol_len
        self.activation = activation
        self.protein_layers = protein_layers
        self.decay = decay
        
        self.regularizer_param = 0.001
        self.initializer = nn.init.xavier_normal_
        
        # Molecular layers
        self.mol_layers_list = nn.ModuleList()
        input_dim = mol_len
        if mol_layers is not None:
            for layer_size in mol_layers:
                layer = nn.Linear(input_dim, layer_size)
                self.initializer(layer.weight)
                self.mol_layers_list.append(layer)
                self.mol_layers_list.append(nn.BatchNorm1d(layer_size))
                self.mol_layers_list.append(nn.ReLU())
                self.mol_layers_list.append(nn.Dropout(dropout))
                input_dim = layer_size
        
        # Protein layers
        if prot_vec == "Convolution":
            self.embedding = nn.Embedding(26, 20)
            self.initializer(self.embedding.weight)
            self.conv_layers = nn.ModuleList()
            self.spatial_dropout = nn.Dropout2d(0.2)
            for stride_size in protein_strides:
                self.conv_layers.append(nn.Conv1d(20, filters, kernel_size=stride_size, padding='same'))
        else:
            input_dim_p = prot_len
        
        if protein_layers:
            self.protein_layers_list = nn.ModuleList()
            input_dim_p = filters * len(protein_strides)
            for protein_layer in protein_layers:
                layer = nn.Linear(input_dim_p, layer_size)
                self.initializer(layer.weight)
                self.protein_layers_list.append(layer)
                self.protein_layers_list.append(nn.BatchNorm1d(protein_layer))
                self.protein_layers_list.append(nn.ReLU())
                self.protein_layers_list.append(nn.Dropout(dropout))
                input_dim_p = protein_layer
        
        # Fully connected layers
        self.fc_layers_list = nn.ModuleList()
        if fc_layers is not None:
            input_dim = input_dim_p + input_dim
            for fc_layer in fc_layers:
                self.fc_layers_list.append(nn.Linear(input_dim, fc_layer))
                self.fc_layers_list.append(nn.BatchNorm1d(fc_layer))
                self.fc_layers_list.append(nn.ReLU())
                input_dim = fc_layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.initializer(self.output_layer.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.BCELoss()

    
    def forward(self, mol_input, prot_input):
        # Molecular pathway
        x = mol_input
        for layer in self.mol_layers_list:
            x = layer(x)
        # Protein pathway
        if self.prot_vec == "Convolution":
            p = self.embedding(prot_input)
            p = p.permute(0, 2, 1)  # Change to (batch, channels, seq_len) for Conv1d
            p = self.spatial_dropout(p)
            ps = []
            for conv in self.conv_layers:
                p_conv = conv(p)
                p_conv = F.relu(p_conv)
                p_conv = F.max_pool1d(p_conv, p_conv.size(2)).squeeze(2)
                ps.append(p_conv)
            if len(ps) > 1:
                p = torch.cat(ps, dim=1)
            else:
                p = ps[0]
        else:
            p = prot_input
        if self.protein_layers:
            for layer in self.protein_layers_list:
                p = layer(p)
        
        # Concatenate
        x = torch.cat([x, p], dim=1)
        for layer in self.fc_layers_list:
            x = layer(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def fit(self, mol_feature, protein_feature, label, n_epoch=10, batch_size=32):
        dataset = torch.utils.data.TensorDataset(torch.tensor(mol_feature, dtype=torch.float32),
                                                 torch.tensor(protein_feature, dtype=torch.int32),
                                                 torch.tensor(label, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(n_epoch):
            self.train()
            epoch_loss = 0
            for mol_batch, prot_batch, label_batch in tqdm(dataloader):
                self.optimizer.zero_grad()
                mol_batch = mol_batch.cuda()
                prot_batch = prot_batch.cuda()
                output = self(mol_batch, prot_batch)
                label_batch = label_batch.cuda()
                loss = self.loss_fn(output, label_batch.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss/len(dataloader)}')
    
    def summary(self):
        print(self)
    
    def validation(self, mol_feature, protein_feature, label, output_file=None, n_epoch=10, batch_size=32, **kwargs):
        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ["window_sizes", "mol_layers", "fc_layers", "learning_rate"]])
            result_df = pd.DataFrame(data = [[self.protein_strides, self.mol_layers, self.fc_layers, self.learning_rate]] * n_epoch, columns=param_tuple)
            result_df["epoch"] = range(1, n_epoch + 1)
        result_dic = {dataset: {"AUC": [], "AUPR": [], "opt_threshold(AUPR)": [], "opt_threshold(AUC)": []} for dataset in kwargs}
        
        print("result dict: ", result_dic)        
        for epoch in range(n_epoch):
            # self.fit(mol_feature, protein_feature, label, 1, batch_size)
            for dataset in kwargs:
                print(f"\tPrediction of {dataset}")
                print(kwargs)
                test_p = torch.tensor(kwargs[dataset]["protein_feature"], dtype=torch.int32).cuda()
                test_d = torch.tensor(kwargs[dataset]["mol_feature"], dtype=torch.float32).cuda()
                test_label = kwargs[dataset]["label"]
                
                self.eval()
                with torch.no_grad():
                    prediction = self.forward(test_d, test_p).cpu().numpy()
                
                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label, prediction)
                distance = (1 - fpr) ** 2 + (1 - tpr) ** 2
                EERs = (1 - recall) / (1 - precision)
                positive = sum(test_label)
                negative = len(test_label) - positive
                ratio = negative / positive
                ratio = ratio.cpu().numpy()
                opt_t_AUC = thresholds_AUC[np.argmin(distance)]
                opt_t_AUPR = thresholds[np.argmin(np.abs(EERs - ratio))]
                AUPR = auc(recall, precision)
                
                print(f"\tArea Under ROC Curve(AUC): {AUC:.3f}")
                print(f"\tArea Under PR Curve(AUPR): {AUPR:.3f}")
                print(f"\tOptimal threshold(AUC)   : {opt_t_AUC:.3f}")
                print(f"\tOptimal threshold(AUPR)  : {opt_t_AUPR:.3f}")
                print("=================================================")
                
                result_dic[dataset]["AUC"].append(AUC)
                result_dic[dataset]["AUPR"].append(AUPR)
                result_dic[dataset]["opt_threshold(AUC)"].append(opt_t_AUC)
                result_dic[dataset]["opt_threshold(AUPR)"].append(opt_t_AUPR)
                self.save("./modesl_bt1024/model"+str(epoch)+".pth")
        if output_file:
            for dataset in kwargs:
                result_df[dataset, "AUC"] = result_dic[dataset]["AUC"]
                result_df[dataset, "AUPR"] = result_dic[dataset]["AUPR"]
                result_df[dataset, "opt_threshold(AUC)"] = result_dic[dataset]["opt_threshold(AUC)"]
                result_df[dataset, "opt_threshold(AUPR)"] = result_dic[dataset]["opt_threshold(AUPR)"]
            
            print(f"save to {output_file}")
            print(result_df)
            result_df.to_csv(output_file, index=False)
    
    def predict(self, mol_feature, protein_feature):
        self.eval()
        with torch.no_grad():
            output = self(torch.tensor(mol_feature, dtype=torch.float32).cuda(),
                          torch.tensor(protein_feature, dtype=torch.int32).cuda())
        return output.detach().cpu().numpy()
    
    def save(self, output_file):
        torch.save(self.state_dict(), output_file)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for seq_padded, mol_features, labels in dataloader:
        seq_padded, mol_features, labels = seq_padded.to(device), mol_features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(seq_padded, mol_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq_padded, mol_features, labels in dataloader:
            seq_padded, mol_features, labels = seq_padded.to(device), mol_features.to(device), labels.to(device)
            outputs = model(seq_padded, mol_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
    return total_loss / len(dataloader.dataset)


def main():
    # ... (rest of the main function, including model training, validation, and prediction)
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate, test deep learning model for prediction of mol-Enz interaction (act)\n
    Deep learning model will be built by Keras with tensorflow.\n
    You can set almost hyper-parameters as you want, See below parameter description\n
    act, mol and protein data must be written as csv file format. And feature should be tab-delimited format for script to parse data.\n
    Basically, this script builds convolutional neural network on sequence.\n
    If you don't want convolutional neural network but traditional dense layers on provide protein feature, specify type of feature and feature length.\n
    \n
    requirement\n
    ============================\n
    tensorflow > 1.0\n
    keras > 2.0\n
    numpy\n
    pandas\n
    scikit-learn\n
    ============================\n
    \n
    contact : dlsrnsladlek@gist.ac.kr\n
    """)
    # train_params
    parser.add_argument("--act_dir", help="Training act information [mol, Enz, label]")
    parser.add_argument("--mol_dir", help="Training mol information [mol, SMILES,[feature_name, ..]]")
    parser.add_argument("--protein_dir", help="Training protein information [protein, seq, [feature_name]]")

    parser.add_argument("--train_dict", help="Training protein information [protein, seq, [feature_name]]")
    parser.add_argument("--valid_dict", help="Training protein information [protein, seq, [feature_name]]")
    parser.add_argument("--test_dict", help="Training protein information [protein, seq, [feature_name]]")

    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--valind-name", '-nv', help="Name of test data sets", nargs="*")

    parser.add_argument("--test-act-dir", "-i", help="Test act [mol, Enz, [label]]", nargs="*")
    parser.add_argument("--test-mol-dir", "-d", help="Test mol information [mol, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test act", action="store_true")
    # structure_params
    parser.add_argument("--window-sizes", '-w', help="Window sizes for model (only works for Convolution)", default=None, nargs="*", type=int)
    parser.add_argument("--protein-layers","-p", help="Dense layers for protein", default=None, nargs="*", type=int)
    parser.add_argument("--mol-layers", '-c', help="Dense layers for mols", default=None, nargs="*", type=int)
    parser.add_argument("--fc-layers", '-f', help="Dense layers for concatenated layers of mol and Enz layer", default=None, nargs="*", type=int)
    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int, default=10)
    # type_params
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--mol-vec", "-V", help="Type of mol feature", type=str, default="morgan_fp")
    parser.add_argument("--mol-len", "-L", help="mol vector length", default=2048, type=int)
    # the other hyper-parameters
    parser.add_argument("--activation", "-a", help='Activation function of model', type=str)
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--n-filters", "-F", help="Number of filters for convolution layer, only works for Convolution", default=64, type=int)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=0.0, type=float)
    # mode_params
    parser.add_argument("--validation", help="Excute validation with independent data, will give AUC and AUPR (No prediction result)", action="store_true")
    parser.add_argument("--predict", help="Predict interactions of independent test set", action="store_true")
    # output_params
    parser.add_argument("--save-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)

    args = parser.parse_args()

    # train data
    args.act_dir = os.path.join(args.train_dict, "training_act.csv")
    args.mol_dir = os.path.join(args.train_dict, "training_compound.csv")
    args.protein_dir = os.path.join(args.train_dict, "training_protein.csv")

  
    train_dic = {
        "act_dir": args.act_dir,
        "mol_dir": args.mol_dir,
        "protein_dir": args.protein_dir,
        "with_label": True
    }
    # create dictionary of test_data
    test_names = args.test_name

    valid_names = args.valind_name
    valid_act = [os.path.join(args.valid_dict, "validation_act.csv")]
    valid_protein = [os.path.join(args.valid_dict, "validation_protein.csv")]
    valid_mol = [os.path.join(args.valid_dict, "validation_compound.csv")]
    valid_sets = zip(valid_names, valid_act, valid_mol, valid_protein)

    test_act = [os.path.join(args.test_dict, "test_act.csv")]
    test_protein = [os.path.join(args.test_dict, "test_protein.csv")]
    test_mol = [os.path.join(args.test_dict, "test_compound.csv")]
    test_sets = zip(test_names, test_act, test_mol, test_protein)

    output_file = args.output
    # model_structure variables
    mol_layers = args.mol_layers
    window_sizes = args.window_sizes
    if window_sizes==0:
        window_sizes = None
    protein_layers = args.protein_layers
    fc_layers = args.fc_layers
    # training parameter
    train_params = {
        "n_epoch": args.n_epoch,
        "batch_size": args.batch_size,
    }
    # type parameter
    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "mol_vec": args.mol_vec,
        "mol_len": args.mol_len,
    }
    # model parameter
    model_params = {
        "mol_layers": mol_layers,
        "protein_strides": window_sizes,
        "protein_layers": protein_layers,
        "fc_layers": fc_layers,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "activation": args.activation,
        "filters": args.n_filters,
        "dropout": args.dropout
    }

    model_params.update(type_params)
    print("\tmodel parameters summary\t")
    print("=====================================================")
    for key in model_params.keys():
        print("{:20s} : {:10s}".format(key, str(model_params[key])))
    print("=====================================================")

    act_prediction_model = MolEnzPrediction(**model_params)
    act_prediction_model = act_prediction_model
    act_prediction_model.load_state_dict(torch.load('/home/skl/yl/ce_project/relation_cl/enzrank_model/modesl_bt1024/model96.pth'))
    act_prediction_model = act_prediction_model.cuda()
    act_prediction_model.summary()

    # read and parse training and test data
    train_dic.update(type_params)
    train_dic = parse_data(**train_dic)

    test_dic = {test_name: parse_data_test(test_act, test_mol, test_protein, with_label=True, **type_params) for test_name, test_act, test_mol, test_protein in test_sets}
    valid_dic = {valid_name: parse_data_test(valid_act, valid_mol, valid_protein, with_label=True, **type_params) for valid_name, valid_act, valid_mol, valid_protein in valid_sets}
    # validation mode
    # if args.validation:
    #     validation_params = {}
    #     validation_params.update(train_params)
    #     validation_params["output_file"] = output_file
    #     print("\tvalidation summary\t")
    #     print("=====================================================")
    #     for key in validation_params.keys():
    #         print("{:20s} : {:10s}".format(key, str(validation_params[key])))
    #     print("=====================================================")
    #     validation_params.update(train_dic)
    #     validation_params.update(valid_dic)
    #     act_prediction_model.validation(**validation_params)
    # prediction mode
    import json
    with open("/home/skl/yl/ce_project/relation_cl/enzrank_model/data_any/test_data/neg_pairs.json" , 'r') as f:
        neg_pairs = json.load(f)
    test_dic = valid_dic
    if args.predict:
        print("prediction")
        train_dic.update(test_dic)
        data = test_dic['30']
        mol_feature = data['mol_feature']
        protein_feature = data['protein_feature']
        label = data['label']
        Compound_ID = data['Compound_ID']
        Protein_ID = data['Protein_ID']
        protein_feature_all = data['Protein_dict']

        result = []
        for i in range(len(mol_feature)):
            test_mol_feature = mol_feature[i]
            test_label = label[i]

            test_Compound_ID = Compound_ID[i]
            test_Protein_ID = Protein_ID[i]
            test_neg_protein = np.random.choice(neg_pairs[test_Compound_ID], 2000,replace=False)
            protein_feature_list = [protein_feature[i]]
            for p in test_neg_protein:
                test_protein_feature_neg = protein_feature_all[p]
                protein_feature_list.append(test_protein_feature_neg)
            test_protein_feature = np.stack(tuple(protein_feature_list), axis=0)
            # test_mol_feature 在第一维度复制到和test_protein_feature一样的维度
            test_mol_feature = np.repeat(test_mol_feature[np.newaxis, :], test_protein_feature.shape[0], axis=0)
            
            prediction = act_prediction_model.predict(test_mol_feature,test_protein_feature)

            result_dict = {}
            result_dict['Compound_ID'] = test_Compound_ID
            result_dict['Protein_ID'] = test_Protein_ID
            result_dict["neg_protein"] = test_neg_protein.tolist()
            result_dict['prediction'] = np.concatenate(prediction.tolist(),axis=0).tolist()
            result.append(result_dict)

        with open(output_file, 'w') as f:
            json.dump(result, f)

        # test_predicted = act_prediction_model.predict(test_dic["30"]['mol_feature'],test_dic["30"]['protein_feature'])
        # prediction = test_predicted
        # test_label = test_dic["30"]['label']
        # fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
        # AUC = auc(fpr, tpr)
        # precision, recall, thresholds = precision_recall_curve(test_label, prediction)
        # distance = (1 - fpr) ** 2 + (1 - tpr) ** 2
        # EERs = (1 - recall) / (1 - precision)
        # positive = sum(test_label)
        # negative = len(test_label) - positive
        # ratio = negative / positive
        # ratio = ratio.cpu().numpy()
        # opt_t_AUC = thresholds_AUC[np.argmin(distance)]
        # opt_t_AUPR = thresholds[np.argmin(np.abs(EERs - ratio))]
        # AUPR = auc(recall, precision)
        
        # print(f"\tArea Under ROC Curve(AUC): {AUC:.3f}")
        # print(f"\tArea Under PR Curve(AUPR): {AUPR:.3f}")
        # print(f"\tOptimal threshold(AUC)   : {opt_t_AUC:.3f}")
        # print(f"\tOptimal threshold(AUPR)  : {opt_t_AUPR:.3f}")
        # print("=================================================")
        # result_df = pd.DataFrame()
        # result_columns = []
        # temp_df = pd.DataFrame()
        # value = test_predicted
        # value = np.squeeze(value)
        # print(value.shape)
        # dataset = "30"

        # temp_df[dataset,'predicted'] = value
        # temp_df[dataset, 'label'] = np.squeeze(test_dic[dataset]['label'])
        # temp_df[dataset, 'Compound_ID'] = np.squeeze(test_dic[dataset]['Compound_ID'])
        # temp_df[dataset, 'Protein_ID'] = np.squeeze(test_dic[dataset]['Protein_ID'])
        # result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
        # result_columns.append((dataset, "predicted"))
        # result_columns.append((dataset, "label"))
        # result_columns.append((dataset, "Compound_ID"))
        # result_columns.append((dataset, "Protein_ID"))
        # result_df.columns = pd.MultiIndex.from_tuples(result_columns)
        # print("save to %s"%output_file)
        # result_df.to_csv(output_file, index=False)

    # save trained model
    # if args.save_model:
    # act_prediction_model.save(args.save_model)
    # exit()

if __name__ == '__main__':
    main()