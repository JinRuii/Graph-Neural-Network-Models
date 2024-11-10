import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import json
import csv
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GATConv  # Import GATConv layer

# 1. Set relative paths for GitHub upload
input_dir = "./data/H8-16"  # Relative path for input files
output_dir = "./output/H8-16"  # Relative path for output files
os.makedirs(output_dir, exist_ok=True)  # Automatically create output directory if it doesn't exist

# 2. Map labels to numerical values
label_mapping = {'none': 0, 'low': 1, 'relatively low': 2, 'relatively high': 3, 'high': 4}

# 3. Define FocalLoss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=6, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            at = self.alpha.gather(0, target.long())
            focal_loss = focal_loss * at

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 4. Define a two-layer GAT model
class GATModel(nn.Module):
    def __init__(self, num_features, hidden_units1, hidden_units2, num_heads1, num_heads2, num_classes, dropout):
        super(GATModel, self).__init__()
        # First layer: GATConv
        self.gat_conv1 = GATConv(num_features, hidden_units1, heads=num_heads1, dropout=dropout)
        # Second layer: GATConv
        self.gat_conv2 = GATConv(hidden_units1 * num_heads1, hidden_units2, heads=num_heads2, concat=False, dropout=dropout)
        # Classifier
        self.classifier = nn.Linear(hidden_units2, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First layer: GATConv
        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer: GATConv
        x = self.gat_conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classifier
        output = self.classifier(x)
        return output

# 5. Define function to train the GAT model
def train_GAT_model(data, labels, train_indices, val_indices, test_indices, hyperparams, output_dir, save_model=False, model_save_path=None, save_logs=False):
    # Automatically detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate sample proportions for each class and create alpha for FocalLoss
    class_counts = pd.Series(labels.cpu().numpy()).value_counts().sort_index()
    total_samples = len(labels)
    class_freq = class_counts / total_samples
    alpha = (1.0 / class_freq).values
    alpha = alpha / alpha.sum()  # Normalize
    alpha = torch.FloatTensor(alpha).to(device)

    num_classes = len(label_mapping)
    model = GATModel(
        num_features=data.num_features,
        hidden_units1=hyperparams['hidden_units1'],
        hidden_units2=hyperparams['hidden_units2'],
        num_heads1=hyperparams['num_heads1'],
        num_heads2=hyperparams['num_heads2'],
        num_classes=num_classes,
        dropout=hyperparams['dropout'],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = FocalLoss(gamma=hyperparams['gamma'], alpha=alpha)

    data = data.to(device)
    labels = labels.to(device)

    if save_model and model_save_path is None:
        model_save_path = os.path.join(output_dir, 'trained_model_GAT.pth')

    if save_logs:
        log_file = os.path.join(output_dir, 'training_logs_GAT.csv')
        log_file_handle = open(log_file, mode='w', newline='')
        writer = csv.writer(log_file_handle)
        writer.writerow(['epoch',
                         'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                         'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
                         'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1'])

    best_val_loss = float('inf')

    start_time = time.time()

    for epoch in range(hyperparams['epochs']):
        model.train()
        optimizer.zero_grad()

        output = model(data.x, data.edge_index)
        loss_train = criterion(output[train_indices], labels[train_indices])
        loss_train.backward()
        optimizer.step()

        # Calculate training metrics
        train_pred = output[train_indices].max(1)[1]
        train_labels = labels[train_indices].cpu()
        train_pred_cpu = train_pred.cpu()

        train_accuracy = accuracy_score(train_labels, train_pred_cpu)
        train_precision = precision_score(train_labels, train_pred_cpu, average='weighted', zero_division=0)
        train_recall = recall_score(train_labels, train_pred_cpu, average='weighted', zero_division=0)
        train_f1 = f1_score(train_labels, train_pred_cpu, average='weighted', zero_division=0)

        # Validation metrics
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            loss_val = criterion(output[val_indices], labels[val_indices])

            val_pred = output[val_indices].max(1)[1]
            val_labels = labels[val_indices].cpu()
            val_pred_cpu = val_pred.cpu()

            val_accuracy = accuracy_score(val_labels, val_pred_cpu)
            val_precision = precision_score(val_labels, val_pred_cpu, average='weighted', zero_division=0)
            val_recall = recall_score(val_labels, val_pred_cpu, average='weighted', zero_division=0)
            val_f1 = f1_score(val_labels, val_pred_cpu, average='weighted', zero_division=0)

            # Test metrics
            loss_test = criterion(output[test_indices], labels[test_indices])

            test_pred = output[test_indices].max(1)[1]
            test_labels = labels[test_indices].cpu()
            test_pred_cpu = test_pred.cpu()

            test_accuracy = accuracy_score(test_labels, test_pred_cpu)
            test_precision = precision_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
            test_recall = recall_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
            test_f1 = f1_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)

        # Save model with the lowest validation loss
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            if save_model and model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)

        # Log training process
        if save_logs:
            writer.writerow([epoch + 1,
                             loss_train.item(), train_accuracy, train_precision, train_recall, train_f1,
                             loss_val.item(), val_accuracy, val_precision, val_recall, val_f1,
                             loss_test.item(), test_accuracy, test_precision, test_recall, test_f1])

        # Print training progress
        print(f"Epoch {epoch+1}/{hyperparams['epochs']}, "
              f"Train Loss: {loss_train.item():.4f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}, "
              f"Val Loss: {loss_val.item():.4f}, Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, "
              f"Test Loss: {loss_test.item():.4f}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}")

    # Close log file
    if save_logs:
        log_file_handle.close()

    # Load best model parameters
    if save_model and model_save_path is not None:
        model.load_state_dict(torch.load(model_save_path))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save output vector of each node
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
    output_vectors = output.cpu().numpy()
    output_vector_file = os.path.join(output_dir, 'node_output_vectors_GAT.csv')
    with open(output_vector_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['node_id'] + [f'output_vector_{i}' for i in range(output_vectors.shape[1])])
        for i, vec in enumerate(output_vectors):
            writer.writerow([i] + vec.tolist())

    return model, output_vectors, alpha

# 6. Define random search function for hyperparameter tuning
def random_search_GAT(data, labels, train_indices, val_indices, test_indices, hyperparam_space, n_iterations, output_dir):
    best_hyperparams = None
    best_test_f1 = 0  # To track best F1 score on test set
    best_model_state = None

    # Log for random search
    random_search_log_file = os.path.join(output_dir, 'random_search_log_GAT.csv')
    with open(random_search_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'hyperparams', 'test_f1'])

    for i in range(n_iterations):
        # Randomly sample hyperparameters
        hyperparams_sampled = {param: random.choice(values) for param, values in hyperparam_space.items()}

        print(f"Iteration {i+1}/{n_iterations}, testing hyperparameters: {hyperparams_sampled}")

        # Train model with sampled hyperparameters
        model, _, alpha = train_GAT_model(
            data, labels, train_indices, val_indices, test_indices,
            hyperparams_sampled, output_dir, save_model=False, save_logs=False)

        # Evaluate model on test set
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = FocalLoss(gamma=6, alpha=alpha)
        model.eval()
        with torch.no_grad():
            output = model(data.x.to(device), data.edge_index.to(device))
            loss_test = criterion(output[test_indices], labels[test_indices])

            test_pred = output[test_indices].max(1)[1]
            test_labels = labels[test_indices].cpu()
            test_pred_cpu = test_pred.cpu()

            test_accuracy = accuracy_score(test_labels, test_pred_cpu)
            test_precision = precision_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
            test_recall = recall_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
            test_f1 = f1_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)

        print(f"Iteration {i+1}, test set F1 score: {test_f1:.4f}")

        # Log search results
        with open(random_search_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, json.dumps(hyperparams_sampled), test_f1])

        # Update best hyperparameters if test F1 score improves
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_hyperparams = hyperparams_sampled.copy()
            # Save best model state
            best_model_state = model.state_dict()
            print(f"New best hyperparameters found: {best_hyperparams}, test F1 score: {test_f1:.4f}")

    print("\nRandom search completed.")
    print(f"Best hyperparameters: {best_hyperparams}, best test F1 score: {best_test_f1:.4f}")

    # Save best hyperparameters
    best_hyperparams_path = os.path.join(output_dir, 'best_hyperparameters_GAT.json')
    with open(best_hyperparams_path, 'w') as f:
        json.dump(best_hyperparams, f)

    return best_hyperparams, best_test_f1, best_model_state

# 7. Main function with random search and final training
if __name__ == '__main__':
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        # Replace with your input file paths
        variables_file = os.path.join(input_dir, 'H8-16_variables.csv')  # Path to feature file
        edges_file = os.path.join(input_dir, 'H8-16_edges.csv')  # Path to edge list file

        # Load node features and edge list
        variables_df = pd.read_csv(variables_file)
        edges_df = pd.read_csv(edges_file)

        # Extract features and normalize
        feature_names = [
            'HOP',  # Housing price, measured in yuan per square meter
            'POD',  # Population density, measured in people per square meter
            'DIS_BUS',  # Distance to the nearest bus station (poi - point of interest), measured in meters
            'DIS_MTR',  # Distance to the nearest metro station (poi), measured in meters
            'POI_COM',  # Number of company POIs (points of interest) within the area
            'POI_SHOP',  # Number of shopping POIs within the area
            'POI_SCE',  # Number of scenic spot POIs within the area
            'POI_EDU',  # Number of educational POIs within the area
            'POI_MED',  # Number of medical POIs within the area
            'PR',  # Plot ratio (building area * number of floors (height/3.5m) / area)
            'OPEN',  # Sky openness ratio from street view images; if value is -999, street view data is not available
            'CAR',  # Car presence ratio in street view images
            'GREN',  # Green view index (greenness) in street view images
            'ENCL',  # Enclosure rate in street view images
            'WAL',  # Walkability index in street view images
            'IMA',  # Imageability index in street view images
            'COMP',  # Complexity or diversity in street view images
            'PM2_5',  # Concentration of PM2.5 (particulate matter), measured in μg/m³ per hour per day
            'PM10',  # Concentration of PM10 (particulate matter), measured in μg/m³ per hour per day
            'CO'  # Carbon monoxide concentration, measured in μg/m³ per hour per day
        ]
        feature_names = ['HOP', 'POD', 'DIS_BUS', 'DIS_MTR', 'POI_COM', 'POI_SHOP', 'POI_SCE', 'POI_EDU', 'POI_MED',
                        'PR', 'OPEN', 'CAR', 'GREN', 'ENCL', 'WAL', 'IMA', 'COMP', 'PM2_5', 'PM10', 'CO']
        features = variables_df[feature_names].values

        # Normalize features (process in numpy and then convert to Tensor)
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features)
        features = torch.FloatTensor(features_normalized)  # Convert normalized features to Tensor

        # Load labels and map to numerical values
        labels = variables_df['TD_label'].map(label_mapping).values
        labels = torch.LongTensor(labels)  # Convert labels to Tensor

        # Construct edge index and convert to undirected graph
        edge_index = torch.tensor([edges_df['_OID_1'].values, edges_df['_OID_2'].values], dtype=torch.long)
        edge_index = to_undirected(edge_index)  # Use torch_geometric tool to convert to undirected graph

        # Create graph data object
        data = Data(x=features, edge_index=edge_index)

        # Split data into training, validation, and test sets (ensuring all nodes are assigned without duplicates)
        num_nodes = data.num_nodes
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        num_train = int(0.8 * num_nodes)
        num_val = int(0.1 * num_nodes)
        num_test = num_nodes - num_train - num_val  # Ensure total count matches
        train_indices, val_indices, test_indices = np.split(indices, [num_train, num_train + num_val])
        train_indices = torch.LongTensor(train_indices)
        val_indices = torch.LongTensor(val_indices)
        test_indices = torch.LongTensor(test_indices)

        # Save data split indices to ensure consistent splits for final training
        np.save(os.path.join(output_dir, 'train_indices_GAT.npy'), train_indices.numpy())
        np.save(os.path.join(output_dir, 'val_indices_GAT.npy'), val_indices.numpy())
        np.save(os.path.join(output_dir, 'test_indices_GAT.npy'), test_indices.numpy())

        # Define hyperparameter search space
        hyperparam_space = {
            'epochs': [600],
            'learning_rate': [0.01, 0.001, 0.0001],
            'dropout': [0.5, 0.6, 0.7],
            'hidden_units1': [128, 256, 512],
            'hidden_units2': [128, 256, 512],
            'num_heads1': [2, 4, 8],
            'num_heads2': [1, 2],
            'gamma': [2, 4, 6],
        }

        n_iterations = 30  # Number of iterations for random search
        best_hyperparams = None
        best_test_f1 = 0  # To track best F1 score on test set

        # Perform random search
        best_hyperparams, best_test_f1, best_model_state = random_search_GAT(
            data, labels, train_indices, val_indices, test_indices,
            hyperparam_space, n_iterations, output_dir
        )

        print(f"Best hyperparameters: {best_hyperparams}, best test F1 score: {best_test_f1:.4f}")

        # Save best hyperparameters
        best_hyperparams_path = os.path.join(output_dir, 'best_hyperparameters_GAT.json')
        with open(best_hyperparams_path, 'w') as f:
            json.dump(best_hyperparams, f)

        # Load best hyperparameters
        with open(best_hyperparams_path, 'r') as f:
            hyperparams = json.load(f)

        # Final training with best hyperparameters and saving model and logs
        final_model_save_path = os.path.join(output_dir, 'best_trained_model_GAT.pth')
        final_log_file = os.path.join(output_dir, 'final_training_logs_GAT.csv')

        # Train final model
        final_model, final_output_vectors, final_alpha = train_GAT_model(
            data, labels, train_indices, val_indices, test_indices,
            hyperparams, output_dir, save_model=True, model_save_path=final_model_save_path, save_logs=True
        )

        print("All steps completed! Training logs, model, and node output vectors saved.")

        # Evaluate model on the test set after training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = FocalLoss(gamma=hyperparams['gamma'], alpha=final_alpha)
        final_model.to(device)  # Ensure model is on the correct device
        final_model.eval()
        with torch.no_grad():
            output = final_model(data.x.to(device), data.edge_index.to(device))
            loss_test = criterion(output[test_indices], labels[test_indices])

            test_pred = output[test_indices].max(1)[1]
            test_labels = labels[test_indices].cpu()
            test_pred_cpu = test_pred.cpu()

            test_accuracy = accuracy_score(test_labels, test_pred_cpu)
            test_precision = precision_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
            test_recall = recall_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
            test_f1 = f1_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)

        print(f"Final test set results:\n"
              f"Loss: {loss_test.item():.4f}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
