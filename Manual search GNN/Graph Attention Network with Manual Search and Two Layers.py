import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# 1. Set input and output directories
input_dir = "./data/W8-15"  # Input directory
output_dir = "./output/W8-15"  # Output directory
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# 2. Model hyperparameter settings
hyperparams = {
    'epochs': 500,
    'learning_rate': 0.0009,
    'dropout': 0.6,
    'hidden_units1': 256,  # Hidden units for the first GAT layer
    'hidden_units2': 256,  # Hidden units for the second GAT layer
    'num_heads1': 2,       # Attention heads for the first layer
    'num_heads2': 1,       # Attention heads for the second layer
}

# Label mapping for categorical values
label_mapping = {'none': 0, 'low': 1, 'relatively low': 2, 'relatively high': 3, 'high': 4}

# Define FocalLoss loss function
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

# Define a two-layer GAT model
class GATModel(nn.Module):
    def __init__(self, num_features, hidden_units1, hidden_units2, num_heads1, num_heads2, num_classes, dropout):
        super(GATModel, self).__init__()
        self.gat_conv1 = GATConv(num_features, hidden_units1, heads=num_heads1, dropout=dropout)
        self.gat_conv2 = GATConv(hidden_units1 * num_heads1, hidden_units2, heads=num_heads2, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_units2, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat_conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        output = self.classifier(x)
        return output

# Train the GAT model and save node vectors
def train_GAT_model(data, labels, train_indices, val_indices, test_indices, hyperparams, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate class sample proportions and generate alpha for focal loss
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
    criterion = FocalLoss(gamma=6, alpha=alpha)

    data = data.to(device)
    labels = labels.to(device)

    model_save_path = os.path.join(output_dir, 'trained_model_GAT.pth')
    log_file = os.path.join(output_dir, 'training_logs_GAT.csv')
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
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

            train_pred = output[train_indices].max(1)[1]
            train_labels = labels[train_indices].cpu()
            train_pred_cpu = train_pred.cpu()

            train_accuracy = accuracy_score(train_labels, train_pred_cpu)
            train_precision = precision_score(train_labels, train_pred_cpu, average='weighted', zero_division=0)
            train_recall = recall_score(train_labels, train_pred_cpu, average='weighted', zero_division=0)
            train_f1 = f1_score(train_labels, train_pred_cpu, average='weighted', zero_division=0)

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

                loss_test = criterion(output[test_indices], labels[test_indices])
                test_pred = output[test_indices].max(1)[1]
                test_labels = labels[test_indices].cpu()
                test_pred_cpu = test_pred.cpu()

                test_accuracy = accuracy_score(test_labels, test_pred_cpu)
                test_precision = precision_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
                test_recall = recall_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)
                test_f1 = f1_score(test_labels, test_pred_cpu, average='weighted', zero_division=0)

            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                torch.save(model.state_dict(), model_save_path)

            writer.writerow([epoch + 1,
                             loss_train.item(), train_accuracy, train_precision, train_recall, train_f1,
                             loss_val.item(), val_accuracy, val_precision, val_recall, val_f1,
                             loss_test.item(), test_accuracy, test_precision, test_recall, test_f1])

            print(f"Epoch {epoch+1}/{hyperparams['epochs']}, "
                  f"Train Loss: {loss_train.item():.4f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}, "
                  f"Val Loss: {loss_val.item():.4f}, Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, "
                  f"Test Loss: {loss_test.item():.4f}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}")

        model.load_state_dict(torch.load(model_save_path))
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f} seconds")

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

    return model, output.cpu().numpy(), alpha

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        variables_file = os.path.join(input_dir, 'W8-15_variables.csv')
        edges_file = os.path.join(input_dir, 'W8-15_edges.csv')

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

        variables_df = pd.read_csv(variables_file)
        edges_df = pd.read_csv(edges_file)

        features = variables_df[feature_names].values
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features)
        features = torch.FloatTensor(features_normalized)

        labels = variables_df['TD_label'].map(label_mapping).values
        labels = torch.LongTensor(labels)

        edge_index = torch.tensor([edges_df['_OID_1'].values, edges_df['_OID_2'].values], dtype=torch.long)
        edge_index = to_undirected(edge_index)

        data = Data(x=features, edge_index=edge_index)

        num_nodes = data.num_nodes
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        num_train = int(0.8 * num_nodes)
        num_val = int(0.1 * num_nodes)
        num_test = num_nodes - num_train - num_val
        train_indices, val_indices, test_indices = np.split(indices, [num_train, num_train + num_val])
        train_indices = torch.LongTensor(train_indices)
        val_indices = torch.LongTensor(val_indices)
        test_indices = torch.LongTensor(test_indices)

        model, output_vectors, alpha = train_GAT_model(data, labels, train_indices, val_indices, test_indices, hyperparams, output_dir)

        print("All steps completed! Training logs, model, and node output vectors have been saved.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = FocalLoss(gamma=6, alpha=alpha)
        model.to(device)
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

        print(f"Test results:\n"
              f"Loss: {loss_test.item():.4f}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}")

        model_save_path = os.path.join(output_dir, 'trained_model_GAT.pth')
        torch.save(model.state_dict(), model_save_path)

    except Exception as e:
        print(f"An error occurred: {e}")
