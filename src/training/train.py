import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Load configuration
with open('/home/binit/classifier/src/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Paths and hyperparameters
train_dir = config['paths']['train_dir']
val_dir = config['paths']['val_dir']
log_dir = config['paths']['log_dir']
checkpoint_dir = config['logging']['checkpoint_dir']
final_model_dir = config['paths']['model']
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = int(config['hyperparameters']['batch_size'])
num_epochs = int(config['hyperparameters']['num_epochs'])
learning_rate = float(config['hyperparameters']['learning_rate'])
weight_decay = float(config['hyperparameters']['weight_decay'])
num_classes = int(config['hyperparameters']['num_classes'])
patience = int(config['hyperparameters']['patience'])
log_interval = int(config['logging']['log_interval'])
save_every_n_steps = int(config['hyperparameters']['save_every_n_steps'])
max_checkpoints = int(config['logging']['max_checkpoints'])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Early stopping and checkpoint settings
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoints_saved = []

# Define a custom dataset to load .pt files and label them based on folder
class ImageFolderWithLabels(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.class_to_label = {'AI': 0, 'real': 1}

        for class_name, label in self.class_to_label.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} does not exist.")
                continue
            for pt_file in os.listdir(class_dir):
                if pt_file.endswith(".pt"):
                    self.data.append((os.path.join(class_dir, pt_file), label))

        print(f"Loaded {len(self.data)} tensor files from {root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pt_path, label = self.data[idx]
        image_tensor = torch.load(pt_path)
        
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        
        return image_tensor, label

# Datasets and DataLoaders
train_dataset = ImageFolderWithLabels(train_dir)
val_dataset = ImageFolderWithLabels(val_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the Swin Transformer model with weights and set dropout if applicable
model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
model.head = nn.Sequential(
    nn.Dropout(config['dropout']['rate']),
    nn.Linear(model.head.in_features, num_classes)
)

# Freeze all layers except the head
for param in model.parameters():
    param.requires_grad = False
    
# Unfreeze the head
for param in model.head.parameters():
    param.requires_grad = True

# Unfreeze the last two stages in the features
for layer in model.features[-2:]:
    for param in layer.parameters():
        param.requires_grad = True


model = model.to(device)

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# TensorBoard setup
writer = SummaryWriter(log_dir=log_dir) if config['logging']['tensorboard'] else None

# Function to save checkpoint
def save_checkpoint(epoch, step, checkpoints_saved):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_step{step}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    checkpoints_saved.append(checkpoint_path)
    if len(checkpoints_saved) > max_checkpoints:
        oldest_checkpoint = checkpoints_saved.pop(0)
        os.remove(oldest_checkpoint)

# Training loop with early stopping, scheduler, and logging
step_count = 0
final_model_path = os.path.join(final_model_dir, 'best_model.pth')
os.makedirs(final_model_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            data_loader = train_loader
        else:
            model.eval()
            data_loader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(tqdm(data_loader)):
            inputs, labels = inputs.to(device), labels.to(device).long()
            step_count += 1

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                    # Logging batch metrics for TensorBoard if enabled
                    if writer and step_count % log_interval == 0:
                        writer.add_scalar('train_batch_loss', loss.item(), step_count)
                        writer.add_scalar('train_batch_accuracy', (preds == labels).float().mean().item(), step_count)

                    if step_count % save_every_n_steps == 0:
                        save_checkpoint(epoch + 1, step_count, checkpoints_saved)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        if writer:
            writer.add_scalar(f'{phase}_Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}_Accuracy', epoch_acc, epoch)

        # Adjust learning rate
        if phase == 'val':
            scheduler.step(epoch_loss)

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                epochs_no_improve = 0

                # Save the best model
                torch.save(model.state_dict(), final_model_path)
                print(f"Best model saved to {final_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered")
                    break

    if epochs_no_improve >= patience:
        break

# Close TensorBoard writer
if writer:
    writer.close()
