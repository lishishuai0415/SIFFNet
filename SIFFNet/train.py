# -*-coding:utf-8-*-
"""
Split dataset into train/validation/test sets and start network training
"""

from SIFFNet_main.model.SIFFNet import SIFFNet
from SIFFNet_main.utils.dataset_feature_label_segmentation import MyDataset

from torch import optim
import torch.nn as nn
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import losses

# Select device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize network (single channel input, 1 output class)
my_net = SIFFNet()
my_net.to(device=device)  # Move network to selected device

# Define dataset paths
train_path_x = "..\\data\\feature_VSP_0621\\"  # Feature data path
train_path_y = "..\\data\\label_VSP_0621\\"  # Label data path
train_path_z = "..\\data\\mask_VSP_0621\\"  # Mask data path

# Split dataset (80% train, 20% validation)
full_dataset = MyDataset(train_path_x, train_path_y, train_path_z)
valida_size = int(len(full_dataset) * 0.2)
train_size = len(full_dataset) - valida_size

# Data loader configuration
batch_size = 16  # Batch size for training
train_dataset, valida_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, valida_size])

# Create data loaders with shuffling
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
valida_loader = torch.utils.data.DataLoader(
    valida_dataset, batch_size=batch_size, shuffle=True)

# Training configuration
epochs = 200  # Total training epochs
LR = 0.0002  # Initial learning rate
optimizer = optim.Adam(my_net.parameters(), lr=LR)

# Loss functions definition
criterion1 = nn.MSELoss(reduction='mean')  # Mean Squared Error loss
criterion2 = losses.SSIM_loss()  # Structural Similarity loss
criterion3 = nn.BCELoss()  # Binary Cross Entropy loss

# Storage for loss values during training
temp_sets1 = []


# Learning rate scheduler (reduce LR after 40 epochs)
def lr_lambda(epoch):
    if epoch < 40:
        return 1.0
    else:
        return 0.5 ** ((epoch - 40) // 40 + 1)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop
start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())
for epoch in range(epochs):
    # Training phase
    my_net.train()
    train_loss = 0.0
    for batch_idx, (batch_x, batch_y, batch_z) in enumerate(train_loader):
        # Transfer data to device
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)
        batch_z = batch_z.to(device, dtype=torch.float32)

        # Forward pass
        err_out, err_out1 = my_net(batch_x)

        # Loss calculation (combination of MSE and BCE losses)
        loss22 = criterion3(err_out, batch_z)
        loss11 = criterion1(err_out1, (batch_x - batch_y))
        loss1 = loss11 + 0.001 * loss22

        # Backpropagation
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        train_loss += loss1.item()

    # Validation phase
    my_net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_x, val_y, val_z in valida_loader:
            # Transfer data to device
            val_x = val_x.to(device, dtype=torch.float32)
            val_y = val_y.to(device, dtype=torch.float32)
            val_z = val_z.to(device, dtype=torch.float32)

            # Forward pass
            err_out3, err_out2 = my_net(val_x)

            # Loss calculation
            loss222 = criterion3(err_out3, val_z)
            loss111 = criterion1(err_out2, (val_x - val_y))
            loss2 = loss111 + 0.001 * loss222

            val_loss += loss2.item()

    # Update learning rate and record metrics
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    train_loss /= (batch_idx + 1)
    val_loss /= (len(valida_loader) + 1)

    # Save metrics
    temp_sets1.append([train_loss, val_loss])
    print(f"epoch={epoch + 1}, train_loss={train_loss:.7f}, val_loss={val_loss:.7f}, lr={current_lr:.7f}")

    # Save model checkpoint
    model_name = f'model_epoch{epoch + 1}'
    torch.save(my_net, os.path.join(
        'D:/UNET/dncnn_pytorch/model/savedir_VSP_DCASSANB_20240711',
        f'{model_name}.pth'))

# Save training time record
end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())
with open('训练时间unet.txt', 'w') as f:
    f.write(f"Start time: {start_time}\nEnd time: {end_time}")

# Save and plot loss history
loss_sets = np.array([item for sublist in temp_sets1 for item in sublist]).reshape(-1, 2)
np.savetxt('loss_sets_VSP_14.txt', loss_sets, fmt='%.7f')

plt.plot(loss_sets[:, 0], label='Training Loss')
plt.plot(loss_sets[:, 1], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_VSP_14.png', bbox_inches='tight')

plt.show()


