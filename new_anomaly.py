import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import glob

def compute_mean_std(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.png'))

    mean_list = []
    std_list = []

    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img /= 255.0
        means = [np.mean(img[:, :, i]) for i in range(3)]
        stds = [np.std(img[:, :, i]) for i in range(3)]
        mean_list.append(means)
        std_list.append(stds)

    mean_list = np.array(mean_list)
    std_list = np.array(std_list)
    mean = np.mean(mean_list, axis=0)
    std = np.mean(std_list, axis=0)

    return mean, std


# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, folder, transform=None, mean=None, std=None):
        self.folder = folder
        self.image_files = sorted(glob.glob(os.path.join(folder, '*.png')))
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32)
        img /= 255.0  # Normalize to the range of [0, 1]
        img = (img * 255).astype(np.uint8)  # Convert the image back to uint8 before applying the transform

        if self.transform:
            img = self.transform(img)  # Apply the transform
        else:
            img = torch.tensor(img).permute(2, 0, 1)  # Convert the numpy array to a tensor

        return img


# Variational Autoencoder model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 512),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(512, 128)
        self.fc_logvar = nn.Linear(512, 128)

        self.decoder_input = nn.Linear(128, 32 * 56 * 56)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder_input(z).view(-1, 32, 56, 56)
        x_recon = F.relu(x_recon)  # Add this line
        return self.decoder(x_recon), mu, logvar

# VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss /= x.size(0) * 224 * 224
    return recon_loss + kld_loss

# Training function
def train(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0

    for batch in dataloader:
        inputs = batch.to(device)
        optimizer.zero_grad()
        outputs, mu, logvar = model(inputs)
        loss = vae_loss(outputs, inputs, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0

    for batch in dataloader:
        inputs = batch.to(device)
        with torch.no_grad():
            outputs, mu, logvar = model(inputs)
            loss = vae_loss(outputs, inputs, mu, logvar)
        val_loss += loss.item()

    return val_loss / len(dataloader)

# Main function
def main():
    # Parameters
    epochs = 100
    batch_size = 16
    learning_rate = 1e-3
    train_folder = 'transistor/train/good'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    mean, std = compute_mean_std(train_folder)

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the ndarray to a PIL image
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(train_folder, transform=transform, mean=mean, std=std)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Create dataset and dataloader for inference
    inference_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the ndarray to a PIL image
        transforms.ToTensor(),
    ])


    inference_dataset = ImageDataset(train_folder, transform=inference_transform, mean=mean, std=std)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    # Initialize model, loss, and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        val_loss = validate(model, val_dataloader, device)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'vae.pth')
        else:
            counter += 1
            print(f'Early stopping counter: {counter} out of {patience}')        
            if counter >= patience:
                print('Early stopping')
                break

    # Load the trained model
    model.load_state_dict(torch.load('vae.pth'))

    # Inference on a test image
    test_image_path = 'transistor/test/damaged_case/009.png'
    test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
    test_image = cv2.resize(test_image, (224, 224))

    mean, std = compute_mean_std(train_folder)

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_image_tensor = test_transform(test_image).unsqueeze(0).to(device)

    diffs = []
    for img_batch in inference_dataloader:
        img_np = img_batch.squeeze().numpy()
        img_tensor = img_batch.to(device)

        with torch.no_grad():
            reconstructed_img, _, _ = model(img_tensor)
            reconstructed_img_np = reconstructed_img.squeeze().cpu().numpy()
            diff = np.abs(img_np - reconstructed_img_np)
            diffs.append(diff)

    diffs = np.array(diffs)


    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)



    # Calculate the difference between the original and reconstructed image

    # Threshold the anomaly map
    threshold = mean_diff + 2 * std_diff

    # Create anomaly map
    with torch.no_grad():
        reconstructed_test_image, _, _ = model(test_image_tensor)
        reconstructed_test_image = reconstructed_test_image.squeeze().cpu().numpy()


    test_image_normalized = test_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    reconstructed_test_image = reconstructed_test_image.transpose(1, 2, 0)  # Change the shape from (3, 224, 224) to (224, 224, 3)
    anomaly_map = np.abs(test_image_normalized - reconstructed_test_image)

    anomaly_map = np.mean(anomaly_map, axis=-1)  # Calculate mean along the channel axis
    anomaly_map = (anomaly_map > threshold).astype(np.uint8)

    # Draw contours on the original image
    test_image_rgb = test_image
    contours, _ = cv2.findContours(anomaly_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(test_image_rgb, contours, -1, (0, 255, 0), 2)

    # Display the images
    plt.figure(figsize=(16, 4))

    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(test_image_rgb)

    plt.subplot(142)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_test_image, cmap='gray')

    plt.subplot(143)
    plt.title('Anomaly Map')
    plt.imshow(anomaly_map, cmap='gray')
    
    plt.show()


if __name__ == '__main__':
    main()

