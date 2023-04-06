import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_files = os.listdir(folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder, self.image_files[index])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(img)

        return img
    
# Variational Autoencoder model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
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
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
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

# Main function
def main():
    # Parameters
    epochs = 50
    batch_size = 16
    learning_rate = 1e-3
    train_folder = 'hazelnut/train/good'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageDataset(train_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        train_loss = train(model, dataloader, optimizer, device)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.6f}')

    # Save the trained model
    torch.save(model.state_dict(), 'vae.pth')

    # Inference on a test image
    test_image_path = 'hazelnut/test/hole/005.png'
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (224, 224))
    test_tensor = torch.tensor(test_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # Get the reconstructed image
    model.eval()
    with torch.no_grad():
        reconstructed_image, _, _ = model(test_tensor)
        reconstructed_image = reconstructed_image.squeeze().cpu().numpy()

    # Calculate the difference between the original and reconstructed image
    anomaly_map = np.abs(test_image / 255.0 - reconstructed_image)

    # Threshold the anomaly map
    threshold = 0.1  # You can adjust this value based on your dataset
    binary_map = np.uint8(anomaly_map > threshold)

    # Find contours in the binary map
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(test_image_rgb, contours, -1, (255, 0, 0), 2)

    # Display the images
    plt.figure(figsize=(16, 4))

    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(test_image_rgb)

    plt.subplot(142)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_image, cmap='gray')

    plt.subplot(143)
    plt.title('Binary Map')
    plt.imshow(binary_map, cmap='gray')

    plt.show()

if __name__ == '__main__':
    main()

