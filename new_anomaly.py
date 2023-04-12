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


def compute_mean_std(folder):
    num_pixels = 0
    sum_pixel_values = 0
    sum_squared_pixel_values = 0

    image_files = os.listdir(folder)
    for image_file in image_files:
        img_path = os.path.join(folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        num_pixels += img.size
        sum_pixel_values += img.sum()
        sum_squared_pixel_values += np.sum(np.square(img))

    mean = sum_pixel_values / num_pixels
    var = sum_squared_pixel_values / num_pixels - mean**2
    var = np.clip(var, 0, None)  # Ensure the variance is non-negative
    std = np.sqrt(var)

    return mean, std


# Custom dataset class
class ImageDataset(Dataset):
    # Add mean and std parameters for normalization
    def __init__(self, folder, transform=None, mean=0.0, std=1.0):
        self.folder = folder
        self.transform = transform
        self.mean = mean
        self.std = std
        self.image_files = os.listdir(folder)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.folder, self.image_files[index])
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        img = img / 255.0  # Normalize to the range of [0, 1]
        img = img.astype(np.float32)

        if self.transform:
            img = self.transform(img)  # Apply the transform
        else:
            img = torch.tensor(img).unsqueeze(0)  # Convert the numpy array to a tensor
            
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
    epochs = 50
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
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (224, 224))

    test_image = (test_image - mean) / (std + 1e-7)
    test_image = test_image.astype(np.float32)

    # Create a new transform for the test image
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std + 1e-7,)),  # Add normalization to the test_transform
    ])


    test_image_tensor = test_transform(test_image).unsqueeze(0).to(device)

    diffs = []
    for img in inference_dataset:
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed_img, _, _ = model(img_tensor)
            reconstructed_img_np = reconstructed_img.squeeze().cpu().numpy()  # Fix the variable name
            diff = np.abs(img.cpu().numpy() - reconstructed_img_np)
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


    test_image_normalized = test_image_tensor.squeeze().cpu().numpy()
    anomaly_map = np.abs(test_image_normalized - reconstructed_test_image)
    anomaly_map = (anomaly_map > threshold).astype(np.uint8)

    # Draw contours on the original image
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
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

