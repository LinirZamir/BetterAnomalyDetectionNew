import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog, QGraphicsView, QGraphicsScene, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PIL.ImageQt import ImageQt


class MVTecDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = conv_block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = conv_block(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = conv_block(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        mid = self.middle(self.pool(enc4))

        dec1 = self.decoder1(torch.cat([enc4, self.up1(mid)], 1))
        dec2 = self.decoder2(torch.cat([enc3, self.up2(dec1)], 1))
        dec3 = self.decoder3(torch.cat([enc2, self.up3(dec2)], 1))
        dec4 = self.decoder4(torch.cat([enc1, self.up4(dec3)], 1))

        return self.final(dec4)

class AnomalyDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Window properties
        self.setWindowTitle('Anomaly Detection')
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout(central_widget)

        # Image display
        self.image_view = QGraphicsView()
        self.image_scene = QGraphicsScene()
        self.image_view.setScene(self.image_scene)
        layout.addWidget(self.image_view)

        # Image import button
        self.import_button = QPushButton('Import Images')
        self.import_button.clicked.connect(self.import_images)
        layout.addWidget(self.import_button)

        # Train button
        self.train_button = QPushButton('Train Model')
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Inference button
        self.inference_button = QPushButton('Run Inference')
        self.inference_button.clicked.connect(self.run_inference)
        layout.addWidget(self.inference_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        
    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(3, 3).to(device) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 20
        # Training loop
        for epoch in range(num_epochs):
            for batch_idx, images in enumerate(self.train_loader):
                images = images.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, images)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                self.progress_bar.setValue(int((batch_idx + 1) * 100 / len(self.train_loader)))

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        torch.save(model.state_dict(), 'unet.pth')
        print('Model saved successfully!')


    def run_inference(self):
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(3, 3).to(device)
        model.load_state_dict(torch.load('unet.pth'))

        # Load test image
        test_image_path, _ = QFileDialog.getOpenFileName(self, "Select Test Image", os.path.expanduser('~'), "Images (*.png *.jpg *.bmp)")
        test_image = Image.open(test_image_path).convert('RGB')

        # Preprocess test image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_image_tensor = transform(test_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(test_image_tensor)


        # Postprocess output
        input_image_np = np.array(test_image.resize((256, 256))).astype(np.uint8)
        output_image = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)

        # Calculate the absolute difference and find the maximum
        diff_image = cv2.absdiff(input_image_np, output_image)
        anomaly_scores = np.mean(diff_image, axis=2)
        max_y, max_x = np.unravel_index(np.argmax(anomaly_scores), anomaly_scores.shape)

        # Draw a red circle around the area with the highest anomaly score
        radius = 20
        cv2.circle(input_image_np, (max_x, max_y), radius, (255, 0, 0), 2)

        # Display test image with the red circle
        self.image_scene.clear()
        qim_input = QImage(input_image_np.data, input_image_np.shape[1], input_image_np.shape[0], input_image_np.strides[0], QImage.Format_RGB888)
        self.image_scene.addPixmap(QPixmap.fromImage(qim_input))
        self.image_view.setScene(self.image_scene)



    def import_images(self):
        image_paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", os.path.expanduser('~'), "Images (*.png *.jpg *.bmp)")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = MVTecDataset(image_paths, transform=transform)
        self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=4)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = AnomalyDetectionApp()
    mainWin.show()
    sys.exit(app.exec_())
