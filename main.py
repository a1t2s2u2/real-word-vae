import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# Dataset to capture images from the camera
class CameraDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.cap = cv2.VideoCapture(0)  # Open the default camera

    def __len__(self):
        return 1000  # Arbitrary length for demonstration

    def __getitem__(self, idx):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")

        # Convert BGR to RGB and apply transformations
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64, 64))
        if self.transform:
            frame = self.transform(frame)

        return frame

    def release(self):
        self.cap.release()

# Define the CNN-based VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128 * 8 * 8, 128)
        self.fc_logvar = nn.Linear(128 * 8 * 8, 128)
        
        # Decoder
        self.fc_decode = nn.Linear(128, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoding
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)

        return x, mu, logvar

# Loss function for VAE
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# Save input and output images to the output folder
def save_and_show_images(input_image, output_image, step, update_interval=0.1):
    os.makedirs("output", exist_ok=True)
    
    # detach()を使用して勾配計算を無効化し、CPUに移動してからNumPy配列に変換
    input_image_np = (input_image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    output_image_np = (output_image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # 保存
    input_path = os.path.join("output/camera", f"input_{step}.png")
    output_path = os.path.join("output/vae", f"output_{step}.png")
    cv2.imwrite(input_path, cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path, cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR))

    # 入力画像と出力画像を表示
    window_size = 400  # ウィンドウサイズ（正方形の一辺のピクセル数）
    cv2.namedWindow("Input Image (Camera)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Input Image (Camera)", window_size, window_size)
    cv2.imshow("Input Image (Camera)", cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))

    cv2.namedWindow("Reconstructed Image (VAE Output)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconstructed Image (VAE Output)", window_size, window_size)
    cv2.imshow("Reconstructed Image (VAE Output)", cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR))

    # 指定した間隔で更新
    key = cv2.waitKey(int(update_interval * 1000))
    if key == ord('q'):  # 'q'を押すと終了
        return True  # 終了フラグを返す

    return False  # 継続フラグを返す




# Main training loop
def train_vae(update_interval=0.1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CameraDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step = 0
    for epoch in range(10):  # Train for 10 epochs
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)  # データをデバイスに送る

            # Forward pass
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # Save input and reconstructed images, and show them in real-time
            if save_and_show_images(batch[0], recon_batch[0], step, update_interval):
                dataset.release()
                cv2.destroyAllWindows()
                return  # ユーザーが'q'を押したら終了

            step += 1

    dataset.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    train_vae()
