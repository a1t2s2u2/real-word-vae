import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import logging
from contextlib import contextmanager

# 定数定義
IMAGE_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@contextmanager
def camera_capture(device_index=0):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        logging.error("カメラをオープンできませんでした。")
        raise RuntimeError("Failed to open camera")
    try:
        yield cap
    finally:
        cap.release()

def get_frame(retries=3, device_index=0):
    for attempt in range(retries):
        try:
            with camera_capture(device_index) as cap:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                    return frame
                else:
                    logging.warning(f"カメラからの画像取得に失敗しました。再試行: {attempt+1}/{retries}")
        except Exception as e:
            logging.warning(f"カメラキャプチャ例外: {e}。再試行: {attempt+1}/{retries}")
    raise RuntimeError("Failed to capture image from camera after several attempts")

class CameraDataset(Dataset):
    def __init__(self, transform=None, length=1000):
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frame = get_frame()
        if self.transform:
            frame = self.transform(frame)
        return frame

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

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def save_and_show_images(input_image, output_image, step, update_interval=0.1):
    os.makedirs("output/camera", exist_ok=True)
    os.makedirs("output/vae", exist_ok=True)
    
    input_image_np = (input_image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    output_image_np = (output_image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    input_path = os.path.join("output/camera", f"input_{step}.png")
    output_path = os.path.join("output/vae", f"output_{step}.png")
    cv2.imwrite(input_path, cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path, cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR))

    window_size = 400
    cv2.namedWindow("Input Image (Camera)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Input Image (Camera)", window_size, window_size)
    cv2.imshow("Input Image (Camera)", cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))

    cv2.namedWindow("Reconstructed Image (VAE Output)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconstructed Image (VAE Output)", window_size, window_size)
    cv2.imshow("Reconstructed Image (VAE Output)", cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(int(update_interval * 1000))
    if key == ord('q'):
        return True
    return False

def train_vae(update_interval=0.1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CameraDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info(f"Using device: {DEVICE}")

    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step = 0
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            batch = batch.to(DEVICE).unsqueeze(0)  # バッチサイズが1の場合のshape調整
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            if save_and_show_images(batch[0], recon_batch[0], step, update_interval):
                cv2.destroyAllWindows()
                return

            step += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_vae()
