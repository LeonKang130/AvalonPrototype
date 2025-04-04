from model import *
from util import save_exr_image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import psutil, os

def main():
    pretraining_data = torch.load("pretraining-dataset.pt")
    dataset = TensorDataset(pretraining_data[..., :-3].clone().detach().cuda(), pretraining_data[..., -3:].clone().detach().cuda())
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    model = Lancelot().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()
    verification_data = torch.load("verification-dataset.pt").cuda()
    sphere_resolution = 32
    for i in range(32):
        epoch_loss = 0
        for xs, ys in tqdm(dataloader):
            optimizer.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {i}: {epoch_loss / len(dataloader)}")
        if i % 4 == 3:
            with torch.no_grad():
                prediction = model(verification_data[..., :-3])
                loss = criterion(prediction, verification_data[..., -3:])
                print(f"Verification loss: {loss.item()}")
                prediction = torch.exp_(prediction) - 1
                save_exr_image(prediction.reshape(sphere_resolution ** 2, sphere_resolution ** 2, 3), f"verification-{i // 4}.exr")

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    main()