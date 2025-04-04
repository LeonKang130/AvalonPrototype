import torch.jit

from model import *
from util import save_exr_image
from torch.utils.data import DataLoader, TensorDataset

def main():
    pretraining_data = torch.load("pretraining-dataset.pt")
    dataset = TensorDataset(pretraining_data[..., :-3].clone().detach().cuda(), pretraining_data[..., -3:].clone().detach().cuda())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = torch.jit.script(Lancelot()).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    for i in range(8):
        epoch_loss = 0
        for xs, ys in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {i}: {epoch_loss / len(dataloader)}")
    verification_data = torch.load("verification-dataset.pt").cuda()
    sphere_resolution = 32
    with torch.no_grad():
        prediction = model(verification_data)
        prediction = torch.exp_(prediction) - 1
        save_exr_image(prediction.reshape(sphere_resolution ** 2, sphere_resolution ** 2, 3), "verification.exr")


if __name__ == "__main__":
    main()