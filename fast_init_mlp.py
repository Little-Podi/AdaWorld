import torch
import torch.nn as nn
import torch.optim as optim


class ActionMLP(nn.Module):
    def __init__(self, num_actions: int = 4, action_dim: int = 32):
        super(ActionMLP, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(num_actions, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 2
        z = self.mapping(x)
        return z


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def train_action_mlp(model, data_loader, num_epochs=3000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize total loss for the epoch
        num_batches = 0  # Initialize batch counter

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        average_loss = epoch_loss / num_batches

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
            save_checkpoint(model, optimizer, epoch + 1, average_loss)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_actions = 2  # nuScenes [x, y] displacement
    action_dim = 32

    inputs = torch.load("path_to/nusc_action_labels.pt").float()
    targets = torch.load("path_to/latent_action_stats.pt").float()

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    model = ActionMLP(num_actions=num_actions, action_dim=action_dim).to(device)

    train_action_mlp(model, data_loader)
