import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from handwritten_text_recognition.scripts.dataloader import CharacterDataset
from tqdm import tqdm


class TransformerModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_classes, num_heads=8, num_layers=3, dropout=0.1
    ):
        super(TransformerModel, self).__init__()

        # Initial feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 32, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Reshape input: B x C x H x W -> B x (H*W) x C
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)  # B x C x (H*W)
        x = x.permute(0, 2, 1)  # B x (H*W) x C

        # Feature extraction
        x = self.feature_extraction(x)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        x = self.fc_out(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train the transformer model
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (i + 1)})

        print(
            f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}"
        )


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    train_dir = os.path.join(project_dir, "dataset", "train")
    model_path = os.path.join(project_dir, "models", "transformer_model.pth")

    # Create models directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    try:
        # Create dataset and loader
        train_dataset = CharacterDataset(train_dir, transform=transform)
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )

        num_classes = len(train_dataset.classes)
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_dataset)}")

        # Initialize model
        model = TransformerModel(
            input_dim=1024, hidden_dim=256, num_classes=num_classes  # 32x32 images
        ).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Train model
        print("\nStarting training...")
        train_model(model, train_loader, criterion, optimizer, device)

        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ Model saved to {model_path}")

    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()
