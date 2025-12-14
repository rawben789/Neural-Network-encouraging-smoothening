# Implementing Custom Loss Function
# Obj: 1. Fitting the noisy training data. 
#      2. Encouraging the neural network’s predictions to be ”smooth”.
# Data Generation
# Generate a noisy dataset from the ground truth function y = sin(x) on the domain x ∈ [0,2π].
# Generate N = 40 training points uniformly distributed in [0,2π].
# Add Gaussian noise: y_noisy = sin(x) + ϵ where ϵ ∼ N(0,0.2).

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. CONFIGURATION
class Config:
    N_SAMPLES = 40
    NOISE_STD = 0.2
    EPOCHS = 3000
    LR = 0.01
    # Lambda controls the trade-off:
    # Higher = Smoother line, Lower = Fits noise
    SMOOTHNESS_WEIGHT = 0.01

# 2. DATA GENERATION
def generate_data():
    """Generates noisy sin(x) data as described in the challenge."""
    X = np.linspace(0, 2 * np.pi, Config.N_SAMPLES)
    y_true = np.sin(X)
    noise = np.random.normal(0, Config.NOISE_STD, size=X.shape)
    y_noisy = y_true + noise

    # Convert Numpy Arrays to PyTorch tensors (N, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor, X, y_true

# 3. MODEL
class SmoothNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard MLP: 1 -> 64 -> 64 -> 1
        # Tanh activation is crucial for calculating higher-order derivatives
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 4. CUSTOM LOSS (The Core Task)
def smoothness_loss(model, x_input, y_target, lambda_reg):
    """
    Calculates: MSE + lambda * ||d^2y/dx^2||^2
    We penalize the 'curvature' of the function to prevent it from
    wiggling to fit the noise.
    """
    x_input.requires_grad = True
    y_pred = model(x_input)

    # 1. Standard MSE Loss (Fit the data)
    mse_loss = nn.MSELoss()(y_pred, y_target)

    # 2. Smoothness Regularizer (Penalize Curvature)
    # First derivative (slope)
    grads = torch.autograd.grad(y_pred, x_input,
                                grad_outputs=torch.ones_like(y_pred),
                                create_graph=True)[0]

    # Second derivative (curvature)
    grad_2 = torch.autograd.grad(grads, x_input,
                                 grad_outputs=torch.ones_like(grads),
                                 create_graph=True)[0]

    # We want to minimize the magnitude of the curvature
    smoothness_penalty = torch.mean(grad_2 ** 2)

    total_loss = mse_loss + (lambda_reg * smoothness_penalty)
    return total_loss, mse_loss.item(), smoothness_penalty.item()

# 5. TRAINING
def train():
    X_train, y_train, X_np, y_true_np = generate_data()
    model = SmoothNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR) # Adam Optimizer (reliable)
    print(f"Training on {Config.N_SAMPLES} points with Smoothness Weight = {Config.SMOOTHNESS_WEIGHT}")
    loss_history = []
    for epoch in range(Config.EPOCHS):
        optimizer.zero_grad()

        # Calculate custom loss
        loss, mse, penalty = smoothness_loss(model, X_train, y_train, Config.SMOOTHNESS_WEIGHT)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Total: {loss.item():.5f} | MSE: {mse:.5f} | Smoothness: {penalty:.5f}")
    return model, X_train, y_train, X_np, y_true_np

# 6. VISUALIZATION
def visualize_results(model, X_train, y_train, X_raw, y_true_raw):
    # Create a dense grid for plotting the learned curve
    X_dense = torch.linspace(0, 2*np.pi, 200).view(-1, 1)
    with torch.no_grad():
        y_pred = model(X_dense).numpy()
    X_dense_np = X_dense.numpy()
    plt.figure(figsize=(10, 6))

    # 1. The Ground Truth
    plt.plot(X_raw, y_true_raw, 'k--', label='Ground Truth (sin x)', alpha=0.6)

    # 2. The Noisy Training Data
    plt.scatter(X_train.detach().numpy(), y_train.detach().numpy(),
                color='red', s=30, alpha=0.7, label='Noisy Observations')

    # 3. The Model Prediction
    plt.plot(X_dense_np, y_pred, 'b-', linewidth=2.5, label='Model (Reinforced Smoothing)')
    plt.title(f'Reinforced Smoothing Result\nSmoothness Penalty $\lambda={Config.SMOOTHNESS_WEIGHT}$', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = "smoothing_result.png"
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to {filename}")
    plt.show()
    from IPython.display import Image
    Image(filename="smoothing_result.png")

if __name__ == "__main__":
    model, X_t, y_t, X_r, y_r = train()
    visualize_results(model, X_t, y_t, X_r, y_r)
    