import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# DipoleRNN model definition
class DipoleRNN(nn.Module):
    """
    Defines an LSTM-based neural network that estimates dipole orientation angles 
    (theta and phi) from input sequences representing radiation patterns.
    The network outputs theta in [0, π] and phi in [-π, π].
    """
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.1 if num_layers > 1 else 0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Performs forward pass through LSTM and fully connected layers,
        producing estimated dipole orientation angles scaled to their physical ranges.
        """
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        orientation = self.fc(final_output)
        theta_d = torch.sigmoid(orientation[:, 0]) * np.pi  # scale to [0, π]
        phi_d = torch.tanh(orientation[:, 1]) * np.pi       # scale to [-π, π]
        return torch.stack([theta_d, phi_d], dim=1)

# Hertzian dipole electric field calculation
def hertzian_dipole_field(theta_d, phi_d, theta_obs, phi_obs, k=1.0, r=1.0):
    """
    Calculates the electric field components (E_theta and E_phi) of a Hertzian dipole
    at observation angles (theta_obs, phi_obs), given the dipole orientation (theta_d, phi_d).
    Returns complex field components representing the far-field pattern.
    """
    if not isinstance(theta_d, torch.Tensor):
        theta_d = torch.tensor(theta_d, dtype=torch.float32)
    if not isinstance(phi_d, torch.Tensor):
        phi_d = torch.tensor(phi_d, dtype=torch.float32)
    if not isinstance(theta_obs, torch.Tensor):
        theta_obs = torch.tensor(theta_obs, dtype=torch.float32)
    if not isinstance(phi_obs, torch.Tensor):
        phi_obs = torch.tensor(phi_obs, dtype=torch.float32)
    k = torch.tensor(k, dtype=torch.float32)
    r = torch.tensor(r, dtype=torch.float32)

    # Dipole moment components
    p_x = torch.sin(theta_d) * torch.cos(phi_d)
    p_y = torch.sin(theta_d) * torch.sin(phi_d)
    p_z = torch.cos(theta_d)

    # Unit vectors in observation direction (theta and phi)
    theta_hat_x = torch.cos(theta_obs) * torch.cos(phi_obs)
    theta_hat_y = torch.cos(theta_obs) * torch.sin(phi_obs)
    theta_hat_z = -torch.sin(theta_obs)

    phi_hat_x = -torch.sin(phi_obs)
    phi_hat_y = torch.cos(phi_obs)
    phi_hat_z = torch.zeros_like(phi_obs)

    # Dot products of dipole moment with unit vectors
    p_dot_theta = p_x * theta_hat_x + p_y * theta_hat_y + p_z * theta_hat_z
    p_dot_phi = p_x * phi_hat_x + p_y * phi_hat_y + p_z * phi_hat_z

    kr = k * r
    phase_factor = torch.complex(torch.cos(-kr), torch.sin(-kr)) / r
    amplitude = k**2 / (4 * np.pi)

    E_theta = amplitude * phase_factor * p_dot_theta.to(torch.complex64)
    E_phi = amplitude * phase_factor * p_dot_phi.to(torch.complex64)
    return E_theta, E_phi

# Generate theoretical radiation pattern from dipole orientation
def generate_pattern_from_dipole(theta_d, phi_d, N_points=100):
    """
    Generates a synthetic far-field radiation pattern dataset by sampling
    the Hertzian dipole electric field over a grid of spherical angles.
    Returns a tensor containing theta, phi, and complex field components.
    """
    n_side = int(np.sqrt(N_points))
    theta_obs = torch.linspace(0, np.pi, n_side)
    phi_obs = torch.linspace(0, 2*np.pi, n_side)
    theta_grid, phi_grid = torch.meshgrid(theta_obs, phi_obs, indexing='ij')
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()

    E_theta, E_phi = hertzian_dipole_field(theta_d, phi_d, theta_flat, phi_flat)

    pattern_data = torch.stack([
        theta_flat,
        phi_flat,
        E_theta.real,
        E_theta.imag,
        E_phi.real,
        E_phi.imag
    ], dim=1).float()

    return pattern_data

# Prepare input sequence for RNN model
def prepare_rnn_sequence(pattern_data, sequence_length=100):
    """
    Prepares a fixed-length input sequence for the RNN by randomly sampling points
    from the pattern data or padding if insufficient points.
    Returns a tensor shaped for batch-first RNN input.
    """
    N_points = pattern_data.shape[0]
    if N_points >= sequence_length:
        indices = torch.randperm(N_points)[:sequence_length]
        sequence = pattern_data[indices]
    else:
        padding = torch.zeros(sequence_length - N_points, pattern_data.shape[1])
        sequence = torch.cat([pattern_data, padding], dim=0)
    return sequence.unsqueeze(0)

# Loss function comparing predicted and true dipole orientations
def orientation_loss(pred_theta_d, pred_phi_d, true_theta_d, true_phi_d):
    """
    Computes a combined mean squared error loss for dipole orientation angles,
    accounting for circular nature of phi using angular difference.
    """
    loss_theta = F.mse_loss(pred_theta_d, true_theta_d)
    delta_phi = torch.atan2(torch.sin(pred_phi_d - true_phi_d), torch.cos(pred_phi_d - true_phi_d))
    loss_phi = torch.mean(delta_phi**2)
    return loss_theta + loss_phi

# Visualize training loss and radiation pattern only (no dipole orientation plots)
def visualize_results(losses, pattern_data, final_orientation, theta_vals, phi_vals):
    """
    Plots the training loss and radiation pattern magnitude. 
    Dipole orientation visualizations (2D/3D) removed as requested.
    """
    plt.figure(figsize=(12,5))

    # Plot training loss curve
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot radiation pattern magnitude
    plt.subplot(1,2,2)
    E_theta_real = pattern_data[:,2].numpy()
    E_theta_imag = pattern_data[:,3].numpy()
    E_phi_real = pattern_data[:,4].numpy()
    E_phi_imag = pattern_data[:,5].numpy()
    E_mag = np.sqrt(E_theta_real**2 + E_theta_imag**2 + E_phi_real**2 + E_phi_imag**2)
    
    sc = plt.scatter(theta_vals.numpy()*180/np.pi, phi_vals.numpy()*180/np.pi, c=E_mag, cmap='viridis', s=10)
    plt.colorbar(sc, label='|E| Magnitude')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Phi (degrees)')
    plt.title('Radiation Pattern Magnitude')

    plt.tight_layout()
    plt.show()

# Main training function to estimate dipole orientation from pattern
def train_dipole_orientation_estimator(true_theta_d, true_phi_d, n_epochs=1000, sequence_length=100):
    """
    Trains the DipoleRNN model to estimate dipole orientation given a theoretical
    radiation pattern generated from true orientation angles.
    Prints training progress and returns the trained model, final orientation, and pattern data.
    """
    print("="*60)
    print("Training dipole orientation estimator from theoretical pattern")
    print(f"True orientation: θ = {float(true_theta_d)*180/np.pi:.1f}°, φ = {float(true_phi_d)*180/np.pi:.1f}°")
    print("="*60)

    model = DipoleRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

    pattern_data = generate_pattern_from_dipole(true_theta_d, true_phi_d, N_points=sequence_length*2)
    rnn_input = prepare_rnn_sequence(pattern_data, sequence_length=sequence_length)

    losses = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        orientation = model(rnn_input)
        pred_theta_d = orientation[0, 0]
        pred_phi_d = orientation[0, 1]

        loss = orientation_loss(pred_theta_d, pred_phi_d, true_theta_d, true_phi_d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f} | θ_pred = {pred_theta_d.item()*180/np.pi:.1f}°, φ_pred = {pred_phi_d.item()*180/np.pi:.1f}°")

    model.eval()
    with torch.no_grad():
        final_orientation = model(rnn_input)
        est_theta = final_orientation[0, 0].item() * 180 / np.pi
        est_phi = final_orientation[0, 1].item() * 180 / np.pi

    print(f"\nFinal estimated orientation: θ = {est_theta:.1f}°, φ = {est_phi:.1f}°")

    visualize_results(losses, pattern_data, final_orientation, pattern_data[:,0], pattern_data[:,1])
    
    return model, final_orientation, pattern_data

if __name__ == "__main__":
    true_theta = torch.tensor(np.pi/4)   # 45 degrees
    true_phi = torch.tensor(np.pi/3)     # 60 degrees

    model, est_orientation, pattern = train_dipole_orientation_estimator(true_theta, true_phi)
