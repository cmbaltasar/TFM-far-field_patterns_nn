import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class DipoleRNN(nn.Module):
    """
    LSTM-based neural network for estimating dipole parameters from radiation patterns.
    Takes sequences of E-field data and outputs orientation angles and position coordinates.
    """
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM to process radiation pattern sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)

        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Take only the last timestep output
        final_output = lstm_out[:, -1, :]
        output = self.fc(final_output)
        
        # Scale outputs to appropriate ranges
        theta_d = torch.sigmoid(output[:, 0]) * np.pi  # [0, π]
        phi_d = torch.tanh(output[:, 1]) * np.pi       # [-π, π]
        pos_x = torch.tanh(output[:, 2]) * 2.0         # [-2λ, 2λ]
        pos_y = torch.tanh(output[:, 3]) * 2.0         
        pos_z = torch.tanh(output[:, 4]) * 2.0         
        
        return torch.stack([theta_d, phi_d, pos_x, pos_y, pos_z], dim=1)


def hertzian_dipole_field_with_position(theta_d, phi_d, pos_x, pos_y, pos_z, theta_obs, phi_obs, k=1.0, r=1.0):
    """
    Computes E-field components for a Hertzian dipole at specified position and orientation.
    Uses far-field approximation with phase correction for dipole displacement.
    """
    # Convert inputs to tensors if needed
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

    # Spherical unit vectors at observation point
    # theta_hat components
    theta_hat_x = torch.cos(theta_obs) * torch.cos(phi_obs)
    theta_hat_y = torch.cos(theta_obs) * torch.sin(phi_obs)
    theta_hat_z = -torch.sin(theta_obs)

    # phi_hat components
    phi_hat_x = -torch.sin(phi_obs)
    phi_hat_y = torch.cos(phi_obs)
    phi_hat_z = torch.zeros_like(phi_obs)

    # Dot products for field components
    p_dot_theta = p_x * theta_hat_x + p_y * theta_hat_y + p_z * theta_hat_z
    p_dot_phi = p_x * phi_hat_x + p_y * phi_hat_y + p_z * phi_hat_z

    # Phase shift calculation (far-field approximation)
    r_hat_x = torch.sin(theta_obs) * torch.cos(phi_obs)
    r_hat_y = torch.sin(theta_obs) * torch.sin(phi_obs)
    r_hat_z = torch.cos(theta_obs)
    
    # Position-dependent phase shift
    phase_shift = k * (pos_x * r_hat_x + pos_y * r_hat_y + pos_z * r_hat_z)
    
    # Total phase factor
    kr = k * r
    phase_factor = torch.complex(torch.cos(-kr + phase_shift), torch.sin(-kr + phase_shift)) / r
    amplitude = k**2 / (4 * np.pi)

    E_theta = amplitude * phase_factor * p_dot_theta.to(torch.complex64)
    E_phi = amplitude * phase_factor * p_dot_phi.to(torch.complex64)
    
    return E_theta, E_phi


def generate_pattern_from_dipole_with_position(theta_d, phi_d, pos_x, pos_y, pos_z, N_points=100):
    """
    Creates synthetic radiation pattern data by sampling dipole fields over angular grid.
    Returns tensor with observation angles and complex E-field components.
    """
    n_side = int(np.sqrt(N_points))
    theta_obs = torch.linspace(0, np.pi, n_side)
    phi_obs = torch.linspace(0, 2*np.pi, n_side)
    theta_grid, phi_grid = torch.meshgrid(theta_obs, phi_obs, indexing='ij')
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()

    E_theta, E_phi = hertzian_dipole_field_with_position(theta_d, phi_d, pos_x, pos_y, pos_z, theta_flat, phi_flat)

    # Pack everything into training format
    pattern_data = torch.stack([
        theta_flat,
        phi_flat,
        E_theta.real,
        E_theta.imag,
        E_phi.real,
        E_phi.imag
    ], dim=1).float()

    return pattern_data


def prepare_rnn_sequence(pattern_data, sequence_length=100):
    """
    Formats pattern data into fixed-length sequences for RNN input.
    Randomly samples or pads data to match required sequence length.
    """
    N_points = pattern_data.shape[0]
    if N_points >= sequence_length:
        # Random sampling if we have enough points
        indices = torch.randperm(N_points)[:sequence_length]
        sequence = pattern_data[indices]
    else:
        # Pad with zeros if insufficient data
        padding = torch.zeros(sequence_length - N_points, pattern_data.shape[1])
        sequence = torch.cat([pattern_data, padding], dim=0)
    
    return sequence.unsqueeze(0)  # Add batch dimension


def orientation_and_position_loss(pred_params, true_params):
    """
    Combined loss function for dipole parameter estimation.
    Handles angular wrapping for phi and includes position error terms.
    """
    pred_theta_d, pred_phi_d = pred_params[:, 0], pred_params[:, 1]
    true_theta_d, true_phi_d = true_params[:, 0], true_params[:, 1]
    
    pred_pos = pred_params[:, 2:5]
    true_pos = true_params[:, 2:5]
    
    # Angle losses
    loss_theta = F.mse_loss(pred_theta_d, true_theta_d)
    # Handle phi wraparound properly
    delta_phi = torch.atan2(torch.sin(pred_phi_d - true_phi_d), torch.cos(pred_phi_d - true_phi_d))
    loss_phi = torch.mean(delta_phi**2)
    
    # Position loss
    loss_pos = F.mse_loss(pred_pos, true_pos)
    
    return loss_theta + loss_phi + loss_pos


def visualize_results_with_position(losses, pattern_data, final_params, theta_vals, phi_vals):
    """
    Creates plots showing training progress, radiation pattern and final parameter estimates.
    Useful for debugging and validating the training process.
    """
    plt.figure(figsize=(15, 5))

    # Training loss evolution
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Radiation pattern visualization
    plt.subplot(1, 3, 2)
    E_theta_real = pattern_data[:, 2].numpy()
    E_theta_imag = pattern_data[:, 3].numpy()
    E_phi_real = pattern_data[:, 4].numpy()
    E_phi_imag = pattern_data[:, 5].numpy()
    E_mag = np.sqrt(E_theta_real**2 + E_theta_imag**2 + E_phi_real**2 + E_phi_imag**2)
    
    sc = plt.scatter(theta_vals.numpy()*180/np.pi, phi_vals.numpy()*180/np.pi, c=E_mag, cmap='viridis', s=10)
    plt.colorbar(sc, label='|E| Magnitude')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Phi (degrees)')
    plt.title('Radiation Pattern Magnitude')

    # Final parameter estimates
    plt.subplot(1, 3, 3)
    param_names = ['θ (°)', 'φ (°)', 'x (λ)', 'y (λ)', 'z (λ)']
    final_values = final_params[0].detach().numpy()
    final_values[0] *= 180/np.pi  # Convert to degrees
    final_values[1] *= 180/np.pi
    
    plt.bar(param_names, final_values)
    plt.title('Estimated Parameters')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def train_dipole_estimator_with_position(true_theta_d, true_phi_d, true_pos_x, true_pos_y, true_pos_z, n_epochs=1000, sequence_length=100):
    """
    Main training loop for the dipole parameter estimation network.
    Generates synthetic data, trains the model and displays results.
    """
    print("="*70)
    print("Training dipole orientation and position estimator")
    print(f"True orientation: θ = {float(true_theta_d)*180/np.pi:.1f}°, φ = {float(true_phi_d)*180/np.pi:.1f}°")
    print(f"True position: x = {float(true_pos_x):.2f}λ, y = {float(true_pos_y):.2f}λ, z = {float(true_pos_z):.2f}λ")
    print("="*70)

    # Setup model and optimizer
    model = DipoleRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

    # Generate training data
    pattern_data = generate_pattern_from_dipole_with_position(true_theta_d, true_phi_d, true_pos_x, true_pos_y, true_pos_z, N_points=sequence_length*2)
    rnn_input = prepare_rnn_sequence(pattern_data, sequence_length=sequence_length)

    true_params = torch.tensor([[true_theta_d, true_phi_d, true_pos_x, true_pos_y, true_pos_z]], dtype=torch.float32)

    # Training loop
    losses = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_params = model(rnn_input)
        loss = orientation_and_position_loss(pred_params, true_params)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Print progress
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            pred_values = pred_params[0].detach()
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f} | "
                  f"θ = {pred_values[0].item()*180/np.pi:.1f}°, φ = {pred_values[1].item()*180/np.pi:.1f}°, "
                  f"x = {pred_values[2].item():.2f}λ, y = {pred_values[3].item():.2f}λ, z = {pred_values[4].item():.2f}λ")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_params = model(rnn_input)

    print(f"\nFinal estimated parameters:")
    final_values = final_params[0].detach()
    print(f"θ = {final_values[0].item()*180/np.pi:.1f}°, φ = {final_values[1].item()*180/np.pi:.1f}°")
    print(f"x = {final_values[2].item():.2f}λ, y = {final_values[3].item():.2f}λ, z = {final_values[4].item():.2f}λ")

    # Show results
    visualize_results_with_position(losses, pattern_data, final_params, pattern_data[:, 0], pattern_data[:, 1])
    
    return model, final_params, pattern_data


# Example usage and testing
if __name__ == "__main__":
    # Set up test case
    true_theta = torch.tensor(np.pi/4)   # 45 degrees
    true_phi = torch.tensor(np.pi/3)     # 60 degrees
    true_pos_x = torch.tensor(0.25)      # quarter wavelength in x
    true_pos_y = torch.tensor(0.0)       # centered in y
    true_pos_z = torch.tensor(0.1)       # small offset in z

    # Run training
    model, est_params, pattern = train_dipole_estimator_with_position(true_theta, true_phi, true_pos_x, true_pos_y, true_pos_z)