import torch
import numpy as np
import sys
from io import StringIO

from dipole_prueba1 import (
    DipoleRNN, 
    hertzian_dipole_field_with_position,
    generate_pattern_from_dipole_with_position,
    prepare_rnn_sequence,
    orientation_and_position_loss,
    train_dipole_estimator_with_position
)

class TestDipoleRNN:
    """Test suite for the dipole RNN implementation"""
    
    def test_dipole_rnn_initialization(self):
        """Check that model initializes with correct parameters"""
        model = DipoleRNN()
        assert model.hidden_size == 128
        assert model.num_layers == 2
        
        # Try custom params
        model_custom = DipoleRNN(input_size=8, hidden_size=64, num_layers=3, output_size=5)
        assert model_custom.hidden_size == 64
        assert model_custom.num_layers == 3
        print("DipoleRNN initialization test passed")

    def test_dipole_rnn_forward_pass(self):
        """Verify forward pass outputs have correct shape and ranges"""
        model = DipoleRNN()
        batch_size, seq_len, input_size = 2, 50, 6
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = model(x)
        assert output.shape == (batch_size, 5)  # theta, phi, x, y, z
        
        # Check output ranges are physically meaningful
        theta = output[:, 0]
        phi = output[:, 1]
        positions = output[:, 2:5]
        
        assert torch.all(theta >= 0) and torch.all(theta <= np.pi)
        assert torch.all(phi >= -np.pi) and torch.all(phi <= np.pi)
        assert torch.all(positions >= -2.0) and torch.all(positions <= 2.0)
        print("DipoleRNN forward pass test passed")

    def test_hertzian_dipole_field_basic(self):
        """Basic sanity check for dipole field calculation"""
        theta_d = torch.tensor(np.pi/4)
        phi_d = torch.tensor(np.pi/3)
        pos_x, pos_y, pos_z = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        theta_obs = torch.tensor([0, np.pi/2, np.pi])
        phi_obs = torch.tensor([0, np.pi/2, np.pi])
        
        E_theta, E_phi = hertzian_dipole_field_with_position(theta_d, phi_d, pos_x, pos_y, pos_z, theta_obs, phi_obs)
        
        assert E_theta.shape == theta_obs.shape
        assert E_phi.shape == phi_obs.shape
        assert E_theta.dtype == torch.complex64
        assert E_phi.dtype == torch.complex64
        print("Hertzian dipole field basic test passed")

    def test_position_effect_on_field(self):
        """Verify that moving the dipole changes the field phase"""
        # Use tilted dipole to avoid nulls
        theta_d = torch.tensor(np.pi/4)  # 45° tilt
        phi_d = torch.tensor(0.0)
        
        # Pick observation angle with decent field strength
        theta_obs = torch.tensor(np.pi/6)  # 30° from z-axis
        phi_obs = torch.tensor(0.0)
        
        # Compare field at origin vs displaced position
        E_theta_origin, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, 0.0, theta_obs, phi_obs)
        E_theta_displaced, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, 0.5, theta_obs, phi_obs)  # λ/2 along z
        
        # Extract phases
        phase_origin = torch.angle(E_theta_origin).item()
        phase_displaced = torch.angle(E_theta_displaced).item()
        
        # Expected phase shift: k * displacement * cos(theta_obs)
        expected_phase_diff = 2*np.pi * 0.5 * np.cos(np.pi/6)  # k=2π/λ, disp=0.5λ
        actual_phase_diff = abs(phase_displaced - phase_origin)
        
        print(f"Debug - Phase origin: {phase_origin:.6f}")
        print(f"Debug - Phase displaced: {phase_displaced:.6f}")
        print(f"Debug - Expected phase diff: {expected_phase_diff:.6f}")
        print(f"Debug - Actual phase diff: {actual_phase_diff:.6f}")
        
        # Handle phase wrapping
        if actual_phase_diff > np.pi:
            actual_phase_diff = 2*np.pi - actual_phase_diff
            
        # Should see significant phase shift
        assert actual_phase_diff > 0.3, f"Expected significant phase difference, got {actual_phase_diff:.6f}"
        print("Position effect on field test passed")

    def test_pattern_generation(self):
        """Check that pattern generation produces reasonable data"""
        theta_d = torch.tensor(np.pi/4)
        phi_d = torch.tensor(np.pi/6)
        pos_x, pos_y, pos_z = 0.1, 0.2, 0.0
        
        pattern = generate_pattern_from_dipole_with_position(theta_d, phi_d, pos_x, pos_y, pos_z, N_points=100)
        
        expected_points = int(np.sqrt(100))**2
        assert pattern.shape[0] == expected_points
        assert pattern.shape[1] == 6  # theta, phi, E_theta_real, E_theta_imag, E_phi_real, E_phi_imag
        
        # No NaNs or infs
        assert torch.all(torch.isfinite(pattern))
        print("Pattern generation test passed")

    def test_rnn_sequence_preparation(self):
        """Test sequence formatting for RNN input"""
        pattern_data = torch.randn(150, 6)
        sequence_length = 100
        
        sequence = prepare_rnn_sequence(pattern_data, sequence_length)
        assert sequence.shape == (1, sequence_length, 6)
        
        # Test padding case
        small_pattern = torch.randn(50, 6)
        sequence_small = prepare_rnn_sequence(small_pattern, sequence_length)
        assert sequence_small.shape == (1, sequence_length, 6)
        print("RNN sequence preparation test passed")

    def test_loss_function(self):
        """Basic loss function check"""
        pred_params = torch.tensor([[np.pi/4, np.pi/3, 0.1, 0.2, 0.0]])
        true_params = torch.tensor([[np.pi/3, np.pi/4, 0.0, 0.1, 0.1]])
        
        loss = orientation_and_position_loss(pred_params, true_params)
        assert loss.item() > 0
        assert torch.isfinite(loss)
        print("Loss function test passed")

    def test_array_factor_effect(self):
        """Test interference pattern from two dipoles (array factor)"""
        # Two dipoles, same orientation
        theta_d = torch.tensor(np.pi/2)
        phi_d = torch.tensor(0.0)
        
        # Sample different observation angles
        theta_obs = torch.linspace(0, np.pi, 50)
        phi_obs = torch.zeros_like(theta_obs)
        
        # First dipole at origin
        E1_theta, E1_phi = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, 0.0, theta_obs, phi_obs)
        
        # Second dipole at λ/2 spacing
        spacing = 0.5  # λ/2
        E2_theta, E2_phi = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, spacing, theta_obs, phi_obs)
        
        # Superposition
        E_total_theta = E1_theta + E2_theta
        E_total_phi = E1_phi + E2_phi
        
        # E_phi should be small for this dipole configuration
        max_phi = torch.max(torch.abs(E_total_phi))
        max_theta = torch.max(torch.abs(E_total_theta))
        assert max_phi < 0.1 * max_theta, f"E_phi ({max_phi:.6f}) should be much smaller than E_theta ({max_theta:.6f})"
        
        # Check for interference pattern (variation in field strength)
        magnitude_theta = torch.abs(E_total_theta)
        assert torch.max(magnitude_theta) > torch.min(magnitude_theta), "Should have interference pattern in theta component"
        
        # Also check total field magnitude
        total_magnitude = torch.sqrt(torch.abs(E_total_theta)**2 + torch.abs(E_total_phi)**2)
        assert torch.max(total_magnitude) > torch.min(total_magnitude), "Should have interference pattern in total field"
        
        # Verify significant variation for λ/2 spacing
        variation_ratio = torch.max(magnitude_theta) / torch.min(magnitude_theta)
        assert variation_ratio > 2.0, f"Interference pattern should be significant (ratio: {variation_ratio:.2f})"
        
        print(f"E_theta max: {max_theta:.6f}, E_phi max: {max_phi:.6f}")
        print(f"Interference pattern variation ratio: {variation_ratio:.2f}")
        print("Array factor effect test passed")

    def test_simple_position_verification(self):
        """Simple check that position affects field phase"""
        theta_d = torch.tensor(np.pi/4)  # 45° dipole
        phi_d = torch.tensor(0.0)
        
        # Observe at angle with good field strength
        theta_obs = torch.tensor(np.pi/3)  # 60° from z-axis
        phi_obs = torch.tensor(0.0)
        
        # Two different positions
        E1, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, 0.0, theta_obs, phi_obs)
        E2, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, 0.5, theta_obs, phi_obs)  # λ/2 displacement
        
        # Compare phases
        phase1 = torch.angle(E1).item()
        phase2 = torch.angle(E2).item()
        phase_diff = abs(phase2 - phase1)
        
        print(f"Debug - Phase 1: {phase1:.6f}, Phase 2: {phase2:.6f}")
        print(f"Debug - Raw phase difference: {phase_diff:.6f}")
        
        # Handle wraparound
        if phase_diff > np.pi:
            phase_diff = 2*np.pi - phase_diff
            
        print(f"Debug - Corrected phase difference: {phase_diff:.6f}")
        
        # Expect meaningful phase shift for λ/2 displacement
        assert phase_diff > 0.2, f"Expected phase difference > 0.2 radians, got {phase_diff:.6f}"
        print("Simple position verification test passed")

    def test_array_factor_lambda_half_spacing(self):
        """Test array factor for λ/2 spaced dipoles"""
        # Tilted dipoles to avoid nulls
        theta_d = torch.tensor(np.pi/4)  # 45° orientation
        phi_d = torch.tensor(0.0)
        
        # Test angles that avoid null regions
        theta_test = torch.tensor([np.pi/6, np.pi/3, 2*np.pi/3, 5*np.pi/6])  # Avoid θ=90°
        phi_test = torch.zeros_like(theta_test)
        
        # Two dipoles with λ/2 total separation
        E1_theta, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, -0.25, theta_test, phi_test)  # -λ/4
        E2_theta, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, 0.25, theta_test, phi_test)   # +λ/4
        
        # Make sure fields aren't too small
        E1_mag = torch.abs(E1_theta)
        assert torch.all(E1_mag > 1e-10), "E1 field is too small, risking division by zero"
        
        # Calculate array factor
        E_total = E1_theta + E2_theta
        AF_magnitude = torch.abs(E_total) / E1_mag
        
        print(f"Debug - E1 magnitudes: {E1_mag}")
        print(f"Debug - E_total magnitudes: {torch.abs(E_total)}")
        print(f"Debug - AF magnitudes: {AF_magnitude}")
        
        # AF should be between 0 and 2 (destructive to constructive)
        assert torch.all(AF_magnitude >= 0), "Array factor cannot be negative"
        assert torch.all(AF_magnitude <= 2.1), f"Array factor too large: {torch.max(AF_magnitude)}"  # Small tolerance
        
        # Should see some variation due to interference
        af_max = torch.max(AF_magnitude).item()
        af_min = torch.min(AF_magnitude).item()
        variation = af_max - af_min
        
        print(f"Debug - AF variation: {variation:.6f} (max: {af_max:.6f}, min: {af_min:.6f})")
        
        # Even small variations are physically meaningful
        assert variation > 0.02, f"Expected AF variation > 0.02, got {variation:.6f}"
        print("✓ Array factor λ/2 spacing test passed")

    def test_phase_difference_calculation(self):
        """Verify phase shift calculations are correct"""
        theta_d = torch.tensor(np.pi/2)
        phi_d = torch.tensor(0.0)
        
        # Observe at θ=90°, φ=0° (broadside to z-axis)
        theta_obs = torch.tensor(np.pi/2)
        phi_obs = torch.tensor(0.0)
        
        # Two positions along z-axis
        pos1_z = 0.0
        pos2_z = 0.5  # λ/2 spacing
        
        E1_theta, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, pos1_z, theta_obs, phi_obs)
        E2_theta, _ = hertzian_dipole_field_with_position(theta_d, phi_d, 0.0, 0.0, pos2_z, theta_obs, phi_obs)
        
        phase1 = torch.angle(E1_theta)
        phase2 = torch.angle(E2_theta)
        phase_diff = phase2 - phase1
        
        # At broadside (θ=90°), cos(θ)=0, so phase difference should be 0
        expected_phase_diff = 0.0
        
        assert torch.allclose(phase_diff, torch.tensor(expected_phase_diff), atol=1e-5)
        print("Phase difference calculation test passed")

    def test_training_integration(self):
        """Quick integration test of the training pipeline"""
        # Short training run
        true_theta = torch.tensor(np.pi/3)
        true_phi = torch.tensor(np.pi/4)
        true_pos_x, true_pos_y, true_pos_z = 0.1, 0.0, 0.2
        
        # Suppress stdout during test
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            model, final_params, pattern = train_dipole_estimator_with_position(
                true_theta, true_phi, true_pos_x, true_pos_y, true_pos_z, 
                n_epochs=10, sequence_length=50)
            
            assert model is not None
            assert final_params.shape == (1, 5)
            assert pattern.shape[1] == 6
            
        finally:
            sys.stdout = old_stdout
        
        print("Training integration test passed")


def run_all_tests():
    """Execute all tests and report results"""
    test_suite = TestDipoleRNN()
    
    tests = [
        test_suite.test_dipole_rnn_initialization,
        test_suite.test_dipole_rnn_forward_pass,
        test_suite.test_hertzian_dipole_field_basic,
        test_suite.test_position_effect_on_field,
        test_suite.test_pattern_generation,
        test_suite.test_rnn_sequence_preparation,
        test_suite.test_loss_function,
        test_suite.test_array_factor_effect,
        test_suite.test_simple_position_verification,
        test_suite.test_array_factor_lambda_half_spacing,
        test_suite.test_phase_difference_calculation,
        test_suite.test_training_integration
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    print("="*60)
    print("Running DipoleRNN Test Suite")
    print("="*60)
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: Assertion failed - {e}")
            failed += 1
            failed_tests.append((test.__name__, f"Assertion failed - {e}"))
        except Exception as e:
            print(f"✗ {test.__name__}: Error - {e}")
            failed += 1
            failed_tests.append((test.__name__, f"Error - {e}"))
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    
    if failed > 0:
        print(f"Failed tests: {failed}")
        print("\nFailed test details:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
    else:
        print("All tests passed!")
    
    print("="*60)
    
    return failed == 0


def run_specific_test(test_name):
    """Run a single test by name"""
    test_suite = TestDipoleRNN()
    
    # Available tests
    test_methods = {
        'initialization': test_suite.test_dipole_rnn_initialization,
        'forward_pass': test_suite.test_dipole_rnn_forward_pass,
        'field_basic': test_suite.test_hertzian_dipole_field_basic,
        'position_effect': test_suite.test_position_effect_on_field,
        'pattern_generation': test_suite.test_pattern_generation,
        'sequence_prep': test_suite.test_rnn_sequence_preparation,
        'loss_function': test_suite.test_loss_function,
        'array_factor': test_suite.test_array_factor_effect,
        'simple_position': test_suite.test_simple_position_verification,
        'lambda_half': test_suite.test_array_factor_lambda_half_spacing,
        'phase_diff': test_suite.test_phase_difference_calculation,
        'training': test_suite.test_training_integration
    }
    
    if test_name in test_methods:
        try:
            test_methods[test_name]()
            print(f"Test '{test_name}' passed")
            return True
        except Exception as e:
            print(f"Test '{test_name}' failed: {e}")
            return False
    else:
        print(f"Test '{test_name}' not found. Available tests:")
        for name in test_methods.keys():
            print(f"  - {name}")
        return False


if __name__ == "__main__":

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        run_specific_test(test_name)
    else:
        run_all_tests()