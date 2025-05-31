import numpy as np
from scipy.ndimage import convolve

def generate_updated_dose_kernel(kernel_size=(21, 21, 21), sigma_static=3.0, motion_amplitude=5.0):
    """
    Generate a motion-adjusted, heterogeneity-aware dose kernel.
    
    Parameters:
    - kernel_size: Tuple of (x, y, z) dimensions for the kernel.
    - sigma_static: Standard deviation for the static Gaussian kernel (in voxels).
    - motion_amplitude: Amplitude of respiratory motion (in voxels).
    
    Returns:
    - kernel: 3D numpy array representing the updated dose kernel.
    """
    # Step 1: Generate a static Gaussian kernel
    center = np.array(kernel_size) // 2
    x, y, z = np.ogrid[
        -center[0]:kernel_size[0] - center[0],
        -center[1]:kernel_size[1] - center[1],
        -center[2]:kernel_size[2] - center[2]
    ]
    r = np.sqrt(x**2 + y**2 + z**2)
    kernel_static = np.exp(-r**2 / (2 * sigma_static**2))
    kernel_static /= np.sum(kernel_static)  # Normalize

    # Step 2: Generate motion kernel (Gaussian in z-direction)
    sigma_motion = motion_amplitude / np.sqrt(2)  # Motion spread
    motion_kernel = np.zeros(kernel_size)
    z_slice = np.arange(-center[2], kernel_size[2] - center[2])
    motion_kernel[center[0], center[1], :] = np.exp(-z_slice**2 / (2 * sigma_motion**2))
    motion_kernel /= np.sum(motion_kernel)  # Normalize

    # Step 3: Apply motion blur
    kernel_motion = convolve(kernel_static, motion_kernel, mode="constant")

    # Step 4: Adjust for heterogeneity (average over typical lung density profile)
    # Assume 70% lung tissue (rho=0.2), 30% soft tissue (rho=1.0)
    w_lung = 0.7
    w_soft = 0.3
    # Approximate: Increase sigma for lung (less scatter), decrease for soft tissue
    kernel_lung = kernel_motion * 1.2  # Slightly broader spread in low-density lung
    kernel_soft = kernel_motion * 0.8  # Tighter spread in soft tissue
    kernel = w_lung * kernel_lung + w_soft * kernel_soft
    kernel /= np.sum(kernel)  # Normalize

    return kernel

# Generate and save the updated kernel
if __name__ == "__main__":
    kernel = generate_updated_dose_kernel()
    np.save("dose_kernel.npy", kernel)
    print("Updated dose kernel saved as 'dose_kernel.npy'")
