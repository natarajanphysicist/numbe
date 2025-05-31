import numpy as np
import matplotlib.pyplot as plt
from tumor_detection_enhanced import detect_tumors

def test_tumor_detection():
    # Create a sample CT slice with a simulated tumor
    image_size = (256, 256)
    ct_slice = np.zeros(image_size)
    
    # Add background variation
    ct_slice += np.random.normal(0.2, 0.05, image_size)
    
    # Add a simulated tumor
    center = (128, 128)
    radius = 20
    y, x = np.ogrid[-center[0]:image_size[0]-center[0], -center[1]:image_size[1]-center[1]]
    mask = x*x + y*y <= radius*radius
    ct_slice[mask] = 0.8 + np.random.normal(0, 0.1, size=np.sum(mask))
    
    # Detect tumors
    results = detect_tumors(ct_slice)
    
    if results is None:
        print("Error: Tumor detection failed")
        return
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(ct_slice, cmap='gray')
    ax1.set_title("Input CT Slice")
    
    # Detection results
    ax2.imshow(ct_slice, cmap='gray')
    ax2.imshow(results['mask'], alpha=0.3, cmap='Reds')
    ax2.set_title(f"Detected Tumor\nConfidence: {results['confidence']:.2f}")
    
    plt.show()
    
    # Print tumor properties
    print("\nTumor Properties:")
    for i, props in enumerate(results['tumor_properties'], 1):
        print(f"\nTumor {i}:")
        print(f"Center: {props['centroid']}")
        print(f"Area: {props['area_mm2']:.1f} mm²")
        print(f"Volume: {props['volume']:.1f} mm³")
        print(f"Eccentricity: {props['eccentricity']:.2f}")
        print(f"Solidity: {props['solidity']:.2f}")

if __name__ == "__main__":
    test_tumor_detection()
