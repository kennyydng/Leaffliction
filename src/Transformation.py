import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

def load_image(image_path):
    """Load image and convert to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image_path}")
    return img

def apply_gaussian_blur(image, ksize=(5, 5)):
    """Apply Gaussian blur to reduce noise."""
    return pcv.gaussian_blur(img=image, ksize=ksize)

def create_mask(image):
    """Create binary mask to isolate leaf from background."""
    s = pcv.rgb2gray_hsv(rgb_img=image, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, object_type='light')
    mask = pcv.fill(bin_img=s_thresh, size=50)
    mask = pcv.fill_holes(bin_img=mask)
    return mask

def extract_roi(image, mask):
    """Extract ROI by applying mask (black background)."""
    return pcv.apply_mask(img=image, mask=mask, mask_color='black')

def analyze_object(image, mask):
    """Analyze morphological properties of the leaf."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    cnt = max(contours, key=cv2.contourArea)
    analyzed_img = image.copy()
    
    # Draw contour
    cv2.drawContours(analyzed_img, [cnt], -1, (0, 255, 0), 2)
    
    # Calculate properties
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Draw bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(analyzed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Draw fitted ellipse
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(analyzed_img, ellipse, (255, 255, 0), 2)
    
    # Add annotations
    y_offset = y - 10
    cv2.putText(analyzed_img, f"Area: {area:.0f}px²", (x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    y_offset -= 20
    cv2.putText(analyzed_img, f"Perimeter: {perimeter:.0f}px", (x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    y_offset -= 20
    cv2.putText(analyzed_img, f"Circularity: {circularity:.3f}", (x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return analyzed_img

def extract_pseudolandmarks(image, mask, n_landmarks=100):
    """Extract characteristic points along the leaf contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    landmark_img = image.copy()
    
    if not contours:
        return landmark_img
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Draw center
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(landmark_img, (cx, cy), 5, (0, 255, 255), -1)
    
    # Extract equidistant points along contour
    perimeter = cv2.arcLength(cnt, True)
    distance = perimeter / n_landmarks
    current_distance = 0
    
    for i in range(len(cnt) - 1):
        p1 = cnt[i][0]
        p2 = cnt[i + 1][0]
        d = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        current_distance += d
        
        if current_distance >= distance:
            color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
            cv2.circle(landmark_img, tuple(p1), 3, color, -1)
            current_distance = 0
    
    return landmark_img

def display_transformations(image_path):
    """Display and save all transformations."""
    # Load and process image
    original = load_image(image_path)
    blurred = apply_gaussian_blur(original)
    mask = create_mask(original)
    roi_img = extract_roi(original, mask)
    analyzed_img = analyze_object(original, mask)
    landmarks_img = extract_pseudolandmarks(original, mask)
    
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Leaf Analysis with PlantCV', fontsize=16)
    
    # Display images
    axs[0, 0].imshow(original)
    axs[0, 0].set_title('Original Image')
    
    axs[0, 1].imshow(blurred)
    axs[0, 1].set_title('Gaussian Blur')
    
    axs[0, 2].imshow(mask, cmap='gray')
    axs[0, 2].set_title('Binary Mask')
    
    axs[1, 0].imshow(roi_img)
    axs[1, 0].set_title('ROI Objects')
    
    axs[1, 1].imshow(analyzed_img)
    axs[1, 1].set_title('Object Analysis')
    
    axs[1, 2].imshow(landmarks_img)
    axs[1, 2].set_title('Pseudolandmarks')
    
    # Remove axes
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save and show
    plt.tight_layout()
    
    # Créer le dossier output à la racine du projet (parent de src/)
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "all_transformations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined image saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Path relatif à la racine du projet (parent de src/)
    image_path = Path(__file__).parent.parent / "input/Apple/Apple_Black_rot/image (1).JPG"
    display_transformations(str(image_path))
