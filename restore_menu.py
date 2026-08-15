"""
SEMICON KLA Hackathon - Single File Restoration Menu
With Enhanced Restoration Quality (Percentile Stretching)
"""
import torch
import numpy as np
import cv2
import os
import sys

# Ensure model is loadable from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now load the model
from models.model import RestoreNet

print("="*50)
print("SEMICON KLA IMAGE RESTORER - ENHANCED QUALITY")
print("="*50)
print("Loading model...")
model = RestoreNet()
checkpoint = torch.load('models/checkpoint_epoch300.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ Model loaded!\n")

def restore_image(input_path, output_path):
    """
    Restores image using percentile-based stretching for best visual quality.
    Outputs: restored .png + comparison .png (degraded | restored)
    """
    # Force output to always be .png extension
    output_path = output_path if output_path.endswith('.png') else output_path + '.png'
    
    # Handle .npy files (KLA dataset format)
    if input_path.endswith('.npy'):
        data = np.load(input_path)
        # Convert to uint8 grayscale
        if data.ndim == 2:
            if data.max() <= 1.0:
                img = (data * 255).astype(np.uint8)
            else:
                img = data.astype(np.uint8)
        else:
            img = data.astype(np.uint8)
    else:
        # Regular image files (.png, .jpg)
        if not os.path.exists(input_path):
            print(f"✗ Error: Cannot find '{input_path}'")
            return False
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"✗ Error: Cannot read image '{input_path}'")
            return False
    
    original_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"📁 Input: {original_name}")
    orig_h, orig_w = img.shape
    print(f"   Size: {orig_w}x{orig_h}")
    
    # Resize to 128x128 (model input)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Run model inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
        output = model(input_tensor)
    
    # ---- ENHANCED RESTORATION: Percentile-based stretching ----
    # This adapts to the actual output range and produces best visual quality
    result_raw = output.squeeze().numpy()
    
    # Method: Percentile p2-p98 stretching (recommended for visual quality)
    p2, p98 = np.percentile(result_raw, (2, 98))
    if p98 > p2:
        result_scaled = (result_raw - p2) / (p98 - p2) * 255
    else:
        result_scaled = (result_raw - result_raw.min()) / (result_raw.max() - result_raw.min()) * 255
    
    result_img = np.clip(result_scaled, 0, 255).astype(np.uint8)
    
    # Resize back to original dimensions
    result_resized = cv2.resize(result_img, (orig_w, orig_h))
    
    # Save restored output as .png
    cv2.imwrite(output_path, result_resized)
    
    # Create comparison: left = enhanced degraded, right = restored
    # Enhance the degraded for visibility using CLAHE
    if img.max() <= 1.0:
        lr_display = ((img * 255) + 0.001).astype(np.uint8)
    else:
        img = cv2.convertScaleAbs(img)
        lr_display = img
    
    # Resize both to 256x256 for comparison display
    lr_display_256 = cv2.resize(lr_display, (256, 256))
    result_256 = cv2.resize(result_resized, (256, 256))
    
    # Apply CLAHE to degraded for visibility
    if lr_display_256.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lr_enhanced = clahe.apply(lr_display_256)
    else:
        lr_enhanced = cv2.cvtColor(lr_display_256, cv2.COLOR_GRAY2BGR)
    
    # Resize enhanced to exactly match
    lr_enhanced_256 = cv2.resize(lr_enhanced, (256, 256))
    result_256 = cv2.resize(result_resized, (256, 256))
    
    # Create side-by-side comparison
    comparison = np.hstack([lr_enhanced_256, result_256])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "DEGRADED INPUT", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "AI RESTORED", (256 + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # Add value ranges
    lr_min, lr_max = float(img.min()), float(img.max())
    result_min, result_max = float(result_raw.min()), float(result_raw.max())
    cv2.putText(comparison, f"Input: {lr_min:.3f} to {lr_max:.3f}", (10, 50), font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, f"Output: {result_min:.1f} to {result_max:.1f}", (256 + 10, 50), font, 0.5, (255, 255, 255), 1)
    
    # Save comparison - always .png
    comp_path = output_path.replace('.png', '_comparison.png')
    cv2.imwrite(comp_path, comparison)
    
    print(f"✓ Saved restored: {output_path}")
    print(f"✓ Saved comparison: {comp_path}\n")
    return True

def single_restore():
    """Option 1: Single file restoration"""
    print("\n=== SINGLE IMAGE RESTORATION ===")
    inp = input("Enter image path (or drag & drop file): ").strip('"').strip("'")
    out = input("Output restored PNG name (e.g., result.png): ").strip('"')
    if not out.endswith('.png'):
        out += '.png'
    
    if not os.path.exists(inp):
        print(f"✗ File not found: {inp}")
        return
    
    print("Processing...")
    restore_image(inp, out)
    
    print("\n✅ Done!")
    print(f"   Restored image: {out}")
    print(f"   Comparison image: {out.replace('.png', '_comparison.png')}")

def batch_restore():
    """Option 2: Batch restore folder"""
    print("\n=== BATCH RESTORATION ===")
    inp = input("Folder path with images to restore: ").strip('"')
    out = input("Output folder for restored images: ").strip('"')
    
    if not os.path.isdir(inp):
        print(f"✗ Directory not found: {inp}")
        return
    
    os.makedirs(out, exist_ok=True)
    print(f"📁 Input folder: {inp}")
    print(f"📁 Output folder: {out}")
    
    supported = ['.png', '.jpg', '.jpeg', '.npy', '.tif', '.bmp']
    files = [f for f in os.listdir(inp) if os.path.splitext(f)[1].lower() in supported]
    
    if not files:
        print(f"✗ No supported images found in {inp}")
        return
    
    print(f"📦 Found {len(files)} images to restore")
    
    for filename in files:
        input_path = os.path.join(inp, filename)
        # Force .png extension for output
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(out, f"restored_{base_name}.png")
        restore_image(input_path, output_path)
    
    print(f"\n✅ Done! {len(files)} images restored to {out}/")

# ==========================================
# MAIN MENU
# ==========================================

if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Single image restoration")
        print("2. Batch restore folder")
        print("3. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            single_restore()
        elif choice == "2":
            batch_restore()
        elif choice == "3":
            print("\nGoodbye! 👋")
            break
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")