import torch
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from torchvision import transforms

def extract_and_save_features(data_dir, checkpoint_path='./dinov3_vitl16.pth', target_size=1536):
    """
    Extracts DINOv3 features for all images in data_dir/image 
    and saves them as .pt files in data_dir/dinov3_features.
    """
    print(f"\n[DINOv3] Starting Feature Extraction...")
    print(f"[DINOv3] Data Source: {data_dir}")
    print(f"[DINOv3] Checkpoint: {checkpoint_path}")
    
    # Input/Output Setup
    img_dir = os.path.join(data_dir, "image")
    out_dir_pt = os.path.join(data_dir, "dinov3_features")
    os.makedirs(out_dir_pt, exist_ok=True)
    
    # Find images
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(images) == 0:
        print("[DINOv3] Error: No images found to process.")
        return

    # Check which ones are already done
    existing_pts = [f for f in os.listdir(out_dir_pt) if f.endswith('.pt')]
    if len(existing_pts) == len(images):
        print("[DINOv3] All features already exist. Skipping extraction.")
        return

    # Load Model
    print("[DINOv3] Loading model weights...")
    try:
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', pretrained=False, trusted=True)
        
        # Checkpoint path logic
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join('..', checkpoint_path) 
            
        if not os.path.exists(checkpoint_path):
            print(f"[DINOv3] Error: Weights not found at {checkpoint_path}")
            return

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict: state_dict = state_dict['model']
        elif 'teacher' in state_dict: state_dict = state_dict['teacher']
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()} 
        
        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        model.eval()
        print("[DINOv3] Model loaded successfully.")
    except Exception as e:
        print(f"[DINOv3] Critical Error loading model: {e}")
        return

    # Preprocessing Pipeline (High Res for quality)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((target_size, target_size), antialias=True), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print(f"[DINOv3] Processing {len(images)} images at resolution {target_size}x{target_size}...")
    
    with torch.no_grad():
        for img_name in tqdm(images):
            save_path = os.path.join(out_dir_pt, os.path.splitext(img_name)[0] + '.pt')
            if os.path.exists(save_path): continue

            img_p = os.path.join(img_dir, img_name)
            img = cv2.imread(img_p)
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(torch.from_numpy(img_rgb).permute(2,0,1)).unsqueeze(0).cuda()
            
            features_dict = model.forward_features(input_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] 
            
            N_tokens = patch_tokens.shape[1]
            grid_h = int(np.sqrt(N_tokens))
            grid_w = int(np.sqrt(N_tokens))
            
            feats_tensor = patch_tokens.squeeze(0).permute(1, 0).reshape(1024, grid_h, grid_w)
            torch.save(feats_tensor.half().cpu(), save_path)

    print(f"[DINOv3] Extraction complete. Features saved to {out_dir_pt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='./dinov3_vitl16.pth')
    args = parser.parse_args()
    extract_and_save_features(args.data_dir, args.ckpt)