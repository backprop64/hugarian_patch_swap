import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

def load_center_crop(path: str, out_h: int, out_w: int, device: torch.device) -> torch.Tensor:
    """
    Returns float32 torch Tensor in [0,1], shape (H, W, 3) on `device`.
    """
    img = Image.open(path).convert("RGB")  # (H0,W0,3)
    W0, H0 = img.size
    # scale so both dims >= target; then center-crop
    s = max(out_w / W0, out_h / H0)
    new_w, new_h = int(np.ceil(W0 * s)), int(np.ceil(H0 * s))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - out_w) // 2
    top  = (new_h - out_h) // 2
    img = img.crop((left, top, left + out_w, top + out_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) float32
    t = torch.from_numpy(arr).to(device)             # (H,W,3) float32
    return t


def adjust_patch_size(h: int, ph: int) -> int:
    """Smallest integer >= ph that divides h."""
    if h % ph == 0:
        return ph
    p = ph
    while h % p != 0:
        p += 1
    return p


def img_to_patches(img: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """
    img: (H, W, C) float32
    returns patches: (N, ph, pw, C)
    where N = (H//ph)*(W//pw)
    """
    H, W, C = img.shape
    assert H % ph == 0 and W % pw == 0
    nh, nw = H // ph, W // pw
    x = img.view(nh, ph, nw, pw, C).permute(0, 2, 1, 3, 4).reshape(nh * nw, ph, pw, C)
    return x  # (N, ph, pw, C)


def patches_to_img(patches: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    patches: (N, ph, pw, C)
    returns image: (H, W, C)
    """
    N, ph, pw, C = patches.shape
    nh, nw = H // ph, W // pw
    x = patches.view(nh, nw, ph, pw, C).permute(0, 2, 1, 3, 4).reshape(H, W, C)
    return x


# ------------------------- feature extraction -------------------------

SOBEL_X = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32)
SOBEL_Y = torch.tensor([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]], dtype=torch.float32)

GRAY_W = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)  # (3,)


def conv_valid_3x3(batch_gray: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    batch_gray: (N, H, W) float32
    kernel: (3,3) float32
    returns (N, H-2, W-2)
    Uses torch.conv2d with padding=0, stride=1.
    """
    N, H, W = batch_gray.shape
    x = batch_gray.unsqueeze(1)                     # (N,1,H,W)
    k = kernel.view(1, 1, 3, 3).to(batch_gray)     # (1,1,3,3)
    y = F.conv2d(x, k, stride=1, padding=0)        # (N,1,H-2,W-2)
    return y.squeeze(1)                             # (N,H-2,W-2)


def extract_features(img: torch.Tensor, ph: int, pw: int, feature_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    img: (H,W,3) float32
    Returns (F, patches)
      F: (N, D) where
        - greyscale: D = ph*pw
        - xy_grad : D = 2*(ph-2)*(pw-2)
      patches: (N, ph, pw, 3) float32 (for reconstruction)
    """
    patches = img_to_patches(img, ph, pw)                                  # (N, ph, pw, 3)
    if feature_type == "greyscale":
        w = GRAY_W.to(img.device).view(1, 1, 1, 3)                          # (1,1,1,3)
        gray = (patches * w).sum(dim=-1)                                    # (N, ph, pw)
        Fv = gray.view(gray.shape[0], -1)                                   # (N, ph*pw)
        return Fv, patches
    elif feature_type == "xy_grad":
        if ph < 3 or pw < 3:
            raise ValueError("xy_grad requires patch_h and patch_w >= 3 for valid 3x3 conv.")
        w = GRAY_W.to(img.device).view(1, 1, 1, 3)
        gray = (patches * w).sum(dim=-1)                                    # (N, ph, pw)
        dx = conv_valid_3x3(gray, SOBEL_X.to(img.device))                   # (N, ph-2, pw-2)
        dy = conv_valid_3x3(gray, SOBEL_Y.to(img.device))                   # (N, ph-2, pw-2)
        Fv = torch.cat([dx.view(dx.shape[0], -1),
                         dy.view(dy.shape[0], -1)], dim=1)                   # (N, 2*(ph-2)*(pw-2))
        return Fv, patches
    else:
        raise ValueError("feature_type must be 'greyscale' or 'xy_grad'.")


# ------------------------- distances & assignment -------------------------

def pairwise_cost(A: torch.Tensor, B: torch.Tensor, metric: str) -> torch.Tensor:
    """
    A: (N, D), B: (N, D) -> cost matrix C: (N, N)
    metric in {'l2','cosine'}
    """
    if metric == "l2":
        a2 = (A * A).sum(dim=1, keepdim=True)                   # (N,1)
        b2 = (B * B).sum(dim=1, keepdim=True).transpose(0, 1)   # (1,N)
        C = a2 + b2 - 2.0 * (A @ B.transpose(0, 1))             # (N,N)
        C.clamp_(min=0)                                         # numerical floor
        return C
    elif metric == "cosine":
        eps = 1e-8
        A_n = A / (A.norm(dim=1, keepdim=True) + eps)           # (N,D)
        B_n = B / (B.norm(dim=1, keepdim=True) + eps)           # (N,D)
        S = A_n @ B_n.transpose(0, 1)                           # (N,N) cosine sim
        C = 1.0 - S                                             # cost = 1 - cos
        
        return C
    else:
        raise ValueError("comparison_metric must be 'l2' or 'cosine'.")


def solve_assignment(C: torch.Tensor):
    """
    C: (N,N) torch -> row_ind, col_ind (both (N,)) numpy
    Uses SciPy Hungarian on CPU.
    """
    C_np = C.detach().to("cpu").numpy()
    row_ind, col_ind = linear_sum_assignment(C_np)
    return row_ind, col_ind


# ------------------------- reconstruction -------------------------

def reconstruct_from_style(style_img: torch.Tensor, content_img: torch.Tensor,
                           ph: int, pw: int,
                           row_ind, col_ind) -> torch.Tensor:
    """
    style_img, content_img: (H,W,3) float32 on same device
    row_ind, col_ind: numpy arrays mapping rows(A=content patches) -> cols(B=style patches)
    Returns stylized content image: (H,W,3) float32
    """
    device = content_img.device
    H, W, _ = content_img.shape
    style_p = img_to_patches(style_img, ph, pw)                        # (N, ph, pw, 3)
    N = style_p.shape[0]
    mapping = torch.empty(N, dtype=torch.long, device=device)          # (N,)
    mapping[torch.from_numpy(np.asarray(row_ind)).to(device)] = torch.from_numpy(np.asarray(col_ind)).to(device)
    chosen = style_p[mapping]                                          # (N, ph, pw, 3)
    out = patches_to_img(chosen, H, W)                                  # (H, W, 3)
    return out


# ------------------------- main pipeline -------------------------

def run_one(style_path: str, content_path: str,
            oh: int, ow: int,
            ph: int, pw: int,
            feature_type: str, metric: str,
            device: torch.device,
            out_path: str = None) -> torch.Tensor:
    # 1) Load & crop
    A = load_center_crop(style_path,  oh, ow, device)                  # (H,W,3) float32 in [0,1]
    B = load_center_crop(content_path, oh, ow, device)                  # (H,W,3) float32 in [0,1]
    H, W, _ = A.shape

    # 2) Patch-size adjustment 
    ph_adj = adjust_patch_size(H, ph)
    pw_adj = adjust_patch_size(W, pw)
    if (ph_adj != ph) or (pw_adj != pw):
        print(f"[info] adjusted patch size to ({ph_adj},{pw_adj}) to evenly tile ({H},{W})")
    ph, pw = ph_adj, pw_adj

    # 3) Extract features per image
    F_style, _   = extract_features(A, ph, pw, feature_type)           # A: (N,D)
    F_content, _ = extract_features(B, ph, pw, feature_type)           # B: (N,D)
    # Shapes:
    #   A,B: (H,W,3); F_*: (N,D); with
    #   - greyscale: D=ph*pw
    #   - xy_grad:  D=2*(ph-2)*(pw-2)

    # 4) Build cost (GPU/CPU) & solve assignment (CPU)
    C = pairwise_cost(F_content, F_style, metric)                      # (N,N)
    row_ind, col_ind = solve_assignment(C)                             # (N,), (N,)

    # 5) Reconstruct stylized content (device)
    out = reconstruct_from_style(A, B, ph, pw, row_ind, col_ind)       # (H,W,3)

    # 6) Save
    if out_path is not None:
        out_img = (out.clamp(0, 1) * 255.0).to(torch.uint8).to("cpu").numpy()
        Image.fromarray(out_img).save(out_path)
    return out


def save_triptych(content_img: torch.Tensor, style_img: torch.Tensor,
                  stylized_img: torch.Tensor, out_path: str):
    """
    each input: (H,W,3) float32 in [0,1]; saves a 3-column PNG
    """
    def to_uint8(x: torch.Tensor):
        return (x.clamp(0, 1) * 255.0).to(torch.uint8).to("cpu").numpy()
    panel = torch.cat([content_img, style_img, stylized_img], dim=1)  # (H, 3W, 3)
    Image.fromarray(to_uint8(panel)).save(out_path)
   

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Patch-based style transfer via optimal assignment")
    role = p.add_mutually_exclusive_group(required=True)
    role.add_argument("--style", type=str, help="Path to style image (when used with --content).")
    role.add_argument("--img1", type=str, help="Path to image 1 (testing mode; creates two triptychs).")
    p.add_argument("--content", type=str, help="Path to content image (when used with --style).")
    p.add_argument("--img2", type=str, help="Path to image 2 (testing mode; creates two triptychs).")

    p.add_argument("--img_height", type=int, required=True)
    p.add_argument("--img_width",  type=int, required=True)
    p.add_argument("--patch_h",    type=int, required=True)
    p.add_argument("--patch_w",    type=int, required=True)
    p.add_argument("--feature_type", choices=["greyscale", "xy_grad"], required=True)
    p.add_argument("--comparison_metric", choices=["l2", "cosine"], required=True)
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda (if available)")

    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[error] CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Mode A: explicit roles (save only stylized content)
    if args.style and args.content:
        out_name = f"stylized_{Path(args.content).stem}_from_{Path(args.style).stem}_{args.feature_type}_{args.comparison_metric}_{args.patch_h}x{args.patch_w}.png"
        out_path = out_dir / out_name
        _ = run_one(
            args.style, args.content,
            args.img_height, args.img_width,
            args.patch_h, args.patch_w,
            args.feature_type, args.comparison_metric,
            device,
            out_path=str(out_path)
        )
        print(f"[ok] saved {out_path}")
        

    # Mode B: testing — two directions, two 3-column images
    if args.img1 and args.img2:
        # img1 as content, img2 as style
        out1 = run_one(
            args.img2, args.img1,  # style, content
            args.img_height, args.img_width,
            args.patch_h, args.patch_w,
            args.feature_type, args.comparison_metric,
            device,
            out_path=None
        )
        A1 = load_center_crop(args.img2, args.img_height, args.img_width, device)  # style
        B1 = load_center_crop(args.img1, args.img_height, args.img_width, device)  # content
        name1 = f"triptych_{Path(args.img1).stem}_content_{Path(args.img2).stem}_style_{args.feature_type}_{args.comparison_metric}_{args.patch_h}x{args.patch_w}.png"
        save_triptych(B1, A1, out1, str(out_dir / name1))
        print(f"[ok] saved {out_dir / name1}")

        # img2 as content, img1 as style
        out2 = run_one(
            args.img1, args.img2,  # style, content
            args.img_height, args.img_width,
            args.patch_h, args.patch_w,
            args.feature_type, args.comparison_metric,
            device,
            out_path=None
        )
        A2 = load_center_crop(args.img1, args.img_height, args.img_width, device)  # style
        B2 = load_center_crop(args.img2, args.img_height, args.img_width, device)  # content
        name2 = f"triptych_{Path(args.img2).stem}_content_{Path(args.img1).stem}_style_{args.feature_type}_{args.comparison_metric}_{args.patch_h}x{args.patch_w}.png"
        save_triptych(B2, A2, out2, str(out_dir / name2))
        print(f"[ok] saved {out_dir / name2}")
        