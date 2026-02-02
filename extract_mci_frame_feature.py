import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import mobileclip
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

"""
Usage

* Single video
python extract_mci_frame_feature.py --video input.mp4 --output_dir ./features

* Directory of videos
python extract_mci_frame_feature.py --video_dir ./videos --output_dir ./features

* Custom sampling rate and model
python extract_mci_frame_feature.py --video input.mp4 --output_dir ./features \
    --sampling_rate 6 --model_name MobileCLIP2-S0 --batch_size 64
"""

class FrameFeatureExtractor:
    def __init__(self, model_name="mobileclip_s0", pretrained_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        is_mobileclip_v2 = model_name.startswith("MobileCLIP2-")

        if is_mobileclip_v2:
            print(f"Loading MobileCLIP2 model: {model_name}")
            model_kwargs = {}
            if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
                model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_path if pretrained_path else "dfndr2b",
                **model_kwargs
            )
            self.model.eval()
            self.model = reparameterize_model(self.model)
            self.model = self.model.to(self.device)
        else:
            print(f"Loading MobileCLIP v1 model: {model_name}")
            if pretrained_path is None and model_name == "mobileclip_s0":
                pretrained_path = os.path.expanduser(
                    "~/.cache/huggingface/hub/models--apple--MobileCLIP-S0/"
                    "snapshots/71aa3e13dda93115871afbd017336535ba29886c/mobileclip_s0.pt"
                )
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                model_name, pretrained=pretrained_path, device=self.device
            )

    @torch.no_grad()
    def extract_single_frame_feature(self, frame):
        """Extract feature from a single PIL Image frame."""
        image_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast():
            feat = self.model.encode_image(image_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch_features(self, frames, batch_size=32):
        """Extract features from a list of PIL Image frames in batches."""
        all_feats = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_tensor = torch.stack([self.preprocess(f) for f in batch]).to(self.device)
            with torch.cuda.amp.autocast():
                feats = self.model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    def process_video(self, video_path, sampling_rate=6, batch_size=32):
        """
        Process video by extracting embeddings at every `sampling_rate`-th frame,
        then filling intermediate frames by copying the nearest preceding sampled
        frame's embedding so that output rows == total frames.

        For a 30 FPS video with sampling_rate=6:
          sampled indices: 0, 6, 12, 18, 24, ...
          frames 0-5   -> embedding of frame 0
          frames 6-11  -> embedding of frame 6
          frames 12-17 -> embedding of frame 12
          ...
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Cannot read video or empty: {video_path}")

        # Determine sampled frame indices
        sampled_indices = list(range(0, total_frames, sampling_rate))

        # Read sampled frames
        sampled_frames = []
        for idx in tqdm(sampled_indices, desc="Reading frames", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(Image.fromarray(frame))
            else:
                # Duplicate last successfully read frame
                if sampled_frames:
                    sampled_frames.append(sampled_frames[-1])
                else:
                    raise RuntimeError(f"Failed to read the first frame of {video_path}")
        cap.release()

        print(f"  FPS: {fps:.1f}, Total frames: {total_frames}, "
              f"Sampled: {len(sampled_frames)} frames (every {sampling_rate})")

        # Extract embeddings for sampled frames
        sampled_feats = self.extract_batch_features(sampled_frames, batch_size=batch_size)
        # sampled_feats shape: [num_sampled, feature_dim]

        # Build full-frame feature array by repeating each sampled embedding
        feat_dim = sampled_feats.shape[1]
        full_features = np.empty((total_frames, feat_dim), dtype=sampled_feats.dtype)

        for i, start_idx in enumerate(sampled_indices):
            end_idx = sampled_indices[i + 1] if i + 1 < len(sampled_indices) else total_frames
            full_features[start_idx:end_idx] = sampled_feats[i]

        return full_features


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-frame MobileCLIP features with fixed sampling rate"
    )
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos (recursive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy features")
    parser.add_argument("--model_name", type=str, default="mobileclip_s0",
                        help="MobileCLIP model name. "
                             "v1: mobileclip_s0, mobileclip_s1, mobileclip_s2, mobileclip_b | "
                             "v2: MobileCLIP2-S0, MobileCLIP2-S2, MobileCLIP2-S3, MobileCLIP2-S4, "
                             "MobileCLIP2-B, MobileCLIP2-L-14 (default: mobileclip_s0)")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model weights")
    parser.add_argument("--sampling_rate", type=int, default=6,
                        help="Extract every N-th frame (default: 6)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction (default: 32)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Video file extension (default: .mp4)")
    args = parser.parse_args()

    if args.video is None and args.video_dir is None:
        parser.error("Either --video or --video_dir must be provided")
    if args.video is not None and args.video_dir is not None:
        parser.error("Cannot use both --video and --video_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading MobileCLIP model: {args.model_name}")
    if args.pretrained_path:
        print(f"Using pretrained weights from: {args.pretrained_path}")
    extractor = FrameFeatureExtractor(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        device=args.device,
    )

    # Collect video-output pairs
    video_output_pairs = []
    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(args.output_dir, f"{video_name}.npy")
        video_output_pairs.append((args.video, output_path))
    else:
        for root, dirs, files in os.walk(args.video_dir):
            for f in files:
                if f.endswith(args.video_ext):
                    video_path = os.path.join(root, f)
                    rel_path = os.path.relpath(video_path, args.video_dir)
                    rel_npy = os.path.splitext(rel_path)[0] + ".npy"
                    output_path = os.path.join(args.output_dir, rel_npy)
                    video_output_pairs.append((video_path, output_path))
        print(f"Found {len(video_output_pairs)} videos")

    for video_path, output_path in tqdm(video_output_pairs, desc="Processing videos"):
        try:
            features = extractor.process_video(
                video_path,
                sampling_rate=args.sampling_rate,
                batch_size=args.batch_size,
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, features)
            tqdm.write(f"Saved {output_path} with shape: {features.shape}")
        except Exception as e:
            tqdm.write(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()
