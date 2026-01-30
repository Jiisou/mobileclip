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

class VideoFeatureExtractor:
    def __init__(self, model_name="mobileclip_s0", pretrained_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Determine if this is MobileCLIP v1 or v2
        is_mobileclip_v2 = model_name.startswith("MobileCLIP2-")

        if is_mobileclip_v2:
            # Load MobileCLIP2 using open_clip
            # Model names: MobileCLIP2-S0, MobileCLIP2-S2, MobileCLIP2-S3, MobileCLIP2-S4, MobileCLIP2-B, MobileCLIP2-L-14
            print(f"Loading MobileCLIP2 model: {model_name}")

            # Set model kwargs based on model name
            model_kwargs = {}
            if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
                model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_path if pretrained_path else "datacompdr",
                **model_kwargs
            )

            # Model needs to be in eval mode for inference
            self.model.eval()

            # Reparameterize for better performance
            self.model = reparameterize_model(self.model)
            self.model = self.model.to(self.device)
        else:
            # Load MobileCLIP v1 using mobileclip package
            # Model names: mobileclip_s0, mobileclip_s1, mobileclip_s2, mobileclip_b
            print(f"Loading MobileCLIP v1 model: {model_name}")

            if pretrained_path is None and model_name == "mobileclip_s0":
                pretrained_path = os.path.expanduser("~/.cache/huggingface/hub/models--apple--MobileCLIP-S0/snapshots/71aa3e13dda93115871afbd017336535ba29886c/mobileclip_s0.pt")

            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_path,
                device=self.device
            )

    @torch.no_grad()
    def extract_window_features(self, frames):
        """Extract features from N frames and return averaged feature vector (Temporal Mean Pooling)"""
        # Preprocess frames
        batch_images = torch.stack([self.preprocess(frame) for frame in frames]).to(self.device)

        # Extract image features using MobileCLIP
        with torch.cuda.amp.autocast():
            image_features = self.model.encode_image(batch_images)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Temporal mean pooling
        return image_features.mean(dim=0, keepdim=True).cpu().numpy()

    def process_video(self, video_path, window_time=2, num_frames=16):
        """Process video with sliding window approach"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / fps)

        video_features = []

        # Calculate half window size for centering
        half_window = window_time // 2

        # Loop through each second of the video
        for target_sec in range(duration_sec):
            # Set window boundaries (centered at target_sec, spanning window_time)
            # For target_sec=0, use 0~window_time/2 to handle edge case
            if target_sec == 0:
                start_t = 0
                end_t = half_window
            else:
                start_t = max(0, target_sec - half_window)
                end_t = min(duration_sec, target_sec + half_window)

            # Uniform sampling: extract num_frames evenly from the time window
            sample_times = np.linspace(start_t, end_t, num_frames, endpoint=False)
            batch_frames = []

            for t in sample_times:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(Image.fromarray(frame))
                else:
                    # Handle frame read failure by duplicating last frame
                    if len(batch_frames) > 0:
                        batch_frames.append(batch_frames[-1])

            # Extract features for this window
            if len(batch_frames) > 0:
                window_feat = self.extract_window_features(batch_frames)
                video_features.append(window_feat)

        cap.release()

        # Return features as [video_duration_sec, feature_dim] array
        return np.vstack(video_features)

def main():
    parser = argparse.ArgumentParser(description="Extract MobileCLIP features from videos with sliding window")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos (recursive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy features")
    parser.add_argument("--model_name", type=str, default="mobileclip_s0",
                        help="MobileCLIP model name. "
                             "v1 models: mobileclip_s0, mobileclip_s1, mobileclip_s2, mobileclip_b | "
                             "v2 models: MobileCLIP2-S0, MobileCLIP2-S2, MobileCLIP2-S3, MobileCLIP2-S4, MobileCLIP2-B, MobileCLIP2-L-14 "
                             "(default: mobileclip_s0)")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model weights")
    parser.add_argument("--window_time", type=int, default=2, help="Window time in seconds (default: 2)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames per window (default: 8)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Video file extension (default: .mp4)")
    args = parser.parse_args()

    # Validate input arguments
    if args.video is None and args.video_dir is None:
        parser.error("Either --video or --video_dir must be provided")
    if args.video is not None and args.video_dir is not None:
        parser.error("Cannot use both --video and --video_dir")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize extractor
    print(f"Loading MobileCLIP model: {args.model_name}")
    if args.pretrained_path:
        print(f"Using pretrained weights from: {args.pretrained_path}")
    extractor = VideoFeatureExtractor(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        device=args.device
    )

    # Get list of (video_path, output_path) pairs to process
    video_output_pairs = []
    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(args.output_dir, f"{video_name}.npy")
        video_output_pairs.append((args.video, output_path))
    else:
        # Recursively find all videos and preserve directory structure
        for root, dirs, files in os.walk(args.video_dir):
            for f in files:
                if f.endswith(args.video_ext):
                    video_path = os.path.join(root, f)
                    # Compute relative path from video_dir
                    rel_path = os.path.relpath(video_path, args.video_dir)
                    # Replace extension with .npy
                    rel_npy = os.path.splitext(rel_path)[0] + ".npy"
                    output_path = os.path.join(args.output_dir, rel_npy)
                    video_output_pairs.append((video_path, output_path))
        print(f"Found {len(video_output_pairs)} videos")

    # Process videos
    for video_path, output_path in tqdm(video_output_pairs, desc="Processing videos"):
        try:
            features = extractor.process_video(video_path, args.window_time, args.num_frames)

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as .npy file
            np.save(output_path, features)
            tqdm.write(f"Saved {output_path} with shape: {features.shape}")
        except Exception as e:
            tqdm.write(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()
