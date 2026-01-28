```
python extract_mci_feature.py --video_dir /path/to/UCF_Crimes/Videos/train --output-dir /path/to/UCF_Crimes/Features/MCi0-SW --num_frames 8
```

### Arguments
- `--video`: Process a single video file
- `--video_dir`: Process all videos in a directory recursively
- `--output_dir`: Directory to save .npy feature files
- `--model_name`: MobileCLIP model variant (default: MobileCLIP-S0)
- `--window_time`: Window size in seconds (default: 2)
- `--num_frames`: Frames per window (default: 16)
- `--device`: cuda/cpu selection

### Single video

`python extract_mci_feature.py --video video.mp4 --output_dir ./features`


### Directory of videos

`python extract_mci_feature.py --video_dir ./videos --output_dir ./features --num_frames 16`


### Custom model and settings

`python extract_mci_feature.py --video_dir ./videos --output_dir ./features --model_name MobileCLIP-S1 --window_time 4`

---
