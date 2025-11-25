from torch.utils.data import Dataset
import pandas as pd
from decord import VideoReader, bridge, gpu
import os.path as osp

bridge.set_bridge("torch")


class VideoDataset(Dataset):
    def __init__(self, metadata: str, root_dir: str = ""):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        video_path = [
            osp.join(self.root_dir, row["savedat"], f"{i}.avi") for i in range(25)
        ]

        label = row["label"]
        total_frames = row["fps"] * row["length"]

        vr = VideoReader(video_path, gpu())
        video_tensor = vr.get_batch(list(range(total_frames))).permute(3, 0, 1, 2)

        return video_tensor, label
