from torch.utils.data import Dataset
import pandas as pd
from decord import VideoReader, bridge, cpu
import os.path as osp
from huggingface_hub import snapshot_download
from typing import cast
import torch

bridge.set_bridge("torch")


class VideoDataset(Dataset):
    def __init__(self, repo_id: str = "Jtic/ddong-data", local_dir: str = "data/"):
        metadata = osp.join(local_dir, "metadata.csv")

        if osp.exists(local_dir):
            # 이미 있는 경우
            self.root_dir = local_dir

        else:
            self.root_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
            )

        self.metadata = pd.read_csv(metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]

        data_dir = osp.join(self.root_dir, row["savedat"])

        clips: list[torch.Tensor] = []

        for i in range(25):
            data_path = osp.join(data_dir, f"{i}.avi")
            vr = VideoReader(data_path, cpu())
            clip = cast(torch.Tensor, vr.get_batch(range(len(vr)))).permute(3, 0, 1, 2)
            clips.append(clip)

        label: str = row["label"]

        videoes = torch.cat(clips, dim=1)

        return videoes, label


if __name__ == "__main__":
    dataset = VideoDataset()
    videoes, label = dataset[0]
    print(videoes.shape, label)
