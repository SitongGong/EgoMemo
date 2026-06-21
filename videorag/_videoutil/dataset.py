import os
import torch
import base64
import datasets as hf_datasets
from typing import Any
from torchvision.io import image as torch_imageio
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset

hf_datasets.disable_progress_bars()

def img_base64_to_tensor(img_base64: str) -> torch.Tensor:
    img_bytes = torch.frombuffer(base64.b64decode(img_base64), dtype=torch.uint8)
    img = torch_imageio.decode_image(img_bytes, mode=ImageReadMode.RGB)
    return img


class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        ann_file: str,
        keep_images: bool = True,
        remove_summarize_turns: bool = False,
        repeat: int = 1,
        neg_frame_sampling_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """Video dataset for multimodal chat.

        :param data_dir: the directory of the data
        :param ann_file: the relative path to the annotation file in the data directory
        :param img_feature_dir: the directory of the pre-extracted image features
        :param img_feature_keys: the keys of the image features to use
        :param keep_images: whether to still keep the images in the sample while
            image features are available
        :param remove_summarize_turns: whether to remove the summarize turns in the
            conversation. Used in offline evaluation.
        :param repeat: the number of times to repeat the dataset
        :param neg_frame_sampling_rate: the rate to sample negative (no-talk) frames
        """
        super(BaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.keep_images = keep_images
        self.remove_summarize_turns = remove_summarize_turns
        self.repeat = repeat
        self.neg_frame_sampling_rate = neg_frame_sampling_rate
        self.ann_file = ann_file
        self.data = hf_datasets.Dataset.from_json(self.ann_file)

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, idx) -> dict[str, Any]:

        if self.repeat > 1:
            idx = idx % len(self.data)

        sample = self.data[idx]
        video_uid = sample["video_uid"]
        start_idx = sample["start_frame_idx"]
        end_idx = sample["end_frame_idx"]
        metadata = sample["metadata"]# [0]
        conversation = sample["conversation"]
        if self.remove_summarize_turns:
            conversation_no_summ = []
            last_end = start_idx
            for t in conversation:
                conversation_no_summ.append(t)
                if t["role"] == "frames":
                    last_end = t["end"]
                if t["role"] == "system" and t["content"].startswith(
                    "Please summarize"
                ):
                    end_idx = last_end
                    break
            conversation = conversation_no_summ

        neg_frame_sampling_rate = self.neg_frame_sampling_rate
        if metadata is not None and metadata.get("summary_only"):
            neg_frame_sampling_rate = -1

        item = {
            "dataset": self.dataset_name,
            "sample_idx": idx,
            "video_uid": video_uid,
            "conversation": conversation,
            "start_frame_idx": start_idx,
            "end_frame_idx": end_idx,
            "neg_frame_sampling_rate": neg_frame_sampling_rate,
        }

        if self.keep_images or "encoded_images" not in item:
            # load the frames and convert them to tensors
            frames_file = os.path.join(self.data_dir, sample["frames_file"])   # [0]
            all_frames_in_video = hf_datasets.Dataset.from_file(frames_file)
            if end_idx > len(all_frames_in_video):
                raise ValueError(
                    f"End frame index {end_idx} is greater than the number of frames "
                    f"({len(all_frames_in_video)}) in the video {video_uid}"
                )
            frames = all_frames_in_video[start_idx:end_idx]["frame"]
            frames = [img_base64_to_tensor(f) for f in frames]  # [(CxHxW)]
            frames = torch.stack(frames)  # (TxCxHxW)
            item["images"] = frames

        return item
