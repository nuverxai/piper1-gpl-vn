import logging

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    # CẤU HÌNH CHECKPOINT
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # Theo dõi chỉ số val_loss được log trong lightning.py
        mode="min",               # "min" nghĩa là càng thấp càng tốt
        save_top_k=1,             # Chỉ lưu giữ 1 file tốt nhất
        save_last=True,           # (Tùy chọn) Lưu thêm file cuối cùng (last.ckpt) để resume nếu cần
        filename="best-epoch={epoch}-val_loss={val_loss:.4f}", # Đặt tên file để dễ nhận biết
        auto_insert_metric_name=False
    )

    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel, 
        VitsDataModule, 
        trainer_defaults={
            "max_epochs": -1, 
            "callbacks": [checkpoint_callback] # callback
        }
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
