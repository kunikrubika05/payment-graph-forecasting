"""Upload backends."""

from payment_graph_forecasting.infra.upload.base import Uploader
from payment_graph_forecasting.infra.upload.yadisk import YandexDiskUploader

__all__ = ["Uploader", "YandexDiskUploader"]
