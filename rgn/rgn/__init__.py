"""rgn — küçük decoder-only dil modeli: YYG transformer (yerel / genel dikkat katmanları) + kelimeye özel nöron.

Paket adı tarihsel (eski RGN çekim dikkati deneyleri git etiketi arsiv/rgn'de).

Alt paketler:
    rgn.model  — model bileşenleri (dikkat, SwiGLU, RoPE, RMSNorm, özel nöron çekirdeği, RGNModel)
    rgn.data   — veri setleri (kayıt, yükleyici, hikâye düzeni)
    rgn.train  — eğitim döngüsü ve adımı, optimizer, değerlendirme, teşhis, checkpoint
"""
from rgn.config import DataConfig, ModelConfig, RunConfig, TrainConfig, load_config
from rgn.model import RGNModel

__all__ = ["DataConfig", "ModelConfig", "RunConfig", "TrainConfig", "load_config", "RGNModel"]
__version__ = "0.1.0"
