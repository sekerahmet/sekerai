"""RGN — Rasyonel Çekim Ağı (Rational Gravity Network).

Decoder-only dil modeli: ters kare çekim dikkati + rasyonel MLP.  Tasarım: docs/RGN_tasarim_dokumani.md

Alt paketler:
    rgn.model  — model bileşenleri (dikkat, MLP, konum, normalizasyon, RGNModel)
    rgn.data   — veri setleri (kayıt, hazırlama, yükleyici)
    rgn.train  — eğitim döngüsü, optimizer, değerlendirme, teşhis, checkpoint
    rgn.bench  — hız ve bellek profilleri
"""
from rgn.config import DataConfig, ModelConfig, RunConfig, TrainConfig, load_config
from rgn.model import RGNModel

__all__ = ["DataConfig", "ModelConfig", "RunConfig", "TrainConfig", "load_config", "RGNModel"]
__version__ = "0.1.0"
