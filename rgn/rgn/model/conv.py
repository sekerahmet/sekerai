"""Kapılı nedensel kısa evrişim — Y (yerel) katmanında dikkatin yerine (LFM2 / Hyena tarzı karıştırıcı).

    b, c, h = W_in x            (her biri d)
    y = c ⊙ dwconv(b ⊙ h)       derinlemesine (kanal başına) nedensel evrişim, çekirdek K (sol dolgu K − 1)
    çıktı = W_out y

Bağlam yalnız son K token (pencere K'lık dikkatin evrişim karşılığı); KV önbelleği yok.  Çekirdek başlangıçta
"kendisi" (son dokunuş 1) + küçük gürültü: ilk adımlarda katman b ⊙ h'yi o konumda geçirir.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv(nn.Module):
    def __init__(self, d, kernel):
        super().__init__()
        self.kernel = int(kernel)
        self.w_in = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Parameter(torch.randn(d, 1, self.kernel) * 0.02)
        with torch.no_grad():
            self.conv[:, 0, -1] += 1.0
        self.w_out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        """x (B, N, d) -> (B, N, d); konum t yalnız t − K + 1 .. t'yi görür."""
        b, c, h = self.w_in(x).chunk(3, dim=-1)
        y = (b * h).transpose(1, 2)                                            # (B, d, N)
        y = F.conv1d(F.pad(y, (self.kernel - 1, 0)), self.conv.to(y.dtype), groups=y.shape[1])
        return self.w_out(c * y.transpose(1, 2))
