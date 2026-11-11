"""A máscara de padding do backbone esconde os tokens marcados (True = escondido), não o contrário."""

import pytest
import torch

from mtah27.jepa.backbone_joint import JointBackbone
from mtah27.tokens import A, F

T = 4


@pytest.mark.parametrize("bias", [True, False])
def test_hidden_token_does_not_affect_visible(bias):
    torch.manual_seed(0)
    enc = JointBackbone(in_dim=F, d_model=32, depth=2, n_heads=4, A=A, same_agent_bias=bias).eval()
    x = torch.randn(1, T * A, F)
    hide = torch.zeros(1, T * A, dtype=torch.bool)
    hide[0, 10:40] = True
    x_hidden, x_visible = x.clone(), x.clone()
    x_hidden[0, 20] += 5.0  # muda um token escondido
    x_visible[0, 60] += 5.0  # muda um token visível
    with torch.no_grad():
        h0 = enc(x, None, hide, T, A)[0].flatten(1, 2)[0]
        h_hid = enc(x_hidden, None, hide, T, A)[0].flatten(1, 2)[0]
        h_vis = enc(x_visible, None, hide, T, A)[0].flatten(1, 2)[0]
    visible = ~hide[0]
    assert torch.allclose(h0[visible], h_hid[visible])  # escondido não vaza para os visíveis
    others = visible.clone()
    others[60] = False
    assert not torch.allclose(h0[others], h_vis[others])  # visíveis se enxergam
