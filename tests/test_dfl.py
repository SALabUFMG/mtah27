import io

import numpy as np

from mtah27 import dfl
from mtah27.tracking import AWAY, HOME

META = b"""<PutDataRequest><MatchInformation>
<Environment PitchX="100.0" PitchY="60.0"/>
<Teams>
 <Team TeamId="T-H" Role="home"><Players><Player PersonId="P1"/></Players></Team>
 <Team TeamId="T-A" Role="guest"><Players><Player PersonId="P2"/></Players></Team>
</Teams></MatchInformation></PutDataRequest>"""


def _frames(first, rows):
    return "".join(
        f'<Frame N="{first + i}" X="{x}" Y="{y}" {extra}/>' for i, (x, y, extra) in enumerate(rows)
    )


def _positions():
    def ball(poss, status):
        return f'BallPossession="{poss}" BallStatus="{status}"'

    first = [(0, 0, ball(1, 1)), (1, 0, ball(2, 1)), (2, 0, ball(1, 0))]
    second = [(-50, -30, ball(2, 1)), (50, 30, ball(2, 0))]

    def fset(section, team, person, first_n, rows):
        head = f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="{person}">'
        return head + _frames(first_n, rows) + "</FrameSet>"

    sets = [
        fset("firstHalf", "T-H", "P1", 10000, [(1, 2, "")] * 3),
        fset("firstHalf", "T-A", "P2", 10001, [(3, 4, "")] * 2),
        fset("firstHalf", "referee", "R", 10000, [(9, 9, "")]),
        fset("firstHalf", "BALL", "B", 10000, first),
        fset("secondHalf", "BALL", "B", 100000, second),
    ]
    return ("<PutDataRequest><Positions>" + "".join(sets) + "</Positions></PutDataRequest>").encode()


def test_parse_dfl_rotulos_do_tracking_e_coordenadas():
    pitch, teams = dfl.parse_meta(io.BytesIO(META))
    assert pitch == (100.0, 60.0) and teams == {"T-H": HOME, "T-A": AWAY}
    trk, labels = dfl.parse_positions(io.BytesIO(_positions()), pitch, teams)
    n = 3 + dfl.GAP + 2
    assert len(labels) == n and trk.players.shape == (n, 2, 2)  # árbitro fora
    assert labels[:3].tolist() == [dfl.H, dfl.A, dfl.D]  # bola morta é D mesmo com dono
    assert (labels[3 : 3 + dfl.GAP] == dfl.X).all() and labels[-2:].tolist() == [dfl.A, dfl.D]
    assert trk.team.tolist() == [HOME, AWAY]
    assert np.allclose(trk.ball[0], [50, 30]) and np.allclose(trk.ball[-2:], [[0, 0], [100, 60]])
    assert np.allclose(trk.players[0, 0], [51, 32]) and np.isnan(trk.players[0, 1]).all()
    assert np.allclose(trk.players[1, 1], [53, 34]) and np.isnan(trk.players[3:]).all()
