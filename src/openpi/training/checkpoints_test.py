import dataclasses
from types import SimpleNamespace

from openpi.training import checkpoints


@dataclasses.dataclass
class _FakeTrainState:
    params: object
    opt_state: object
    ema_params: object


class _FakeManager:
    def __init__(self):
        self.step = None
        self.items = None

    def save(self, step, items):
        self.step = step
        self.items = items


class _FakeDataLoader:
    def data_config(self):
        return SimpleNamespace(norm_stats=None, asset_id=None)


def test_save_state_params_only_drops_raw_params_and_optimizer_state():
    manager = _FakeManager()
    state = _FakeTrainState(
        params={"raw": 1},
        opt_state={"adam": 2},
        ema_params={"ema": 3},
    )

    checkpoints.save_state(
        manager,
        state,
        _FakeDataLoader(),
        9999,
        params_only=True,
    )

    assert manager.step == 9999
    assert manager.items["params"] == {"params": {"ema": 3}}
    assert manager.items["train_state"].params == {}
    assert manager.items["train_state"].opt_state == {}
    assert manager.items["train_state"].ema_params is None
