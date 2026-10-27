"""Matrix/vector anchor patching and EA_Template <-> Python policy isomorphism tests."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine_mt5.patcher import MQL5Patcher, SourcePatchError
from engine_rl import linear_policy as contract

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "engine_mt5" / "source" / "EA_Template.mq5"
MATRIX_SOURCE = (
    "#define ROWS 2\r\n"
    "#define COLS 3\r\n"
    "input double Risk = 0.1; // OPTIMIZED:Risk\r\n"
    "const double W[ROWS][COLS] = // OPTIMIZED_MATRIX:W\r\n"
    "  {\r\n"
    "     {0.0, 0.0, 0.0},\r\n"
    "     {0.0, 0.0, 0.0}\r\n"
    "  };\r\n"
    "const double B[ROWS] = // OPTIMIZED_MATRIX:B\r\n"
    "  {0.0, 0.0};\r\n"
    "double after = 1.0;\r\n"
)


def parse_matrix(source: str, name: str) -> np.ndarray:
    block = re.search(rf"{name}\[[^\]]+\]\[[^\]]+\]\s*=\s*//[^\n]*\n(.*?)\}};", source, re.S).group(1)
    rows = re.findall(r"\{([^{}]*)\}", block)
    return np.array([[float(value) for value in row.split(",")] for row in rows])


def parse_vector(source: str, name: str) -> np.ndarray:
    body = re.search(rf"{name}\[[^\]]+\]\s*=\s*//[^\n]*\n\s*\{{([^{{}}]*)\}};", source).group(1)
    return np.array([float(value) for value in body.split(",")])


def defines(source: str) -> dict[str, int]:
    return {name: int(value) for name, value in re.findall(r"^#define (\w+)\s+(\d+)", source, re.M)}


@pytest.fixture
def matrix_source(tmp_path):
    path = tmp_path / "EA.mq5"
    path.write_bytes(MATRIX_SOURCE.encode("utf-8"))
    return path


def test_matrix_and_vector_patch_exact_shape_line_endings_and_scalar_together(matrix_source):
    weights = [[1.5, -2.0, 3e-7], [0.0, 4, -0.25]]
    MQL5Patcher().patch_source(matrix_source, {"W": weights, "B": [0.5, -1.0], "Risk": 0.3})
    text = matrix_source.read_bytes().decode("utf-8")
    assert "\n" not in text.replace("\r\n", "")
    np.testing.assert_array_equal(parse_matrix(text, "W"), np.array(weights, dtype=float))
    np.testing.assert_array_equal(parse_vector(text, "B"), [0.5, -1.0])
    assert "Risk = 0.3;" in text and text.endswith("double after = 1.0;\r\n")
    MQL5Patcher().patch_source(matrix_source, {"W": weights})
    assert matrix_source.read_bytes().decode("utf-8") == text


@pytest.mark.parametrize("value", [
    {"W": [[0.0] * 3]},                                  # too few rows
    {"W": [[0.0] * 3, [0.0] * 4]},                       # ragged
    {"W": [[0.0] * 3, [0.0, float("nan"), 0.0]]},        # non-finite
    {"W": [[0.0] * 3, [0.0, True, 0.0]]},                # bool is not numeric
    {"W": [[0.0] * 3, [0.0, "1); hack(", 0.0]]},         # injection
    {"W": [0.0, 0.0]},                                   # flat vector for a matrix
    {"B": [0.0, 0.0, 0.0]},                              # wrong vector length
    {"B": [[0.0], [0.0]]},                               # nested for a vector
    {"B": [float("inf"), 0.0]},
])
def test_matrix_patch_rejects_bad_values_atomically(matrix_source, value):
    original = matrix_source.read_bytes()
    with pytest.raises(SourcePatchError):
        MQL5Patcher().patch_source(matrix_source, {"Risk": 0.2, **value})
    assert matrix_source.read_bytes() == original


@pytest.mark.parametrize("source", [
    MATRIX_SOURCE.replace("// OPTIMIZED_MATRIX:W", "// OPTIMIZED_MATRIX:WX"),
    MATRIX_SOURCE.replace("{0.0, 0.0, 0.0},", "{0.0, 0.0, 0.0}, // note"),
    MATRIX_SOURCE.replace("#define COLS 3\r\n", ""),
    MATRIX_SOURCE.replace("#define COLS 3\r\n", "#define COLS 3\r\n#define COLS 3\r\n"),
    "/*\r\n" + MATRIX_SOURCE + "*/\r\n",
    MATRIX_SOURCE + MATRIX_SOURCE.split("input double")[1].split("\r\n", 1)[1],
])
def test_matrix_patch_rejects_missing_duplicate_commented_or_unresolved(tmp_path, source):
    path = tmp_path / "EA.mq5"
    path.write_bytes(source.encode("utf-8"))
    with pytest.raises(SourcePatchError):
        MQL5Patcher().patch_source(path, {"W": [[1.0] * 3] * 2})
    assert path.read_bytes() == source.encode("utf-8")


def test_ea_template_structure_has_no_legacy_strict_and_never_trades():
    source = TEMPLATE.read_text(encoding="utf-8")
    assert not re.search(r"^\s*#property\s+strict", source, re.M)
    for fragment in (
        "#define OBS_DIM 12", "#define ACTION_DIM 3",
        '#include "../include/ONNXAlpha.mqh"', '#include "../include/FeatureBuilder.mqh"',
        "const double PPO_WEIGHTS[ACTION_DIM][OBS_DIM] = // OPTIMIZED_MATRIX:PPO_WEIGHTS",
        "const double PPO_BIAS[ACTION_DIM] = // OPTIMIZED_MATRIX:PPO_BIAS",
        "return INIT_FAILED;", "g_alpha.Init(InpModelFile, ALPHA_FEATURE_COUNT",
        "obs[OBS_ALPHA] = alpha_prediction;", "double sum = PPO_BIAS[i];",
        "sum += PPO_WEIGHTS[i][j] * obs[j];", "CopyRates(_Symbol, _Period, 1,",
    ):
        assert fragment in source, fragment
    weights, bias = parse_matrix(source, "PPO_WEIGHTS"), parse_vector(source, "PPO_BIAS")
    np.testing.assert_array_equal(weights, np.zeros((3, 12)))
    # Unpatched placeholder must resolve to FLAT (no exposure).
    assert contract.ACTION_NAMES[contract.linear_policy_action(weights, bias, np.ones(12))] == "FLAT"
    for forbidden in ("OrderSend", "CTrade", "Trade.mqh", "PositionClose", "OrderSendAsync"):
        assert forbidden not in source
    for index in range(8):
        assert f"input double InpAlphaStd{index} = 0.0; // OPTIMIZED:InpAlphaStd{index}" in source
        assert f"input double InpAlphaMean{index} = 0.0; // OPTIMIZED:InpAlphaMean{index}" in source


def test_ea_layout_matches_python_contract_and_trading_env_actions():
    from engine_rl.envs.trading_env import STATE_FEATURES

    values = defines(TEMPLATE.read_text(encoding="utf-8"))
    names = contract.OBSERVATION_NAMES
    assert (values["OBS_DIM"], values["ACTION_DIM"]) == (contract.OBS_DIM, contract.ACTION_DIM) == (12, 3)
    assert names[values["OBS_ALPHA"]] == "alpha_prediction"
    zscore = names[values["OBS_ZSCORE_BEGIN"]:values["OBS_ZSCORE_BEGIN"] + 8]
    assert zscore == contract.ZSCORE_FEATURE_COLUMNS
    assert names[values["OBS_SPREAD_FRACTION"]] == "spread_fraction"
    assert names[values["OBS_POSITION_SIDE"]] == "position_side"
    assert names[values["OBS_UNREALIZED_PNL"]] == "unrealized_pnl_fraction"
    assert set(contract.STATE_OBSERVATION_FEATURES) <= set(STATE_FEATURES)
    # TradingEnv target = (action - 1) * quantity: 0 short, 1 flat, 2 long.
    for action, name in enumerate(contract.ACTION_NAMES):
        assert values[f"ACTION_{name}"] == action


def test_market_observations_are_lagged_one_closed_bar():
    from data.feature_engineer import TabularFeatureEngineer

    names = list(TabularFeatureEngineer.FEATURE_NAMES)
    quotes = pd.DataFrame({"bid": [100.0, 101, 102, 103, 104], "ask": [100.2, 101.1, 102.3, 103.1, 104.2]})
    features = pd.DataFrame({name: np.arange(5.0) + offset for offset, name in enumerate(names)})
    alpha = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    frame = contract.build_market_observations(quotes, features, alpha)
    assert list(frame.index) == [1, 2, 3, 4]
    assert frame.loc[3, "alpha_prediction"] == 0.3
    assert frame.loc[3, "z_rsi"] == features.loc[2, "rsi"]
    assert frame.loc[3, "spread_fraction"] == pytest.approx((103.1 - 103.0) / 103.0)
    with pytest.raises(ValueError):
        contract.build_market_observations(quotes, features[names[::-1]], alpha)


def contract_env(rows: int = 400):
    from data.feature_engineer import TabularFeatureEngineer
    from engine_rl.envs.trading_env import TradingEnv

    rng = np.random.default_rng(11)
    mids = 100 + np.cumsum(rng.normal(0, .5, rows))
    quotes = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC"),
        "open": mids, "high": mids + 1, "low": mids - 1, "close": mids,
        "volume": np.ones(rows), "bid": mids - .05, "ask": mids + .05,
    })
    scaled = pd.DataFrame(rng.normal(size=(rows, 8)), columns=list(TabularFeatureEngineer.FEATURE_NAMES))
    alpha = pd.Series(rng.normal(0, 1e-3, rows))
    frame = contract.build_market_observations(quotes, scaled, alpha).reset_index(drop=True)
    return TradingEnv(
        frame, feature_columns=list(contract.MARKET_OBSERVATION_COLUMNS),
        state_features=contract.STATE_OBSERVATION_FEATURES, initial_cash=10000, quantity=1,
        point_value=1, slippage=0, commission=0, lambda_term=.1,
    )


def test_trading_env_state_features_extend_observation():
    env = contract_env()
    assert env.observation_space.shape == (12,)
    obs, _ = env.reset()
    assert obs.shape == (12,) and obs[10] == 0 and obs[11] == 0
    obs, *_ = env.step(2)
    assert obs[10] == 1.0
    assert obs[11] == pytest.approx(np.float32(env.unrealized_pnl / env.equity))
    obs, *_ = env.step(0)
    assert obs[10] == -1.0
    with pytest.raises(ValueError):
        type(env)(env._data, feature_columns=["spread_fraction"], state_features=["bars_held"],
                  initial_cash=1, quantity=1, point_value=1, slippage=0, commission=0, lambda_term=0)


def test_trained_linear_ppo_exports_patches_and_matches_ea_head(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from data.feature_engineer import TabularFeatureEngineer
    from engine_rl.sb3_trainer import SB3Trainer
    from models.onnx_bridge import mql5_zscore_inputs

    previous = torch.get_num_threads()
    try:
        env = contract_env()
        model = SB3Trainer(seed=5, policy_architecture="linear_actor").train(env, 2048)
        assert model.policy.net_arch == {"pi": [], "vf": [128, 128]}
        exported = contract.export_linear_policy(model)
        with pytest.raises(ValueError, match="hidden layers"):
            contract.export_linear_policy(PPO("MlpPolicy", contract_env(), device="cpu", seed=1))
    finally:
        torch.set_num_threads(previous)

    staged = tmp_path / "EA_Template.mq5"
    staged.write_bytes(TEMPLATE.read_bytes())
    manifest = {"zscore": {
        "feature_names": list(TabularFeatureEngineer.FEATURE_NAMES),
        "mean": [50.0, 0.5, 0.4, 0.1, 3.0, 0.02, -0.01, 0.01],
        "std": [12.0, 2.0, 1.5, 0.5, 1.0, 0.01, 0.005, 0.005],
    }}
    MQL5Patcher().patch_source(staged, {**mql5_zscore_inputs(manifest), **exported})
    text = staged.read_text(encoding="utf-8")
    weights, bias = parse_matrix(text, "PPO_WEIGHTS"), parse_vector(text, "PPO_BIAS")
    np.testing.assert_array_equal(weights, exported["PPO_WEIGHTS"])
    np.testing.assert_array_equal(bias, exported["PPO_BIAS"])
    assert "input double InpAlphaStd0 = 12.0; // OPTIMIZED:InpAlphaStd0" in text

    rng = np.random.default_rng(3)
    probes = list(rng.normal(0, 3, size=(200, 12)).astype(np.float32))
    obs, _ = env.reset()
    for _ in range(200):
        probes.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()
    for probe in probes:
        expected, _ = model.predict(probe, deterministic=True)
        assert contract.linear_policy_action(weights, bias, probe) == int(expected)
    with pytest.raises(SourcePatchError):
        MQL5Patcher().patch_source(staged, {"PPO_WEIGHTS": weights[:, :11].tolist()})


def test_linear_policy_action_ties_resolve_to_lowest_index_like_torch():
    assert contract.linear_policy_action(np.zeros((3, 12)), np.zeros(3), np.ones(12)) == 0
    with pytest.raises(ValueError):
        contract.linear_policy_action(np.full((3, 12), np.inf), np.zeros(3), np.ones(12))
