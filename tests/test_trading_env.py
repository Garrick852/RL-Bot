"""Current-quote accounting and forced-liquidation contracts."""

import unittest

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from engine_rl.envs.trading_env import DoubleEntryReconciliationError, TradingEnv


def bars(bids=(100, 104, 110), asks=(102, 106, 112)):
    mids = (np.asarray(bids, dtype=float) + np.asarray(asks, dtype=float)) / 2
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(bids), freq="min", tz="UTC"),
        "open": mids, "high": mids + 1, "low": mids - 1, "close": mids,
        "volume": np.ones(len(bids)), "bid": bids, "ask": asks,
        "signal": np.arange(len(bids), dtype=float) / 10,
    })


class TradingEnvTests(unittest.TestCase):
    def env(self, frame=None, **kwargs):
        settings = dict(
            feature_columns=["signal"], initial_cash=10000, quantity=2,
            point_value=10, slippage=.2, commission=.3, lambda_term=1,
        )
        settings.update(kwargs)
        return TradingEnv(bars() if frame is None else frame, **settings)

    def test_gymnasium_interface_and_determinism(self):
        env = self.env()
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info["position"], 0)
        with self.assertRaises(ValueError):
            env.step(True)
        with self.assertRaises(ValueError):
            env.step(3)

    def test_long_current_quote_reserves_exit_costs_and_terminal_formula(self):
        env = self.env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)
        self.assertAlmostEqual(env.entry_price, 102.5)
        self.assertAlmostEqual(info["unrealized_pnl"], -60)
        self.assertAlmostEqual(reward, -60)
        self.assertEqual(info["executed_row"], 0)
        self.assertEqual(obs[0], np.float32(.1))
        self.assertFalse(terminated or truncated)
        env.step(2)
        _, reward, terminated, truncated, info = env.step(0)
        expected_exit = 111 - (1 + .2 + .3 + 1)
        self.assertEqual(env.trades[-1]["exit_price"], expected_exit)
        self.assertAlmostEqual(reward, 100)
        self.assertAlmostEqual(info["terminal_liquidation_pnl"], 120)
        self.assertEqual(info["unrealized_pnl"], 0.0)
        self.assertEqual(info["position"], 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual([fill["kind"] for fill in env.fills], ["ENTRY", "TERMINAL_EXIT"])
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_short_terminal_formula_and_penalty(self):
        env = self.env(bars((110, 104, 100), (112, 106, 102)))
        env.reset()
        env.step(0)
        self.assertEqual(env.entry_price, 109.5)
        env.step(0)
        _, _, done, _, info = env.step(2)
        self.assertTrue(done)
        self.assertEqual(env.trades[-1]["exit_price"], 101 + (1 + .2 + .3 + 1))
        self.assertAlmostEqual(info["terminal_liquidation_pnl"], 120)
        self.assertEqual(info["unrealized_pnl"], 0)

    def test_flat_market_terminal_reward_is_exact_lambda_cost(self):
        for action in (0, 2):
            env = self.env(bars((100, 100, 100), (102, 102, 102)))
            env.reset()
            rewards = [env.step(action)[1], env.step(action)[1]]
            _, reward, _, _, info = env.step(action)
            self.assertAlmostEqual(reward, -20)
            self.assertAlmostEqual(sum(rewards) + reward, info["cash"] - env.initial_cash)
            self.assertAlmostEqual(info["cash"] - env.initial_cash, -80)

    def test_reversal_and_ordinary_close_do_not_double_count_terminal_pnl(self):
        env = self.env(bars((100, 104, 100, 100), (102, 106, 102, 102)))
        env.reset()
        env.step(2)
        env.step(0)
        env.step(1)
        _, _, done, _, info = env.step(2)
        self.assertTrue(done)
        self.assertEqual(len(env.trades), 2)
        self.assertEqual(info["terminal_liquidation_pnl"], 0)
        self.assertAlmostEqual(info["cash"] - env.initial_cash, sum(env.realized_trades_pnl))
        self.assertEqual(len(env.fills), 4)

    def test_internal_and_explicit_truncation_liquidate_without_future_quotes(self):
        env = self.env(max_episode_steps=2)
        env.reset()
        env.step(2)
        _, _, terminated, truncated, info = env.step(2)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.trades[-1]["exit_row"], 1)
        self.assertEqual(info["position"], 0)
        self.assertEqual(info["unrealized_pnl"], 0)
        env = self.env()
        env.reset()
        env.step(2)
        _, reward, terminated, truncated, info = env.truncate()
        self.assertEqual(env.trades[-1]["exit_row"], 0)
        self.assertAlmostEqual(reward, -20)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(info["unrealized_pnl"], 0)

    def test_terminal_bar_and_one_row_episode_cannot_open_new_positions(self):
        env = self.env(bars((100,), (102,)))
        with self.assertRaises(RuntimeError):
            env.step(2)
        env.reset()
        _, reward, done, _, info = env.step(2)
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        self.assertEqual(env.fills, [])
        self.assertEqual(info["cash"], env.initial_cash)

    def test_future_quotes_cannot_influence_current_execution_or_equity(self):
        altered = bars((100, 1000, 2000), (102, 1200, 2200))
        left, right = self.env(), self.env(altered)
        left.reset()
        right.reset()
        first, second = left.step(2), right.step(2)
        self.assertEqual(first[1:], second[1:])
        self.assertEqual(left.fills, right.fills)

    def test_reconciliation_corruption_raises_instead_of_returning_done(self):
        env = self.env()
        env.reset()
        env.step(2)
        env.step(2)
        env.cash += .0001
        with self.assertRaises(DoubleEntryReconciliationError):
            env.step(1)
        with self.assertRaises(RuntimeError):
            env.step(1)
        # The tolerance is absolute: large cash balances cannot hide errors.
        env = self.env(initial_cash=1e9)
        env.reset()
        env.cash += .01
        with self.assertRaises(DoubleEntryReconciliationError):
            env.truncate()

    def test_parameters_and_data_are_validated_and_inputs_are_copied(self):
        for field, value in (
            ("initial_cash", 0), ("quantity", -1), ("point_value", True),
            ("slippage", -.1), ("commission", np.inf), ("lambda_term", np.nan),
            ("max_episode_steps", True), ("max_episode_steps", 0),
            ("feature_columns", []), ("feature_columns", ["absent"]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.env(**{field: value})
        frame = bars()
        env = self.env(frame)
        frame.loc[0, "ask"] = 999
        env.reset()
        env.step(2)
        self.assertEqual(env.entry_price, 102.5)
        frame["signal"] = np.inf
        with self.assertRaises(ValueError):
            self.env(frame)

    def test_random_action_ledgers_and_reward_telescoping(self):
        random = np.random.default_rng(42)
        for _ in range(20):
            bids = 100 + random.integers(-10, 10, size=30).astype(float)
            env = self.env(bars(bids, bids + 2))
            env.reset()
            rewards = []
            for action in random.integers(0, 3, size=30):
                _, reward, done, truncated, info = env.step(int(action))
                rewards.append(reward)
                if done or truncated:
                    break
            self.assertAlmostEqual(sum(rewards), info["cash"] - env.initial_cash, places=8)
            self.assertAlmostEqual(
                info["cash"] - env.initial_cash,
                sum(env.realized_trades_pnl) + env.terminal_liquidation_pnl,
                places=8,
            )
            self.assertEqual(info["unrealized_pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
