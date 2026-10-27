"""Fixed PPO profile, real CPU telemetry, and thread governance."""

import logging
import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from engine_rl.sb3_trainer import (
    DEFAULT_PPO_HYPERPARAMETERS, METRICS, THREAD_VARIABLES, SB3Trainer,
    _telemetry_callback, validate_ppo_hyperparameters,
)


class SB3TrainerTests(unittest.TestCase):
    def test_bounded_search_hyperparameters_reject_invalid_values_before_torch_import(self):
        for overrides in (
            {"learning_rate": 0}, {"learning_rate": float("nan")}, {"learning_rate": True},
            {"ent_coef": -1}, {"ent_coef": float("inf")}, {"batch_size": 63},
            {"batch_size": 128.0}, {"n_steps": 1024}, {"unrecognized": 1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_ppo_hyperparameters({**DEFAULT_PPO_HYPERPARAMETERS, **overrides})

    def test_sampled_hyperparameters_reach_real_cpu_ppo(self):
        import torch
        from engine_rl.envs.trading_env import TradingEnv

        count = 8
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=count, freq="min", tz="UTC"),
            "open": [100.] * count, "high": [101.] * count, "low": [99.] * count,
            "close": [100.] * count, "volume": [1.] * count,
            "bid": [99.9] * count, "ask": [100.1] * count, "signal": np.arange(count),
        })
        env = TradingEnv(
            frame, feature_columns=["signal"], initial_cash=10000, quantity=1,
            point_value=1, slippage=0, commission=0, lambda_term=.1,
        )
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        parameters = {
            **DEFAULT_PPO_HYPERPARAMETERS, "learning_rate": 1e-4,
            "ent_coef": .02, "batch_size": 128,
        }
        with patch.dict(os.environ):
            trainer = SB3Trainer(seed=7, hyperparameters=parameters)
            model = trainer.train(env, 2048)
            self.assertEqual(model.learning_rate, 1e-4)
            self.assertEqual(model.ent_coef, .02)
            self.assertEqual(model.batch_size, 128)
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertEqual(parameters, trainer.hyperparameters)

    def test_invalid_thread_seed_and_budget_contracts(self):
        for arguments in ({"seed": True}, {"seed": -1}, {"seed": 2**32},
                          {"seed": 7, "num_threads": 0}, {"seed": 7, "num_threads": 1.5}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SB3Trainer(**arguments)
        with patch.dict(os.environ):
            trainer = SB3Trainer(seed=7)
            for budget in (0, -1, True, 1.5):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    trainer.train(None, budget)
            with self.assertRaisesRegex(ValueError, "instantiated"):
                trainer.train(None, 1)

    def test_real_cpu_ppo_profile_and_final_optimizer_telemetry(self):
        import torch
        from engine_rl.envs.trading_env import TradingEnv

        count = 12
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=count, freq="min", tz="UTC"),
            "open": np.full(count, 100.), "high": np.full(count, 101.),
            "low": np.full(count, 99.), "close": np.full(count, 100.),
            "volume": np.ones(count), "bid": np.full(count, 99.9),
            "ask": np.full(count, 100.1), "signal": np.linspace(-1, 1, count),
        })
        env = TradingEnv(
            frame, feature_columns=["signal"], initial_cash=10000, quantity=1,
            point_value=1, slippage=.01, commission=.01, lambda_term=.1,
        )
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        with patch.dict(os.environ):
            trainer = SB3Trainer(seed=7, num_threads=1)
            self.assertTrue(all(os.environ[name] == "1" for name in THREAD_VARIABLES))
            self.assertEqual(torch.get_num_threads(), 1)
            model = trainer.train(env, 4096)
            self.assertEqual(model.device.type, "cpu")
            self.assertEqual(model.n_steps, 2048)
            self.assertEqual(model.batch_size, 256)
            self.assertEqual(model.learning_rate, 3e-4)
            self.assertEqual(model.clip_range(1), .2)
            self.assertEqual(model.gae_lambda, .95)
            self.assertEqual(model.gamma, .99)
            self.assertEqual(model.policy.net_arch, {"pi": [128, 128], "vf": [128, 128]})
            self.assertEqual(len(trainer.telemetry), 2)
            self.assertEqual([row["timesteps"] for row in trainer.telemetry], [2048, 4096])
            for row in trainer.telemetry:
                self.assertTrue(all(math.isfinite(row[name]) for name in METRICS))
            self.assertEqual(trainer.telemetry[-1]["optimizer_updates"], model._n_updates)

    def test_missing_nonfinite_telemetry_is_logged_not_fabricated(self):
        from types import SimpleNamespace
        records = []
        logger = logging.getLogger("test.ppo.health")
        callback = _telemetry_callback(records, logger)
        callback.model = SimpleNamespace(
            _n_updates=10,
            logger=SimpleNamespace(name_to_value={
                "train/approx_kl": float("nan"), "train/clip_fraction": .1,
            }),
        )
        callback.num_timesteps = 2048
        with self.assertLogs(logger, level="WARNING"):
            callback._on_training_end()
        self.assertIsNone(records[0]["approx_kl"])
        self.assertIsNone(records[0]["explained_variance"])
        self.assertEqual(records[0]["clip_fraction"], .1)
        callback._on_rollout_start()
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
