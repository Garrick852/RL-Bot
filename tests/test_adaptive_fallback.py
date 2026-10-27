"""Directive-only fallback protocol boundaries and independence."""

import json
import unittest

from governance.fallback import AdaptiveFallbackStateMachine, FallbackState


class AdaptiveFallbackTests(unittest.TestCase):
    def setUp(self):
        self.machine = AdaptiveFallbackStateMachine()

    def test_zero_never_authorizes_activation_or_recovery(self):
        directive = self.machine.evaluate(0)
        self.assertEqual(directive["state"], FallbackState.NO_ACTION)
        self.assertEqual(directive["stage"], 0)
        self.assertEqual(directive["active_config_action"], "UNCHANGED")
        self.assertIsNone(directive["risk_size_scale"])
        self.assertIsNone(directive["ppo_ent_coef"])
        self.assertIsNone(directive["alert"])

    def test_first_failure_requests_recalibration_but_never_accepts_candidate(self):
        directive = self.machine.evaluate(1)
        self.assertEqual(directive["state"], FallbackState.RECALIBRATE_AND_RETRY)
        self.assertEqual(directive["stage"], 1)
        self.assertTrue(directive["reject_candidate"])
        self.assertTrue(directive["expand_optimization_bounds"])
        self.assertEqual(directive["ppo_ent_coef"], .03)
        self.assertTrue(directive["require_fresh_oos_holdout"])
        self.assertTrue(directive["retain_last_known_good"])

    def test_second_failure_reduces_reference_risk_not_compounded_risk(self):
        directive = self.machine.evaluate(2)
        self.assertEqual(directive["state"], FallbackState.GRACEFUL_DEGRADATION)
        self.assertEqual(directive["stage"], 2)
        self.assertTrue(directive["reject_candidate"])
        self.assertEqual(directive["active_config_action"], "RETAIN_LAST_KNOWN_GOOD")
        self.assertTrue(directive["require_verified_last_known_good"])
        self.assertEqual(directive["risk_size_scale"], .5)
        self.assertEqual(directive["risk_size_reference"], "LAST_KNOWN_GOOD")
        self.assertFalse(directive["expand_optimization_bounds"])
        self.assertIsNone(directive["ppo_ent_coef"])
        self.assertEqual(directive, self.machine.evaluate(2))

    def test_circuit_breaker_inclusive_boundary_and_alert_serialization(self):
        for failures in (3, 4, 100000):
            with self.subTest(failures=failures):
                directive = self.machine.evaluate(failures)
                self.assertEqual(directive["state"], FallbackState.SUSPENDED)
                self.assertEqual(directive["stage"], 3)
                self.assertTrue(directive["block_new_entries"])
                self.assertTrue(directive["require_explicit_recovery_authorization"])
                self.assertEqual(directive["risk_size_scale"], 0)
                self.assertTrue(directive["reject_candidate"])
                self.assertEqual(directive["alert"]["severity"], "CRITICAL")
                self.assertIn(str(failures), directive["alert"]["message"])
                self.assertEqual(json.loads(json.dumps(directive))["state"], "SUSPENDED")

    def test_invalid_counts_are_not_coerced(self):
        for count in (-1, True, False, 1.0, "2", None, [], {}):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.machine.evaluate(count)

    def test_results_share_no_mutable_state(self):
        first = self.machine.evaluate(3)
        first["alert"]["message"] = "modified"
        first["risk_size_scale"] = 99
        second = self.machine.evaluate(3)
        self.assertNotEqual(first["alert"], second["alert"])
        self.assertEqual(second["risk_size_scale"], 0)
        self.assertEqual(second["consecutive_failures"], 3)


if __name__ == "__main__":
    unittest.main()
