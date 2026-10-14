# SPDX-License-Identifier: Apache-2.0
"""Cost reservations and transactional scene patches without inference or a GPU."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.mimo_chat import Agent, Budget, apply_replacements, observe_objects, request_messages, restore_messages

TIER = "fast"


class TestRequestHistory(unittest.TestCase):
    def test_first_turn_is_preserved_exactly(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "inspect"}]
        self.assertEqual(request_messages(messages), messages)

    def test_old_sources_are_archived_but_failures_and_current_tool_calls_survive(self):
        current = [
            {"role": "user", "content": "Try the cable"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "inspect_scene", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": '{"objects":{"cable":{"count":16}}}'},
        ]
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Poke the block"},
            {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "read_scene", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": json.dumps({"source": "archived source\n" * 10000})},
            {
                "role": "assistant",
                "tool_calls": [{"id": "b", "function": {"name": "run_routine", "arguments": '{"name":"poke"}'}}],
            },
            {"role": "tool", "tool_call_id": "b", "content": '{"ok":true,"metrics":{"passed":false,"slide_mm":160.9}}'},
            *current,
        ]
        original = copy.deepcopy(messages)
        compact = request_messages(messages)
        self.assertEqual(messages, original, "the audit history must remain untouched")
        self.assertEqual(compact[-len(current) :], current)
        self.assertLess(len(json.dumps(compact)), 2000)
        self.assertIn('"passed": false', compact[1]["content"])
        self.assertIn('"slide_mm": 160.9', compact[1]["content"])
        self.assertNotIn("archived source", compact[1]["content"])


class TestWorldObservation(unittest.TestCase):
    def test_deformables_expose_live_geometry_and_solver_ownership(self):
        refs = {
            "cloth": SimpleNamespace(particles=np.array([1, 2]), bodies=np.array([], dtype=int)),
            "cable": SimpleNamespace(particles=np.array([], dtype=int), bodies=np.array([1, 2, 3])),
            "arm": SimpleNamespace(particles=np.array([], dtype=int), bodies=np.array([0])),
        }
        sim = mock.MagicMock()
        sim.__getitem__.side_effect = refs.__getitem__
        sim.state.body_q.numpy.return_value = np.array([[0, 0, 1], [1, 0, 1], [2, 1, 1], [3, 0, 1]])
        sim.state.particle_q.numpy.return_value = np.array([[99, 99, 99], [1, 2, 3], [3, 4, 5]])
        sim.scene.entities = [
            SimpleNamespace(name="cloth", domain="cloth"),
            SimpleNamespace(name="cable", domain="rod"),
            SimpleNamespace(name="arm", domain="articulated"),
        ]
        sim.pipeline.manifest = SimpleNamespace(
            coupler="proxy",
            islands=[SimpleNamespace(name="free", backend=SimpleNamespace(solver="vbd"), entities=("cloth", "cable"))],
            channels=[SimpleNamespace(source="arm", destination="free", coupling="two_way")],
        )
        result = observe_objects(sim)
        self.assertNotIn("arm", result["objects"])
        self.assertEqual(result["objects"]["cloth"]["centroid_world"], [2, 3, 4])
        self.assertEqual(result["objects"]["cloth"]["bounds_world"], [[1, 2, 3], [3, 4, 5]])
        self.assertEqual(result["objects"]["cable"]["count"], 3)
        self.assertEqual(result["objects"]["cable"]["samples_world"][-1]["xyz"], [3, 0, 1])
        self.assertEqual(result["physics"]["channels"][0]["coupling"], "two_way")
        json.dumps(result, allow_nan=False)


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "usage.json"
        self.budget = Budget(self.path, {"input": 0.435, "output": 0.87}, 0.02)

    def body(self):
        return {"messages": [{"role": "user", "content": "Inspect the robot."}], "max_tokens": 512}

    def test_output_already_includes_reasoning(self):
        row = self.budget.reserve(self.body())
        self.budget.settle(
            row, {"prompt_tokens": 382, "completion_tokens": 126, "completion_tokens_details": {"reasoning_tokens": 38}}
        )
        self.assertAlmostEqual(self.budget.spent, 0.00027579)
        self.assertEqual(self.budget.reserved, 0)

    def test_unknown_usage_keeps_reservation(self):
        row = self.budget.reserve(self.body())
        with self.assertRaises(RuntimeError):
            self.budget.settle(row, None)
        self.assertGreater(self.budget.reserved, 0)
        self.assertEqual(self.budget.spent, 0)

    def test_reservations_prevent_unbounded_retries(self):
        count = 0
        with self.assertRaises(RuntimeError):
            for _ in range(100):
                self.budget.reserve(self.body())
                count += 1
        self.assertGreater(count, 0)
        self.assertLessEqual(self.budget.reserved, self.budget.limit)

    def test_full_conversation_and_tools_count_toward_allowance(self):
        body = self.body()
        body["tools"] = [{"description": "x" * 100000}]
        with self.assertRaises(RuntimeError):
            self.budget.reserve(body)
        self.assertEqual(self.budget.calls, [])

    def test_restart_cannot_reset_cost_history(self):
        with self.assertRaises(FileExistsError):
            Budget(self.path, self.budget.prices, 1.0)

    def test_resume_keeps_settled_costs_and_unresolved_reservations(self):
        settled = self.budget.reserve(self.body())
        self.budget.settle(settled, {"prompt_tokens": 100, "completion_tokens": 50})
        self.budget.reserve(self.body())
        resumed = Budget.resume(self.path, self.budget.prices, self.budget.limit)
        self.assertEqual(resumed.summary(), self.budget.summary())
        self.assertEqual(resumed.reserve(self.body())["number"], 3)

    def test_resume_cannot_silently_raise_the_spending_limit(self):
        with self.assertRaises(ValueError):
            Budget.resume(self.path, self.budget.prices, 1.0)

    def test_explicit_budget_increase_keeps_spend_reservations_and_audit(self):
        row = self.budget.reserve(self.body())
        self.budget.settle(row, {"prompt_tokens": 100, "completion_tokens": 50})
        self.budget.reserve(self.body())
        resumed = Budget.resume(self.path, self.budget.prices, 1.0, allow_budget_increase=True)
        self.assertEqual(resumed.limit, 1.0)
        self.assertEqual(resumed.spent, self.budget.spent)
        self.assertEqual(resumed.reserved, self.budget.reserved)
        self.assertEqual(resumed.calls, self.budget.calls)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["budget_changes"][-1]["from"], 0.02)
        self.assertEqual(saved["budget_changes"][-1]["to"], 1.0)

    def test_larger_output_allowance_is_reserved_before_request(self):
        self.budget.reserve({**self.body(), "max_tokens": 16384})
        self.assertGreater(self.budget.reserved, 16384 * 0.87 / 1e6)
        with self.assertRaises(RuntimeError):
            self.budget.reserve({**self.body(), "max_tokens": 16384})


class TestResponseLimits(unittest.TestCase):
    def test_slow_complete_stream_reassembles_tool_arguments_and_excludes_private_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent.__new__(Agent)
            agent.out = Path(directory)
            agent.budget = Budget(agent.out / "usage.json", {"input": 0.435, "output": 0.87}, 0.25)
            agent.max_output_tokens = 32768
            agent.messages = [{"role": "user", "content": "Move the cube."}]
            agent.token = "test-credential"
            agent.emit = mock.Mock()
            events = [
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "private test reasoning",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call1",
                                        "function": {"name": "run_", "arguments": '{"name":'},
                                    }
                                ],
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"name": "routine", "arguments": '"push"}'}}]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 18000},
                },
            ]
            response = io.BytesIO(
                b"".join(b"data: " + json.dumps(e).encode() + b"\n" for e in events) + b"data: [DONE]\n"
            )
            response.status, response.headers = 200, {}
            # A valid long response must survive the former ten-minute deadline.
            with (
                mock.patch("scripts.mimo_chat.urllib.request.urlopen", return_value=response),
                mock.patch("scripts.mimo_chat.time.monotonic", side_effect=[0, 1, 700, 701, 702]),
            ):
                message = agent.completion()
            call = message["tool_calls"][0]
            self.assertEqual(call["id"], "call1")
            self.assertEqual(call["function"]["name"], "run_routine")
            self.assertEqual(json.loads(call["function"]["arguments"]), {"name": "push"})
            self.assertNotIn("private test reasoning", (agent.out / "response_001.json").read_text())
            self.assertNotIn("private test reasoning", str(agent.emit.call_args_list))
            self.assertEqual(agent.budget.reserved, 0)
            self.assertGreater(agent.budget.spent, 0)

    def test_truncated_tool_call_is_charged_but_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent.__new__(Agent)
            agent.out = Path(directory)
            agent.budget = Budget(agent.out / "usage.json", {"input": 0.435, "output": 0.87}, 0.25)
            agent.max_output_tokens = 16384
            agent.messages = [{"role": "user", "content": "Move the cube."}]
            agent.token = "test-credential"
            agent.emit = mock.Mock()
            agent.dispatch = mock.Mock()
            import threading

            agent.lock = threading.RLock()
            events = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call1",
                                        "function": {"name": "edit_scene", "arguments": '{"replacements":'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 16384,
                        "completion_tokens_details": {"reasoning_tokens": 16000},
                    },
                },
            ]
            response = io.BytesIO(
                b"".join(b"data: " + json.dumps(e).encode() + b"\n" for e in events) + b"data: [DONE]\n"
            )
            response.status, response.headers = 200, {}
            with mock.patch("scripts.mimo_chat.urllib.request.urlopen", return_value=response) as opened:
                agent.run()
            self.assertEqual(json.loads(opened.call_args.args[0].data)["max_tokens"], 16384)
            self.assertEqual(agent.budget.calls[0]["usage"]["completion_tokens"], 16384)
            self.assertGreater(agent.budget.spent, 0)
            self.assertEqual(agent.budget.reserved, 0)
            agent.dispatch.assert_not_called()
            self.assertEqual(len(agent.messages), 1)
            error = next(c.kwargs["text"] for c in agent.emit.call_args_list if c.args[0] == "error")
            self.assertIn("16,384-token", error)
            self.assertIn("16,000 reasoning", error)


class TestObservationTransport(unittest.TestCase):
    def test_numpy_outcome_values_round_trip_as_plain_data(self):
        agent = Agent.__new__(Agent)
        sample = {"height_mm": np.float64(25), "passed": np.bool_(True), "cube": np.array([0.2, 0, 0.025])}

        def ctl(cmd, code):
            self.assertEqual(cmd, "eval")
            return {"ok": True, "value": repr(eval(code, {"sample": sample}))}

        agent.ctl = ctl
        self.assertEqual(agent.read_value("sample"), {"height_mm": 25.0, "passed": True, "cube": [0.2, 0, 0.025]})

    def test_measurement_error_retains_completed_motion_and_actual_state(self):
        agent = Agent.__new__(Agent)
        agent.inspect = mock.Mock(
            side_effect=[{"positions": {"cube": [0.45, 0, 0.775]}}, {"positions": {"cube": [0.2, 0, 0.025]}}]
        )
        agent.ctl = mock.Mock(side_effect=[{"ok": True}, {"playing": 0, "last_error": ""}])
        agent.read_value = mock.Mock(side_effect=ValueError("Invalid measurement"))
        with mock.patch("scripts.mimo_chat.time.sleep"):
            result = agent.dispatch("run_routine", {"name": "drop"})
        self.assertTrue(result["motion_completed"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["after"]["positions"]["cube"][2], 0.025)
        self.assertEqual(result["measurement_error"], "Invalid measurement")
        self.assertEqual(agent.ctl.call_args_list, [mock.call("play", name="drop"), mock.call("status")])


class TestSessionRestore(unittest.TestCase):
    def test_completed_tool_results_survive_and_truncated_calls_are_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            complete = {
                "role": "assistant",
                "content": "Inspecting",
                "tool_calls": [
                    {"id": "call1", "type": "function", "function": {"name": "inspect_scene", "arguments": "{}"}}
                ],
            }
            partial = {
                "role": "assistant",
                "tool_calls": [{"id": "call2", "function": {"name": "edit_scene", "arguments": "{"}}],
            }
            for number, message, finish in [(1, complete, "tool_calls"), (2, partial, "length")]:
                (out / f"response_{number:03}.json").write_text(
                    json.dumps({"message": message, "finish_reason": finish})
                )
            events = [
                {"kind": "user", "text": "Inspect."},
                {"kind": "response", "number": 1},
                {"kind": "tool_end", "call_id": "call1", "result": {"ok": True}},
                {"kind": "response", "number": 2},
            ]
            restored = restore_messages(events, out)
            self.assertEqual([m["role"] for m in restored], ["system", "user", "assistant", "tool"])
            self.assertEqual(restored[-1]["tool_call_id"], "call1")
            with self.assertRaises(ValueError):
                restore_messages(events[:2], out)


class TestPatches(unittest.TestCase):
    def test_multiple_edits_are_validated_together(self):
        source = "TARGET_Y = 0.2\nPUSH_SPEED = 0.065\n"
        result = apply_replacements(
            source,
            [
                {"old": "TARGET_Y = 0.2", "new": "TARGET_Y = -0.18"},
                {"old": "PUSH_SPEED = 0.065", "new": "PUSH_SPEED = 0.045"},
            ],
        )
        self.assertEqual(result, "TARGET_Y = -0.18\nPUSH_SPEED = 0.045\n")

    def test_ambiguous_or_invalid_patches_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_replacements("a = 1\nb = 1\n", [{"old": "1", "new": "2"}])
        with self.assertRaises(SyntaxError):
            apply_replacements("a = 1\n", [{"old": "1", "new": "("}])


if __name__ == "__main__":
    unittest.main()
