import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import turn_memory as tm


class TurnMemory(unittest.TestCase):
    def acts(self):
        return [
            tm.action_from_call("workspace_write", {"path": "site/index.html", "content": "<h1>hi</h1>" * 50}, True, "Created site/index.html"),
            tm.action_from_call("design_check", {"path": "site/index.html"}, False, "design_check failed"),
        ]

    def test_record_keeps_code_and_resume(self):
        rec = tm.build_record("INTERRUPTED", self.acts(), [{"text": "Planning", "ok": True}], [{"filename": "a.html", "url": "https://x/a.html"}])
        self.assertIn("<h1>hi</h1>", rec)
        self.assertIn("RESUME:", rec)
        self.assertIn("workspace_write", rec)
        self.assertIn("a.html", rec)

    def test_finished_has_no_resume(self):
        rec = tm.build_record("finished", self.acts(), [])
        self.assertNotIn("RESUME:", rec)

    def test_history_old_turns_shortened_new_kept_full(self):
        rows = []
        for i in range(8):
            rows.append({"role": "user", "content": f"question {i}"})
            rows.append({"role": "assistant", "content": f"answer {i}" + tm.build_record("finished", self.acts(), [])})
        h = tm.build_history(rows)
        self.assertEqual(len(h), 16)
        self.assertIn("<h1>hi</h1>", h[-1]["content"])            # newest: full record with code
        self.assertNotIn("<h1>hi</h1>", h[1]["content"])          # oldest: shortened
        self.assertIn("shortened", h[1]["content"])

    def test_budget_drops_oldest_into_digest(self):
        rows = []
        for i in range(30):
            rows.append({"role": "user", "content": f"topic number {i} " + "x" * 2000})
            rows.append({"role": "assistant", "content": "y" * 2000})
        h = tm.build_history(rows, budget=20_000)
        self.assertLess(len(h), 60)
        self.assertEqual(h[0]["role"], "user")
        self.assertIn("EARLIER CONVERSATION", h[0]["content"])

    def test_last_unfinished(self):
        rows = [{"role": "user", "content": "build"}, {"role": "assistant", "content": "partial" + tm.build_record("FAILED", self.acts(), [])}]
        self.assertTrue(tm.last_unfinished(rows))
        self.assertTrue(tm.has_records(rows))

    def test_vision_failure_is_stated(self):
        n = tm.attachment_note(["https://x/i.jpg"], [], "", False)
        self.assertIn("NOT seen", n)


if __name__ == "__main__":
    unittest.main()


class Leak(unittest.TestCase):
    def test_copied_record_removed(self):
        bad = "Here you go.\n\n[TURN RECORD — what I really did.]\nSTATUS: finished\nTHOUGHT:•Thinking…•Working…]"
        self.assertEqual(tm.strip_memory_notes(bad), "Here you go.")

    def test_record_with_code_removed_by_scrub(self):
        from leak_guard import scrub_internal
        bad = "Done.\n\n[TURN RECORD — x]\nCODE:\n```\nprint(1)\n```\n]"
        self.assertEqual(scrub_internal(bad).strip(), "Done.")

    def test_half_typed_marker_hidden_while_streaming(self):
        self.assertEqual(tm.strip_memory_notes("Sure.\n\n[TURN REC"), "Sure.")
        self.assertEqual(tm.strip_memory_notes("array[0] is fine"), "array[0] is fine")

    def test_noise_steps_not_in_record(self):
        r = tm.build_record("finished", [], [{"text": "Thinking…"}, {"text": "Working…"}])
        self.assertNotIn("THOUGHT", r)


class LongChat(unittest.TestCase):
    def rows(self, n=80):
        out = []
        for i in range(n):
            out.append({"role": "user", "content": f"topic {i} " + "x" * 1500})
            out.append({"role": "assistant", "content": "ok " + "y" * 1500})
        return out

    def test_first_messages_named_and_summary_at_end(self):
        h = tm.build_history(self.rows(), budget=30_000)
        self.assertIn("topic 0 ", h[0]["content"])
        self.assertIn("topic 4 ", h[0]["content"])
        last = h[-1]["content"]
        self.assertIn("CONVERSATION SO FAR", last)
        self.assertIn("topic 0", last)
        self.assertIn("topic 79", last)
        self.assertLess(last.count("→"), 30)

    def test_no_summary_when_everything_fits(self):
        h = tm.build_history([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        self.assertNotIn("CONVERSATION SO FAR", h[-1]["content"])

    def test_summary_flags_unfinished_task(self):
        rows = self.rows(40)
        rows[-1]["content"] += tm.build_record("INTERRUPTED", [], [])
        h = tm.build_history(rows, budget=20_000)
        self.assertIn("NOT finished", h[-1]["content"])
