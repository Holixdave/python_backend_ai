"""Offline tests for ai_push (FCM for the AI chat). Run:  python -m unittest tests.test_ai_push -v"""
import json, os, sys, time, unittest
from datetime import timedelta
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ai_push as ap


def patch(tc, obj, name, value):
    had, old = hasattr(obj, name), getattr(obj, name, None)
    setattr(obj, name, value)
    tc.addCleanup(lambda: setattr(obj, name, old) if had else delattr(obj, name))


class Text(unittest.TestCase):
    def test_suggestion_tags_are_extracted_and_never_leak(self):
        t, s = ap.extract_suggestions("Here you go.\n<<suggestion>>Okay<<suggestion>>\n<suggestion>Alright</suggestion>\n<<suggestion>>cut off")
        self.assertEqual(s, ["Okay", "Alright"])
        self.assertEqual(t, "Here you go.")
        self.assertEqual(ap.extract_suggestions("plain"), ("plain", []))

    def test_preview_flattens_markdown(self):
        md = "# Title\n\nSome **bold** and ==marked== text with a [link](https://x.io) and `code`.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```py\nprint(1)\n```\n\n- one\n- two\n\n> quote\n\n---\n<<note>>careful<<note>>"
        p = ap.clean_preview(md, 400)
        self.assertEqual(p, "Title Some bold and marked text with a link and code. [code] one two quote careful")
        for junk in ("**", "==", "](", "|---", "<<", "```", "# "):
            self.assertNotIn(junk, p)

    def test_preview_cuts_at_a_word_with_an_ellipsis(self):
        p = ap.clean_preview("word " * 100, 40)
        self.assertTrue(p.endswith("…") and len(p) <= 41 and not p.endswith(" …"))
        self.assertEqual(ap.clean_preview(""), "")

    def test_clip(self):
        self.assertEqual(len(ap.clip("x" * 900)), 600)
        self.assertEqual(ap.clip(None), "")


class Builders(unittest.TestCase):
    def test_reply_carries_suggestions_for_the_buttons(self):
        final = {"answer": "Photosynthesis is how plants make food.\n<<suggestion>>Explain with an example<<suggestion>>\n<<suggestion>>Okay<<suggestion>>", "blocks": []}
        d = ap.build_reply("u1", "job1", final)
        self.assertEqual((d["notifType"], d["jobId"], d["recipientId"], d["title"]), ("ai_reply", "job1", "u1", "OOOR"))
        self.assertEqual(d["body"], "Photosynthesis is how plants make food.")
        self.assertEqual(json.loads(d["suggestions"]), [{"l": "Explain with an example", "t": "Explain with an example"}, {"l": "Okay", "t": "Okay"}])
        self.assertEqual(d["canContinue"], "false")
        self.assertTrue(all(isinstance(v, str) for v in d.values()))

    def test_final_suggestions_win_dedupe_and_long_labels_are_cut_but_text_is_kept(self):
        long = "Explain this to me again but slowly and with a worked example please"
        d = ap.build_reply("u", "j", {"answer": "x y z", "suggestions": [long, "okay", "OKAY", {"text": "Alright"}, "extra"]})
        s = json.loads(d["suggestions"])
        self.assertEqual([x["t"] for x in s], [long, "okay", "Alright"])         # capped at 3, "OKAY" deduped
        self.assertEqual(len(s[0]["l"]), ap.LABEL_CHARS)
        self.assertTrue(s[0]["l"].endswith("…"))

    def test_cut_off_reply_offers_continue(self):
        d = ap.build_reply("u", "j", {"answer": "half an answ"}, {"show": True, "reason": "cut_off"})
        self.assertEqual(d["canContinue"], "true")

    def test_empty_answer_pushes_nothing_unless_there_is_a_card_or_file(self):
        self.assertIsNone(ap.build_reply("u", "j", {"answer": "  <<suggestion>>Hi<<suggestion>> "}))
        d = ap.build_reply("u", "j", {"answer": "", "blocks": [{"type": "card"}]})
        self.assertEqual(d["body"], "Your answer is ready.")

    def test_error(self):
        d = ap.build_error("u", "j", "Something went wrong. Please try again.", {"show": True})
        self.assertEqual((d["notifType"], d["canContinue"]), ("ai_error", "true"))

    def test_question_keeps_two_options_for_buttons(self):
        d = ap.build_question("u", "j", {"question": "Which subject first?", "options": [{"label": "Maths"}, "English", {"label": "Physics"}]})
        self.assertEqual(d["notifType"], "ai_question")
        self.assertEqual([o["t"] for o in json.loads(d["options"])], ["Maths", "English"])
        self.assertIsNone(ap.build_question("u", "j", {"question": "   "}))

    def test_approval_has_ids_risk_and_a_see_more_text(self):
        cmd = "rm -rf build && " + "x" * 800
        d = ap.build_approval("u", "j", {"id": "ap1", "kind": "command", "title": "OOOR wants to run this command", "target": cmd,
                                         "why": "To clean the build.", "risk": "high", "expiresAt": 123})
        self.assertEqual((d["notifType"], d["approvalId"], d["approvalKind"], d["risk"]), ("ai_approval", "ap1", "command", "high"))
        self.assertEqual(d["title"], "OOOR wants to run this command")
        self.assertLessEqual(len(d["body"]), 120)                              # the collapsed line is short...
        self.assertTrue(d["bigText"].startswith("To clean the build.\nrm -rf build"))   # ...the expanded one has the command
        self.assertLessEqual(len(d["bigText"]), ap.MAX_VALUE)


class Delivery(unittest.TestCase):
    def setUp(self):
        ap._token_cache.clear()
        self.sent, self.docs = [], {"u1": {"fcmToken": "tok1"}}
        patch(self, ap, "_user_doc", lambda uid: self.docs.get(uid, {}))
        patch(self, ap, "_fcm_send", lambda token, data, ttl: self.sent.append((token, data, ttl)))
        patch(self, ap, "_run", lambda fn, *a: fn(*a))                          # run inline

    def test_sends_data_only_strings_to_the_users_token(self):
        self.assertTrue(ap.deliver("u1", {"notifType": "ai_reply", "n": 5, "none": None, "big": "x" * 900}))
        token, data, ttl = self.sent[0]
        self.assertEqual(token, "tok1")
        self.assertTrue(all(isinstance(v, str) for v in data.values()))
        self.assertEqual((data["n"], data["none"], len(data["big"])), ("5", "", 600))
        self.assertNotIn("notification", data)

    def test_no_token_or_switched_off_sends_nothing(self):
        self.assertFalse(ap.deliver("nobody", {"notifType": "x"}))
        self.docs["u2"] = {"fcmToken": "t", "aiPushEnabled": False}
        self.assertFalse(ap.deliver("u2", {"notifType": "x"}))
        self.docs["u3"] = {"fcmToken": "t", "aiPushEnabled": True}
        self.assertTrue(ap.deliver("u3", {"notifType": "x"}))
        self.assertEqual(len(self.sent), 1)

    def test_stale_token_is_forgotten_not_retried(self):
        cleared = []
        patch(self, ap, "_forget_token", lambda uid, tok: cleared.append((uid, tok)))
        def boom(token, data, ttl): raise ap.StaleToken("unregistered")
        patch(self, ap, "_fcm_send", boom)
        self.assertFalse(ap.deliver("u1", {"notifType": "x"}))
        self.assertEqual(cleared, [("u1", "tok1")])

    def test_nothing_ever_raises(self):
        def bad_doc(uid): raise RuntimeError("firestore down")
        patch(self, ap, "_user_doc", bad_doc)
        self.assertFalse(ap.deliver("u1", {"notifType": "x"}))
        def bad_send(token, data, ttl): raise ValueError("fcm down")
        ap._token_cache.clear(); patch(self, ap, "_user_doc", lambda uid: self.docs.get(uid, {})); patch(self, ap, "_fcm_send", bad_send)
        self.assertFalse(ap.deliver("u1", {"notifType": "x"}))
        patch(self, ap, "deliver", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
        ap.notify_reply("u1", "j", {"answer": "hello there"})                  # hooks swallow everything
        ap.notify_error("u1", "j", "m"); ap.notify_question("u1", "j", {"question": "q?"})
        ap.notify_approval("u1", "j", {"id": "a"}); ap.notify_approval_closed("u1", "j", "a", "expired")

    def test_hooks_send_the_right_types_and_a_fresh_approval_ttl(self):
        ap.notify_reply("u1", "j1", {"answer": "All done here."})
        ap.notify_error("u1", "j2", "boom")
        ap.notify_question("u1", "j3", {"question": "Pick one?", "options": ["A", "B"]})
        ap.notify_approval("u1", "j4", {"id": "ap9", "kind": "mcp", "title": "OOOR wants to use Notion", "target": "Notion › create_page",
                                         "expiresAt": int(time.time() * 1000) + 120_000})
        ap.notify_approval_closed("u1", "j4", "ap9", "approved")
        self.assertEqual([d["notifType"] for _, d, _ in self.sent], ["ai_reply", "ai_error", "ai_question", "ai_approval", "ai_clear"])
        ttl = self.sent[3][2]
        self.assertTrue(timedelta(seconds=100) <= ttl <= timedelta(seconds=121))      # expires WITH the approval
        self.assertEqual(self.sent[4][1]["approvalId"], "ap9")
        self.assertEqual(ap._approval_ttl({"expiresAt": 1}), timedelta(seconds=30))   # already expired -> the floor
        self.assertEqual(ap._approval_ttl({}), timedelta(seconds=300))

    def test_token_lookups_are_cached_briefly(self):
        calls = []
        patch(self, ap, "_user_doc", lambda uid: calls.append(uid) or {"fcmToken": "t"})
        ap.deliver("u1", {"a": 1}); ap.deliver("u1", {"a": 2})
        self.assertEqual(len(calls), 1)


class Wiring(unittest.TestCase):
    def test_ai_jobs_calls_every_hook(self):
        src = open(os.path.join(HERE, "ai_jobs.py"), encoding="utf-8").read()
        for hook, n in (("notify_reply", 1), ("notify_error", 2), ("notify_question", 1), ("notify_approval(", 1), ("notify_approval_closed", 1)):
            self.assertEqual(src.count("ai_push." + hook), n, hook)

    def test_main_still_imports_with_the_new_module(self):
        import py_compile
        py_compile.compile(os.path.join(HERE, "ai_push.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()
