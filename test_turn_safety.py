"""Offline tests for the 'forgot to send / did nothing / invented link / rebuilt the file' fixes.
Uses a scripted fake model - no network.   python -m unittest tests.test_turn_safety -v"""
import json, os, sys, tempfile, types, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
_TMP = tempfile.mkdtemp(prefix="turnsafety-")
os.environ.setdefault("WORKSPACE_ROOT", os.path.join(_TMP, "ws"))
os.environ.setdefault("WORKSPACE_PERSIST", "0")
if "gpt2_functions" not in sys.modules:
    _stub = types.ModuleType("gpt2_functions")
    _stub._upload_bytes_to_supabase = lambda *a, **k: "https://fake/x"
    sys.modules["gpt2_functions"] = _stub

import gpt2_workspace as w
import gpt2_filetrack as ft
import gpt2_native as n

U = "turn-user"
REAL = "https://abc.supabase.co/storage/v1/object/public/bucket/turn-user/page.html"
FAKE = "https://abc.supabase.co/storage/v1/object/public/bucket/turn-user/made-up.pdf"


def _patch(testcase, obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    testcase.addCleanup(lambda: setattr(obj, name, old))


def call(name, **args):
    return {"id": f"c_{name}_{len(args)}", "name": name, "arguments": json.dumps(args)}


class Harness:
    """Scripted fake model + fake tool executor."""
    def __init__(self, rounds, initial_calls=None):
        self.rounds, self.executed, self.seen, self.kws = list(rounds), [], [], []
        self.initial_calls = initial_calls
        self.refuse_forced = False
        self.fails_left = 0                 # next N model rounds fail like "every model is rate limited"
        self.prompt = ""

    def stream(self, providers, messages, tools, **kw):
        self.seen.append([dict(m) for m in messages])
        self.kws.append(dict(kw))
        if self.fails_left > 0:
            self.fails_left -= 1
            yield ("fail", "openrouter/x: rate limited (429) | openrouter/y: HTTP 404")
            return
        if kw.get("tool_choice") and self.refuse_forced:
            yield ("fail", "provider refused tool_choice=required")
            return
        if not self.rounds:
            yield ("text", "(script ran out)")
            yield ("done", {"provider": "fake", "tool_calls": []})
            return
        r = self.rounds.pop(0)
        if r.get("text"):
            yield ("text", r["text"])
        yield ("done", {"provider": "fake", "tool_calls": r.get("calls", [])})

    def execute(self, name, args, ctx):
        self.executed.append(name)
        if name == "workspace_write":
            return True, json.dumps(w.workspace_write(U, args["path"], args["content"]))
        if name == "output_files":
            files = args.get("files") or []
            return True, json.dumps({"success": True, "files": [
                {"filename": os.path.basename(str(f)), "url": REAL} for f in files], "failed": []})
        if name == "design_check":
            return True, json.dumps({"success": True, "status": "passed"})
        return True, json.dumps({"success": True})

    def run(self):
        state = {}
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        events = list(n.run_native_turn(
            msgs, [{"name": "fake", "enabled": True}], {"userid": U, "prompt": self.prompt}, tools=[{"function": {"name": x}} for x in (
                "workspace_write", "output_files", "design_check", "workspace_run")],
            execute=self.execute, describe_call=lambda t, a: {"text": t}, stream=self.stream,
            max_rounds=20, state=state, initial_calls=self.initial_calls))
        self.state, self.messages = state, msgs
        return [e for e in events if e["type"] == "final"][-1], events


class SetUp(unittest.TestCase):
    def setUp(self):
        ft.reset()


class UnsentFiles(SetUp):
    PAGE = call("workspace_write", path="outputs/page.html", content="<html><title>x</title></html>", deliver=False)

    def test_model_corrects_itself_after_a_nudge(self):
        h = Harness([{"calls": [self.PAGE]},
                     {"text": "Here is your website!"},                                   # forgot to send
                     {"calls": [call("output_files", files=["outputs/page.html"])]},
                     {"text": "Sent - check the card."}])
        final, _ = h.run()
        self.assertEqual(h.executed, ["workspace_write", "output_files"])
        self.assertIn("NOT received", json.dumps(h.seen[2]))                            # the nudge reached the model
        self.assertEqual(ft.pending(U), [])

    def test_server_sends_it_when_the_model_keeps_forgetting(self):
        h = Harness([{"calls": [self.PAGE]}, {"text": "Here is your website!"}, {"text": "It is ready."},
                     {"text": "Enjoy!"}, {"text": "Sent it - see the card."}])
        final, _ = h.run()
        self.assertEqual(h.executed[0], "workspace_write")
        self.assertEqual(h.executed[-1], "output_files")                                  # sent by the server
        self.assertIn("design_check", h.executed)                                         # looked at first
        self.assertEqual(ft.pending(U), [])
        self.assertIn("Sent it", final["answer"])

    def test_user_said_dont_send_yet(self):
        h = Harness([{"calls": [self.PAGE]}, {"text": "Saved it. Say the word and I will send it."}])
        h.run_ctx = {"userid": U, "prompt": "make the page but don't send it yet"}
        orig = n.run_native_turn
        state = {}
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        list(orig(msgs, [{"name": "fake", "enabled": True}], h.run_ctx, tools=[{"function": {"name": "workspace_write"}}],
                  execute=h.execute, describe_call=lambda t, a: {"text": t}, stream=h.stream, state=state))
        self.assertEqual(h.executed, ["workspace_write"])                                # nothing forced out
        self.assertEqual(ft.pending(U), ["outputs/page.html"])                           # still waiting for "drop it"
        self.assertTrue(ft.says_dont_send("save it, do not send it"))
        self.assertFalse(ft.says_dont_send("make me a website"))

    def test_sent_file_is_not_nudged(self):
        h = Harness([{"calls": [call("workspace_write", path="outputs/a.html", content="<html></html>", deliver=True)]},
                     {"text": "Done."}])
        # deliver=True returns a `delivered` list in the real tool; emulate it
        real = h.execute
        h.execute = lambda name, args, ctx: (True, json.dumps({**json.loads(real(name, args, ctx)[1]),
                                              "delivered": [{"url": REAL}]})) if name == "workspace_write" else real(name, args, ctx)
        h.run()
        self.assertEqual(h.executed, ["workspace_write"])

    def test_plain_working_files_are_not_chased(self):
        h = Harness([{"calls": [call("workspace_write", path="proj/src/helper.py", content="x = 1\n")]},
                     {"text": "Added the helper."}])
        h.run()
        self.assertEqual(h.executed, ["workspace_write"])


class StalledTurns(SetUp):
    def test_announced_step_is_pushed_through(self):
        h = Harness([{"calls": [call("workspace_run")]},
                     {"text": "Let's write a python script to inspect the ReportLab structure of the uploaded PDF."},
                     {"calls": [call("workspace_run")]},
                     {"text": "Done - the structure is inspected."}])
        final, _ = h.run()
        self.assertEqual(h.executed, ["workspace_run", "workspace_run"])
        self.assertIn("NOTHING happened", json.dumps(h.seen[2]))
        self.assertIn("inspected", final["answer"])

    def test_signoffs_and_offers_are_left_alone(self):
        for text in ("All done! Let me know if you need anything else.",
                     "I can also add images if you want. Shall I?",
                     "I'll add more detail if you want me to.",
                     "Saved it. Say the word and I will send it.",
                     "Tell me when you are ready and I will build the PDF."):
            h = Harness([{"text": text}])
            h.run()
            self.assertEqual(len(h.seen), 1, text)

    def test_nudges_are_capped(self):
        h = Harness([{"text": "Let me create the file now."}] * 6)
        final, _ = h.run()
        self.assertEqual(len(h.seen), n.MAX_STALL_NUDGES + 1)                              # the first round + exactly the capped nudges
        self.assertTrue(final["answer"])

    def test_phrase_detector(self):
        yes = ["Let's write a python script to inspect it.", "Now I'll run the tests.", "I will create the PDF next.",
               "Let me check the workspace.", "Great. Next, I'm going to build the page."]
        no = ["Here is the summary.", "Let me know what you think.", "Do you want me to run it?",
              "I'll be happy to help.", "The file is ready.", "", "Should I create it?"]
        self.assertEqual([t for t in yes if not n.announces_next_step(t)], [])
        self.assertEqual([t for t in no if n.announces_next_step(t)], [])


SCREENSHOT = ("Let me run a script to inspect the exact structure, font names, margins, and design setup of the "
              "reference PDF so we can build an exact equivalent for Chemistry: Chemical Reactions.")


class ScreenshotCase(SetUp):
    """The exact sentences that ended the user's real turns."""
    def test_real_sentences_are_caught(self):
        for t in (SCREENSHOT,
                  "Let's write a python script to inspect the ReportLab structure or layout of the uploaded PDF.",
                  "I have the structure. Now I'll build the PDF with ReportLab: header, banner, and tables.",
                  "Here is the script:"):
            self.assertTrue(n.announces_next_step(t), t)
        for t in ("The PDF is ready: see the card.", "Note: it is saved in outputs.", "Do you want me to run it?"):
            self.assertFalse(n.announces_next_step(t), t)

    def test_second_nudge_requires_a_tool_call_after_tools_were_used(self):
        h = Harness([{"calls": [call("workspace_run")]}, {"text": SCREENSHOT}, {"text": SCREENSHOT},
                     {"calls": [call("workspace_run")]}, {"text": "Built it - see the card."}])
        final, _ = h.run()
        self.assertEqual(h.executed, ["workspace_run", "workspace_run"])
        self.assertEqual([bool(k.get("tool_choice")) for k in h.kws], [False, False, False, True, False])
        self.assertEqual(h.kws[3]["tool_choice"], "required")
        self.assertIn("Built it", final["answer"])

    def test_no_forcing_when_no_tool_was_used_yet(self):
        h = Harness([{"text": SCREENSHOT}] * 5)
        h.run()
        self.assertFalse(any(k.get("tool_choice") for k in h.kws))

    def test_if_no_model_accepts_forcing_the_turn_still_continues(self):
        h = Harness([{"calls": [call("workspace_run")]}, {"text": SCREENSHOT}, {"text": SCREENSHOT},
                     {"text": "Okay - here is what I found so far."}])
        h.refuse_forced = True
        final, _ = h.run()
        self.assertTrue(any(k.get("tool_choice") for k in h.kws))                        # it tried
        self.assertIn("what I found", final["answer"])                                    # and recovered, no crash


class ForcedChoiceOnTheWire(unittest.TestCase):
    class _Resp:
        def __init__(self, code, lines=(), text=""):
            self.status_code, self._lines, self.text, self.headers = code, list(lines), text, {}
        def iter_lines(self, decode_unicode=True):
            return iter(self._lines)
        def close(self): pass

    def _providers(self):
        mk = lambda name: {"name": name, "model": "m", "url": "http://x/" + name, "headers": {}, "enabled": True}
        return [mk("strict"), mk("ok")]

    def test_required_is_sent_and_a_refusing_model_is_skipped_not_blacklisted(self):
        sent = []
        def post(url, headers=None, json=None, timeout=None, stream=None):
            sent.append((url, json.get("tool_choice")))
            if url.endswith("strict"):
                return self._Resp(400, text="tool_choice required not supported")
            return self._Resp(200, ['data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}', "data: [DONE]"])
        tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]
        out = list(n.stream_chat(self._providers(), [{"role": "user", "content": "x"}], tools, temperature=0, max_tokens=50,
                                 tool_choice="required", post=post, sleep=lambda s: None))
        self.assertEqual([c for _u, c in sent], ["required", "required"])
        self.assertEqual(out[-1][0], "done")
        self.assertTrue(n.HEALTH.ok(("tools", "strict")) if hasattr(n.HEALTH, "ok") else True)   # not blacklisted
        # and by default nothing changes: still "auto"
        sent.clear()
        list(n.stream_chat(self._providers()[1:], [{"role": "user", "content": "x"}], tools, temperature=0, max_tokens=50,
                           post=post, sleep=lambda s: None))
        self.assertEqual([c for _u, c in sent], ["auto"])


REAL_LOG_SENTENCES = [
    "I am now analyzing the exact font styles, colors, page structure, and canvas layout of the reference PDF so I can "
    "build your comprehensive 13+ page Chemistry study guide on Covalent Bonding and Chemical Reactions.",
    "I am copying the reference PDF from your uploads to the workspace so I can analyze its design and layout.",
    "Let me check the exact location of the reference file in the workspace.",
]
BIG_PROMPT = ("check my upload folder for a PDF file analyze the PDF file use the same design and same format for me "
              "to create a chemistry pdf on topic covalent bond and another topic reaction. must be 15 pages")


class SecondRealCase(SetUp):
    """From the user's Render log + screenshots: the turn ended on 'I am now analyzing ... so I can build'."""
    def test_progressive_announcements_are_caught(self):
        self.assertEqual([t[:30] for t in REAL_LOG_SENTENCES if not n.announces_next_step(t)], [])
        for ok in ("I am now able to help with that.", "I'm happy to explain more.", "All set - it is in the card."):
            self.assertFalse(n.announces_next_step(ok), ok)

    def test_a_long_build_can_stall_several_times_and_still_finishes(self):
        rounds = []
        for i in range(4):                                       # stall, tool, stall, tool, ...
            rounds += [{"text": REAL_LOG_SENTENCES[i % 3]}, {"calls": [call("workspace_run")]}]
        rounds += [{"text": "All done - the PDF is in the card."}]
        h = Harness([{"calls": [call("workspace_run")]}] + rounds)
        final, _ = h.run()
        self.assertEqual(h.executed.count("workspace_run"), 5)
        self.assertIn("All done", final["answer"])

    def test_unfinished_file_task_is_caught_whatever_the_wording(self):
        h = Harness([{"calls": [call("workspace_run")]}, {"text": "I have analyzed the reference style."},
                     {"calls": [call("workspace_write", path="outputs/guide.pdf", content="x")]},
                     {"text": "The guide is built."}])
        h.prompt = BIG_PROMPT
        final, _ = h.run()
        self.assertIn("workspace_write", h.executed)                                      # it was pushed on to make the file

    def test_questions_and_refusals_are_not_pushed(self):
        for text in ("Which of the two topics should come first?", "Sorry, I can't open that PDF."):
            h = Harness([{"calls": [call("workspace_run")]}, {"text": text}])
            h.prompt = BIG_PROMPT
            h.run()
            self.assertEqual(h.executed, ["workspace_run"], text)


class RateLimitMidTask(SetUp):
    def test_waits_and_retries_instead_of_stopping(self):
        _patch(self, n, "RATE_WAIT_S", 0.0)
        h = Harness([{"text": "Working on it.", "calls": [call("workspace_run")]},
                     {"text": "All done - it is ready."}])
        h.run_first = True
        # fail the round that follows the tool call, twice, then recover
        orig = h.stream
        state = {"seen_tool": False}
        def stream(providers, messages, tools, **kw):
            if state["seen_tool"] and h.fails_left == 0 and not state.get("armed"):
                h.fails_left, state["armed"] = 2, True
            yield from orig(providers, messages, tools, **kw)
            state["seen_tool"] = True
        h.stream = stream
        final, _ = h.run()
        self.assertIn("All done", final["answer"])
        self.assertEqual(h.fails_left, 0)

    def test_gives_up_politely_with_the_partial_kept(self):
        _patch(self, n, "RATE_WAIT_S", 0.0)
        h = Harness([{"text": "Starting the build.", "calls": [call("workspace_run")]}])
        orig = h.stream
        def stream(providers, messages, tools, **kw):
            if len(h.seen) >= 1:
                h.fails_left = 99
            yield from orig(providers, messages, tools, **kw)
        h.stream = stream
        final, _ = h.run()
        self.assertIn("Starting the build", final["answer"])
        self.assertIn("continue", final["answer"])                                         # tells the user what to do


class DropTheFile(SetUp):
    def test_existing_file_is_delivered_not_rebuilt(self):
        w.workspace_write(U, "outputs/guide.pdf", "%PDF fake")
        ft.note_made(U, "outputs/guide.pdf")
        paths = ft.send_request_paths(U, "drop the file")
        self.assertEqual(paths, ["outputs/guide.pdf"])
        h = Harness([{"text": "Here you go - it is in the card."}], initial_calls=ft.build_delivery_calls(U, paths))
        final, _ = h.run()
        self.assertNotIn("workspace_write", h.executed)                                  # NOT recreated
        self.assertEqual(h.executed[-1], "output_files")
        self.assertEqual(ft.pending(U), [])

    def test_no_pending_file_means_the_model_handles_it(self):
        self.assertEqual(ft.send_request_paths("nobody-here", "drop the file"), [])

    def test_normal_messages_do_not_trigger(self):
        w.workspace_write(U, "outputs/x.pdf", "%PDF")
        ft.note_made(U, "outputs/x.pdf")
        self.assertEqual(ft.send_request_paths(U, "how do I send a file in python"), [])
        self.assertEqual(ft.send_request_paths(U, "make me a website for my bakery"), [])


class BadJsonAndLinks(SetUp):
    def test_cut_off_file_call_tells_the_model_to_split_it(self):
        bad = {"id": "bad1", "name": "workspace_write", "arguments": '{"path": "outputs/a.html", "content": "<html><body>'}
        h = Harness([{"calls": [bad]}, {"text": "Ok, I will send it in parts."}])
        h.run()
        self.assertEqual(h.executed, [])                                                  # nothing half-saved
        self.assertIn("workspace_append", json.dumps(h.seen[1]))

    def test_invented_storage_links_are_removed_real_ones_kept(self):
        h = Harness([{"calls": [call("output_files", files=["outputs/page.html"])]},
                     {"text": f"Your site: [open it]({FAKE}) and the file {REAL} too."}])
        w.workspace_write(U, "outputs/page.html", "<html></html>")
        final, _ = h.run()
        self.assertNotIn("made-up", final["answer"])
        self.assertIn(REAL, final["answer"])
        self.assertIn("open it", final["answer"])

    def test_scrubber_leaves_normal_links_alone(self):
        t = "See [Wikipedia](https://en.wikipedia.org/wiki/Chemistry) for more."
        self.assertEqual(ft.scrub_invented_links(t, set()), t)

    def test_success_false_is_not_counted_as_success(self):
        h = Harness([{"calls": [call("design_check", path="x.html")]}, {"text": "Checked."}])
        h.execute = lambda name, args, ctx: (True, json.dumps({"success": False, "error": "nope"}))
        h.run()
        self.assertNotIn("design_check", h.state["tools_ok"])
        self.assertIn("design_check", h.state["tools_called"])


class RegistryEffects(SetUp):
    def test_each_tool_updates_the_registry(self):
        w.workspace_write(U, "outputs/r.md", "# hi")
        w.workspace_write(U, "outputs/c.pdf", "%PDF")
        made = ft.note_result(U, "workspace_write", {}, json.dumps({"success": True, "path": "outputs/r.md"}), True)
        self.assertEqual(made, ["outputs/r.md"])
        made = ft.note_result(U, "run_code", {}, json.dumps({"success": True, "saved_to_workspace": [{"path": "outputs/c.pdf"}]}), True)
        self.assertEqual(made, ["outputs/c.pdf"])
        self.assertEqual(sorted(ft.pending(U)), ["outputs/c.pdf", "outputs/r.md"])
        ft.note_result(U, "output_files", {"files": ["outputs/c.pdf"]}, json.dumps(
            {"success": True, "files": [{"filename": "c.pdf", "url": REAL}], "failed": []}), True)
        self.assertEqual(ft.pending(U), ["outputs/r.md"])
        # a failed delivery leaves it pending; editing a sent file makes it pending again
        ft.note_result(U, "output_files", {"files": ["outputs/r.md"]}, json.dumps(
            {"success": False, "files": [], "failed": [{"file": "outputs/r.md", "error": "x"}]}), True)
        self.assertEqual(ft.pending(U), ["outputs/r.md"])
        ft.note_made(U, "outputs/c.pdf")
        self.assertIn("outputs/c.pdf", ft.pending(U))

    def test_files_note_lists_what_exists(self):
        w.workspace_write(U, "outputs/n.pdf", "%PDF")
        ft.note_made(U, "outputs/n.pdf")
        note = ft.files_note(U)
        self.assertIn("NOT yet received", note)
        self.assertIn("outputs/n.pdf", note)
        self.assertIn("NEVER recreate", note)
        self.assertEqual(ft.files_note("user-with-nothing"), "")


if __name__ == "__main__":
    unittest.main()
