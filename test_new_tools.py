"""Offline tests for: long-prompt guard, save_pasted_text, workspace_append, python_check,
design_check wiring/static checks.  Run:  python -m unittest discover -s tests -v"""
import ast, os, shutil, sys, tempfile, types, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="newtools-")
os.environ["WORKSPACE_ROOT"] = os.path.join(TMP, "ws")
os.environ["WORKSPACE_PERSIST"] = "0"
if "gpt2_functions" not in sys.modules:           # gpt2_workspace lazily needs it (Firebase/Groq at import)
    stub = types.ModuleType("gpt2_functions")
    stub._upload_bytes_to_supabase = lambda *a, **k: "https://fake/x"
    sys.modules["gpt2_functions"] = stub

import gpt2_workspace as w
import gpt2_codecheck as cc
import gpt2_design_check as dc
from search_query import is_long_prompt, instruction_zone, needs_web_search, LONG_PROMPT_CHARS

U = "nt-user"
_real_look = dc._look          # keep the real function: another test swaps dc._look for a fake


def _patch(testcase, obj, name, value):
    """Set obj.name = value for one test and put the original back afterwards (the stubbed
    gpt2_functions module is shared with other test files)."""
    missing = object()
    old = getattr(obj, name, missing)
    setattr(obj, name, value)
    testcase.addCleanup(lambda: delattr(obj, name) if old is missing else setattr(obj, name, old))


def _read(path):
    with open(w._safe_path(U, path), encoding="utf-8") as f:
        return f.read()


class LongPromptGuard(unittest.TestCase):
    def test_threshold(self):
        self.assertFalse(is_long_prompt("hi"))
        self.assertTrue(is_long_prompt("x" * (LONG_PROMPT_CHARS + 1)))

    def test_long_text_never_keyword_searched(self):
        long = "latest news search the web jamb utme " * 80
        self.assertTrue(is_long_prompt(long))
        self.assertFalse(needs_web_search(long))
        self.assertTrue(needs_web_search("latest news on tinubu"))      # short ones still work

    def test_instruction_zone_is_only_the_edges(self):
        body = "A" * 400 + "MIDDLE-JAMB-WORD" * 100 + "Z" * 400
        zone = instruction_zone(body)
        self.assertNotIn("MIDDLE", zone)
        self.assertTrue(zone.startswith("A") and zone.endswith("Z"))
        self.assertEqual(instruction_zone("short text"), "short text")

    def test_classify_intent_skips_model_for_long_prompts(self):
        # source-level check (gpt2_functions can't be imported offline): the guard must come
        # before the classifier HTTP call and before the keyword fallback
        src = open(os.path.join(HERE, "gpt2_functions.py"), encoding="utf-8").read()
        fn = src[src.index("def classify_intent"):]
        self.assertLess(fn.index("is_long_prompt(prompt)"), fn.index("requests.post"))
        self.assertLess(fn.index("is_long_prompt(prompt)"), fn.index("_fallback_intent(prompt)"))
        self.assertIn('"topic": "general"', src[src.index("def _long_prompt_intent"):src.index("def _long_prompt_intent") + 600])


class Append(unittest.TestCase):
    def test_build_long_file_in_parts(self):
        parts = [f"Section {i}\n" + ("line of text\n" * 50) for i in range(5)]
        r = w.workspace_write(U, "outputs/long.txt", parts[0])
        self.assertTrue(r["success"])
        for p in parts[1:]:
            r = w.workspace_append(U, "outputs/long.txt", p)
            self.assertTrue(r["success"], r)
        self.assertEqual(_read("outputs/long.txt"), "".join(parts))
        self.assertEqual(r["total_bytes"], len("".join(parts).encode()))

    def test_append_creates_missing_file_and_rejects_bad_input(self):
        self.assertTrue(w.workspace_append(U, "outputs/new_one.txt", "hello")["success"])
        self.assertFalse(w.workspace_append(U, "outputs/new_one.txt", "")["success"])
        self.assertFalse(w.workspace_append(U, "outputs/", "x")["success"])
        with self.assertRaises(ValueError):
            w.workspace_append(U, "../escape.txt", "x")


class PastedText(unittest.TestCase):
    BODY = ("Chapter one.\n\nThe quick brown fox — 'quotes', \"double\", ₦500, é ü.\n\n"
            + "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(30)))

    def test_saved_verbatim_without_the_instruction(self):
        prompt = "put this in a text file man\n\n" + self.BODY
        r = w.save_pasted_text(U, prompt, [], "outputs/p1.txt", deliver=False)
        self.assertTrue(r["success"], r)
        self.assertEqual(_read("outputs/p1.txt"), self.BODY.strip() + "\n")
        self.assertEqual(r["instruction_left_out"], ["put this in a text file man"])

    def test_instruction_at_the_end_and_attachment_notes_dropped(self):
        prompt = self.BODY + "\n\nPlease save that as a .txt file" + "\n\n[Image analysis: a cat]"
        r = w.save_pasted_text(U, prompt, [], "outputs/p2.txt", deliver=False)
        self.assertTrue(r["success"], r)
        saved = _read("outputs/p2.txt")
        self.assertEqual(saved, self.BODY.strip() + "\n")
        self.assertNotIn("Image analysis", saved)

    def test_text_that_merely_mentions_files_is_not_cut(self):
        body = "Intro line that talks about a file once.\n\n" + "\n\n".join("x " * 80 for _ in range(10))
        r = w.save_pasted_text(U, "save this as txt\n\n" + body, [], "outputs/p3.txt", deliver=False)
        self.assertTrue(r["success"])
        self.assertEqual(r["instruction_left_out"], ["save this as txt"])
        self.assertTrue(_read("outputs/p3.txt").startswith("Intro line that talks about a file once."))

    def test_earlier_message_source(self):
        history = [{"role": "user", "content": self.BODY}, {"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "short"}]
        r = w.save_pasted_text(U, "put that in a file", history, "outputs/p4.txt", source="earlier_message", deliver=False)
        self.assertTrue(r["success"], r)
        self.assertEqual(_read("outputs/p4.txt"), self.BODY.strip() + "\n")

    def test_instruction_only_is_refused(self):
        r = w.save_pasted_text(U, "put this in a text file", [], "outputs/p5.txt", deliver=False)
        self.assertFalse(r["success"])
        self.assertFalse(os.path.exists(w._safe_path(U, "outputs/p5.txt")))


class PythonCheck(unittest.TestCase):
    def test_check_source(self):
        self.assertEqual(cc.check_source("x = 1\n")["syntax"], [])
        bad = cc.check_source("def f(:\n    pass\n")["syntax"][0]
        self.assertEqual(bad["line"], 1)
        self.assertIn("SyntaxError", bad["error"])
        self.assertTrue(cc.check_source("if True:\n  x=1\n    y=2\n")["syntax"])

    def test_tool_on_folder(self):
        w.workspace_write(U, "proj/good.py", "print('hi')\n")
        w.workspace_write(U, "proj/sub/bad.py", "def f()\n    return 1\n")
        w.workspace_write(U, "proj/node_modules/skip.py", "def (((\n")
        r = cc.python_check(U, "proj")
        self.assertTrue(r["success"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["files_checked"], 2)                       # node_modules skipped
        self.assertEqual(r["syntax_errors"][0]["file"], "proj/sub/bad.py")
        self.assertEqual(r["syntax_errors"][0]["line"], 1)
        self.assertIn("workspace_edit", r["next_step"])

    def test_clean_result_says_it_is_not_a_run(self):
        w.workspace_write(U, "clean/a.py", "import os\nprint(os.getcwd())\n")
        r = cc.python_check(U, "clean/a.py")
        self.assertTrue(r["ok"])
        self.assertIn("RUN it for real", r["next_step"])
        self.assertIn("does not run", r["not_tested"])

    def test_errors(self):
        self.assertFalse(cc.python_check(U, "nope/missing.py")["success"])
        w.workspace_write(U, "txtonly/a.txt", "hi")
        self.assertFalse(cc.python_check(U, "txtonly")["success"])


class DesignCheckStatic(unittest.TestCase):
    GOOD = ("<html><head><title>t</title><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>*{box-sizing:border-box}</style></head><body><h1>Bakery</h1>"
            "<p>Fresh bread every morning, baked in town since 1998.</p></body></html>")

    def test_good_page_has_no_errors(self):
        errs, _ = dc._static_html(self.GOOD, lambda r: True)
        self.assertEqual(errs, [])

    def test_bad_page_is_flagged(self):
        bad = ("<html><body><div style='width:800px'>Lorem ipsum dolor sit amet, this is long enough text</div>"
               "<img src='missing.png'></body></html>")
        errs, _ = dc._static_html(bad, lambda r: False)
        joined = " | ".join(errs)
        self.assertIn("viewport", joined)
        self.assertIn("missing local file", joined)
        self.assertIn("800px", joined)
        self.assertIn("placeholder", joined)

    def test_verdict_parser_never_fakes_a_pass(self):
        self.assertIsNone(dc._parse_verdict("looks fine to me"))
        self.assertEqual(dc._parse_verdict('{"ok": true, "issues": ["ignored"]}'), {"ok": True, "issues": []})
        v = dc._parse_verdict('```json\n{"ok": false, "issues": ["text cut off"]}\n```')
        self.assertEqual(v, {"ok": False, "issues": ["text cut off"]})
        self.assertFalse(dc._parse_verdict('{"ok": false}')["ok"])

    def test_unsupported_and_missing_files(self):
        w.workspace_write(U, "d/notes.txt", "hello")
        self.assertFalse(dc.design_check(U, "d/notes.txt")["success"])
        self.assertFalse(dc.design_check(U, "d/none.html")["success"])
        self.assertFalse(dc.design_check(U, "")["success"])

    @unittest.skipUnless(any(__import__("importlib").util.find_spec(m) for m in ("pymupdf", "fitz")), "PyMuPDF not installed")
    def test_end_to_end_with_fake_vision(self):
        w.workspace_write(U, "site/index.html", self.GOOD)
        answers = iter(['{"ok": true, "issues": []}'] * 5)
        _patch(self, dc, "_look", lambda kind, brief, label, url, why=None: dc._parse_verdict(next(answers)))
        r = dc.design_check(U, "site/index.html", brief="bakery page")
        self.assertEqual(r["status"], "passed", r)


class Gate(unittest.TestCase):
    def setUp(self):
        import gpt2_output as o
        self.o = o
        w.workspace_write(U, "gate/report.pdf", "%PDF-1.4 fake")           # content irrelevant for the gate
        w.workspace_write(U, "gate/page.html", "<html><title>t</title></html>")
        w.workspace_write(U, "gate/notes.txt", "plain text file for the gate test")
        _patch(self, sys.modules["gpt2_functions"], "_upload_bytes_to_supabase",
               lambda uid, name, data, ctype: f"https://fake/{uid}/{name}")

    def test_unchecked_pdf_and_html_are_held_back(self):
        r = self.o.output_files(U, ["gate/report.pdf", "gate/page.html"])
        self.assertFalse(r["success"])
        self.assertEqual(len(r["failed"]), 2)
        self.assertIn("design_check", r["failed"][0]["error"])

    def test_plain_files_and_uploads_are_never_gated(self):
        self.assertTrue(self.o.output_files(U, ["gate/notes.txt"])["success"])
        w.workspace_write(U, "uploads/theirs.pdf", "%PDF user file")
        self.assertTrue(self.o.output_files(U, ["uploads/theirs.pdf"])["success"])

    def test_after_design_check_it_goes_through_and_edit_needs_recheck(self):
        _patch(self, dc, "_look", lambda *a, **k: None)      # reviewer unavailable: attempt still counts
        dc.design_check(U, "gate/page.html")
        self.assertTrue(self.o.output_files(U, ["gate/page.html"])["success"])
        import time as _t; _t.sleep(0.05)
        w.workspace_write(U, "gate/page.html", "<html><title>changed</title></html>")
        os.utime(w._safe_path(U, "gate/page.html"), (_t.time() + 5, _t.time() + 5))
        self.assertFalse(self.o.output_files(U, ["gate/page.html"])["success"])

    def test_gate_can_be_switched_off(self):
        os.environ["DESIGN_CHECK_GATE"] = "0"
        try:
            self.assertTrue(self.o.output_files(U, ["gate/report.pdf"])["success"])
        finally:
            os.environ.pop("DESIGN_CHECK_GATE")


class ReviewerAnswers(unittest.TestCase):
    def test_thinking_text_does_not_break_the_verdict(self):
        v = dc._parse_verdict('<think>maybe {"ok": true}? check again</think>{"ok": false, "issues": ["text cut off"]}')
        self.assertEqual(v, {"ok": False, "issues": ["text cut off"]})
        self.assertIsNone(dc._parse_verdict('<think>no answer, still thinking {"ok": true}'))
        self.assertIsNone(dc._parse_verdict("The page looks fine."))

    def test_unparseable_answer_is_retried_once_then_reported_with_a_reason(self):
        calls = []
        _patch(self, sys.modules["gpt2_functions"], "ask_with_vision",
               lambda prompt, urls, hist: (calls.append(prompt) or {"answer": "looks fine to me"}))
        why = []
        self.assertIsNone(_real_look("html", "", "screen 1", "data:x", why))
        self.assertEqual(len(calls), 2)
        self.assertIn("not the JSON verdict", why[-1])

    def test_good_answer_on_retry_is_accepted(self):
        answers = iter(["hmm", '{"ok": true, "issues": []}'])
        _patch(self, sys.modules["gpt2_functions"], "ask_with_vision", lambda prompt, urls, hist: {"answer": next(answers)})
        self.assertEqual(_real_look("pdf", "", "page 1", "data:x", []), {"ok": True, "issues": []})

    def test_no_answer_at_all_gives_a_clear_reason(self):
        _patch(self, sys.modules["gpt2_functions"], "ask_with_vision", lambda prompt, urls, hist: {"answer": ""})
        why = []
        self.assertIsNone(_real_look("pdf", "", "page 1", "data:x", why))
        self.assertIn("no vision provider", why[0])


class LongFileProgress(unittest.TestCase):
    def test_write_and_append_report_words_and_clean_end(self):
        r = w.workspace_write(U, "prog/g.md", "# Title\n\n" + "word " * 100 + "end.")
        self.assertEqual(r["total_words"], 103)
        self.assertTrue(r["ends_cleanly"])
        r = w.workspace_append(U, "prog/g.md", "\n\nAnd then the next part continues and")
        self.assertEqual(r["total_words"], 103 + 7)
        self.assertFalse(r["ends_cleanly"])


class WiringInRegistry(unittest.TestCase):
    NEW = {"design_check", "python_check", "save_pasted_text", "workspace_append"}

    def _dict_keys(self, name):
        tree = ast.parse(open(os.path.join(HERE, "gpt2_tools.py"), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        self.fail(f"{name} not found")

    def test_registered_and_described(self):
        self.assertTrue(self.NEW <= self._dict_keys("TOOL_REGISTRY"))
        self.assertTrue(self.NEW <= self._dict_keys("TOOL_DESCRIPTIONS"))

    def test_signatures_the_session_can_fill(self):
        import inspect
        self.assertEqual(list(inspect.signature(dc.design_check).parameters)[:2], ["userid", "path"])
        params = inspect.signature(w.save_pasted_text).parameters
        self.assertIn("prompt", params)      # injected from the session, never typed by the model
        self.assertIn("history", params)

    def test_hints_exist(self):
        import prompts
        self.assertIn("design_check", prompts.CHECK_WORK_HINT)
        self.assertIn("python_check", prompts.CHECK_WORK_HINT)
        self.assertIn("save_pasted_text", prompts.LONG_TEXT_HINT)
        self.assertIn("workspace_append", prompts.LONG_TEXT_HINT)


if __name__ == "__main__":
    unittest.main()
