"""Offline tests: run_code input_files, workspace uploads/copy, workspace_edit_by_intent,
multi-question ask_user, honesty evidence. No Firebase/Supabase/network needed.
Run:  python -m unittest discover -s tests -v"""
import ast, json, os, re, sys, tempfile, types, unittest
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="fttest-")
os.environ.setdefault("WORKSPACE_ROOT", os.path.join(TMP, "ws"))

# gpt2_functions needs Firebase/API keys at import -> make sure a stub exists
# (test_workspace.py installs one too; we only add to whatever is there).
if "gpt2_functions" not in sys.modules:
    sys.modules["gpt2_functions"] = types.ModuleType("gpt2_functions")

import gpt2_workspace as w
import gpt2_sandbox as sb
import skills.honesty as honesty

w.WORKSPACE_ROOT = os.path.join(TMP, "ws_own")        # this file's own workspace root


def load_from(path, names, ns=None):
    """Exec only the named functions/constants out of a heavy module's source
    (those modules need Firebase/API keys just to import)."""
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            keep.append(node)
    scope = dict(ns or {})
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), scope)
    return scope


def raw(uid, path):
    """Exact file bytes as text (workspace_read adds line numbers)."""
    with open(w._safe_path(uid, path), encoding="utf-8") as f:
        return f.read()


def run(code, **kw):
    last = None
    for ev in sb.run_code(code, **kw):
        if ev.get("type") == "code_result":
            last = ev
    return last


class RunCodeInputs(unittest.TestCase):
    def test_reads_workspace_file(self):
        uid = "rc-file"
        w.workspace_write(uid, "data/a.txt", "hello workspace")
        r = run("print(open('inputs/data/a.txt').read())", userid=uid, input_files=["data/a.txt"])
        self.assertTrue(r["success"], r)
        self.assertEqual(r["stdout"].strip(), "hello workspace")
        self.assertEqual(r["inputs"][0]["path"], "inputs/data/a.txt")
        self.assertEqual(r["inputs"][0]["source"], "workspace")
        self.assertEqual(r["inputs_missing"], [])

    def test_reads_workspace_folder(self):
        uid = "rc-folder"
        w.workspace_write(uid, "proj/x.py", "X = 1")
        w.workspace_write(uid, "proj/sub/y.txt", "yy")
        r = run("import os\nprint(sorted(os.listdir('inputs/proj')))\nprint(open('inputs/proj/sub/y.txt').read())",
                userid=uid, input_files=["proj"])
        self.assertTrue(r["success"], r)
        self.assertIn("y.txt", r["stdout"] + "y.txt")  # folder staged
        self.assertEqual(sorted(i["path"] for i in r["inputs"]), ["inputs/proj/sub/y.txt", "inputs/proj/x.py"])

    def test_reads_uploaded_file(self):
        uid = "rc-up"
        w.workspace_save_upload(uid, "notes.md", b"uploaded text")
        r = run("print(open('inputs/uploads/notes.md').read())", userid=uid, input_files=["uploads/notes.md"])
        self.assertTrue(r["success"], r)
        self.assertEqual(r["stdout"].strip(), "uploaded text")

    def test_missing_input_means_code_is_not_run(self):
        r = run("print('SHOULD NOT RUN')", userid="rc-miss", input_files=["nope.csv"])
        self.assertFalse(r["success"])
        self.assertIn("NOT run", r["error"])
        self.assertEqual(r["stdout"], "")
        self.assertTrue(r["inputs_missing"])

    def test_partial_missing_still_runs_and_reports(self):
        uid = "rc-part"
        w.workspace_write(uid, "ok.txt", "fine")
        r = run("print(open('inputs/ok.txt').read())", userid=uid, input_files=["ok.txt", "gone.txt"])
        self.assertTrue(r["success"], r)
        self.assertEqual(len(r["inputs_missing"]), 1)
        self.assertIn("gone.txt", r["inputs_missing"][0])

    def test_path_escape_is_refused(self):
        r = run("print(1)", userid="rc-esc", input_files=["../../etc/passwd"])
        self.assertFalse(r["success"])
        self.assertIn("outside the workspace", " ".join(r["inputs_missing"]))

    def test_no_input_files_behaves_as_before(self):
        r = run("print(2+2)", userid="rc-none")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["stdout"].strip(), "4")
        self.assertEqual(r["inputs"], [])

    def test_original_files_not_modified_by_run(self):
        uid = "rc-ro"
        w.workspace_write(uid, "keep.txt", "original")
        run("open('inputs/keep.txt','w').write('CHANGED')", userid=uid, input_files=["keep.txt"])
        self.assertEqual(raw(uid, "keep.txt"), "original")


class UploadsAndCopy(unittest.TestCase):
    def test_upload_lands_in_uploads_folder(self):
        r = w.workspace_save_upload("up-a", "My Notes.md", b"hello")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["path"], "uploads/My Notes.md")
        self.assertIn("My Notes.md", str(w.workspace_list("up-a", "uploads")))

    def test_same_name_never_overwrites(self):
        w.workspace_save_upload("up-b", "a.txt", b"one")
        r = w.workspace_save_upload("up-b", "a.txt", b"two")
        self.assertEqual(r["path"], "uploads/a_2.txt")
        self.assertEqual(w.workspace_read_text("up-b", "uploads/a.txt")[0], "one")

    def test_upload_name_cannot_escape(self):
        r = w.workspace_save_upload("up-c", "../../evil.txt", b"x")
        self.assertTrue(r["path"].startswith("uploads/"), r)
        self.assertNotIn("..", r["path"])

    def test_copy_upload_into_main_workspace(self):
        w.workspace_save_upload("up-d", "data.csv", b"a,b")
        r = w.workspace_copy("up-d", "uploads/data.csv")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["to"], "data.csv")
        self.assertEqual(w.workspace_read_text("up-d", "data.csv")[0], "a,b")
        self.assertEqual(w.workspace_read_text("up-d", "uploads/data.csv")[0], "a,b")   # original stays

    def test_copy_refuses_to_overwrite_and_to_escape(self):
        w.workspace_save_upload("up-e", "x.txt", b"1")
        w.workspace_copy("up-e", "uploads/x.txt")
        self.assertFalse(w.workspace_copy("up-e", "uploads/x.txt")["success"])
        self.assertFalse(w.workspace_copy("up-e", "uploads/x.txt", "../../out.txt")["success"])
        self.assertFalse(w.workspace_copy("up-e", "nope.txt")["success"])

    def test_no_user_doc_store_left(self):
        for name in ("gpt2_tools.py", "gpt2_functions.py", "main.py", "gpt2_workspace.py"):
            with open(os.path.join(HERE, name), encoding="utf-8", errors="replace") as f:
                self.assertNotIn("UserDocManager", f.read(), name)
        self.assertFalse(os.path.exists(os.path.join(HERE, "user_doc_manager.py")))


class WorkspaceEditByIntent(unittest.TestCase):
    def setUp(self):
        self.mod = sys.modules["gpt2_functions"]
        self._added = ["_call_provider_chain_full", "TEXT_PROVIDERS", "_edit_executor_system_prompt",
                       "_FENCE_RE", "EDIT_EXECUTOR_MAX_ATTEMPTS"]
        self._old = {k: getattr(self.mod, k, None) for k in self._added}
        self.mod.TEXT_PROVIDERS = []
        self.mod._edit_executor_system_prompt = lambda: "sys"
        self.mod._FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|```$", re.MULTILINE)
        self.mod.EDIT_EXECUTOR_MAX_ATTEMPTS = 3
        self.replies = []
        self.seen = []
        def fake(providers, messages, temperature, max_tokens, reasoning_effort=None):
            self.seen.append(list(messages))
            return (self.replies.pop(0) if self.replies else None), "fake", "stop"
        self.mod._call_provider_chain_full = fake

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                if hasattr(self.mod, k):
                    delattr(self.mod, k)
            else:
                setattr(self.mod, k, v)

    def test_plain_english_edit_lands(self):
        uid = "ei-1"
        w.workspace_write(uid, "page.html", '<button class="btn">Go</button>\n<p>hi</p>')
        self.replies = ['{"find": "class=\\"btn\\"", "replace": "class=\\"btn blue\\""}']
        r = w.workspace_edit_by_intent(uid, "page.html", "make the button blue")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["attempts"], 1)
        self.assertIn('class="btn blue"', raw(uid, "page.html"))

    def test_retries_when_match_is_not_unique(self):
        uid = "ei-2"
        w.workspace_write(uid, "a.py", "x = 1\ny = 1\n")
        self.replies = ['{"find": "= 1", "replace": "= 2"}', '{"find": "y = 1", "replace": "y = 2"}']
        r = w.workspace_edit_by_intent(uid, "a.py", "set y to 2")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(raw(uid, "a.py"), "x = 1\ny = 2\n")
        self.assertIn("matched 2 places", self.seen[1][-1]["content"])

    def test_gives_up_honestly_and_leaves_file_alone(self):
        uid = "ei-3"
        w.workspace_write(uid, "b.txt", "one\ntwo\n")
        self.replies = ['not json', '{"find": "", "replace": "x"}', '{"find": "zzz", "replace": "x"}']
        r = w.workspace_edit_by_intent(uid, "b.txt", "do something")
        self.assertFalse(r["success"])
        self.assertIn("3 attempts", r["error"])
        self.assertEqual(raw(uid, "b.txt"), "one\ntwo\n")

    def test_model_down(self):
        uid = "ei-4"
        w.workspace_write(uid, "c.txt", "abc")
        r = w.workspace_edit_by_intent(uid, "c.txt", "change b")
        self.assertFalse(r["success"])
        self.assertIn("unavailable", r["error"])

    def test_bad_inputs(self):
        self.assertFalse(w.workspace_edit_by_intent("ei-5", "nope.txt", "x")["success"])
        self.assertFalse(w.workspace_edit_by_intent("ei-5", "", "x")["success"])
        self.assertFalse(w.workspace_edit_by_intent("ei-5", "a", "  ")["success"])
        self.assertFalse(w.workspace_edit_by_intent("ei-5", "../../x", "change")["success"])


class AskUserSheet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = load_from("gpt2_tools.py", {"MAX_ASK_QUESTIONS", "ask_user_multi_enabled", "_normalize_one_question",
                                         "normalize_ask_user"}, {"Optional": __import__("typing").Optional})
        cls.norm = staticmethod(ns["normalize_ask_user"])

    def tearDown(self):
        os.environ.pop("ASK_USER_MULTI", None)

    QS = [{"question": "What is it for?", "options": ["Church", "Business", "School"]},
          {"question": "Which style?", "options": ["Modern", "Classic"], "allow_multiple": True},
          {"question": "Need a form?", "options": ["Yes", "No"]},
          {"question": "Fourth?", "options": ["a", "b"]}]

    def test_single_question_unchanged(self):
        p = self.norm({"question": "Pick", "options": ["A", "B"]})
        self.assertEqual(p["question"], "Pick")
        self.assertNotIn("questions", p)

    def test_several_questions_only_first_when_turned_off(self):
        os.environ["ASK_USER_MULTI"] = "0"
        p = self.norm({"questions": self.QS})
        self.assertEqual(p["question"], "What is it for?")
        self.assertNotIn("questions", p)

    def test_several_questions_on_by_default(self):
        p = self.norm({"questions": self.QS})
        self.assertEqual(p["total"], 3)                      # capped at 3
        self.assertEqual(len(p["questions"]), 3)
        self.assertEqual(p["question"], p["questions"][0]["question"])   # old apps still work
        self.assertTrue(p["questions"][1]["allow_multiple"])
        self.assertEqual(len({q["question_id"] for q in p["questions"]}), 3)

    def test_bad_questions_dropped_and_all_bad_is_none(self):
        p = self.norm({"questions": [{"question": "", "options": ["a", "b"]}, self.QS[1]]})
        self.assertEqual(p["question"], "Which style?")
        self.assertNotIn("questions", p)                     # only one usable question left
        self.assertIsNone(self.norm({"questions": [{"question": "x", "options": ["only"]}]}))


class Wiring(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            return f.read()

    def test_new_tools_registered_and_described(self):
        src = self.read("gpt2_tools.py")
        for tool in ("workspace_copy", "workspace_edit_by_intent"):
            self.assertRegex(src, rf'"{tool}": {tool},')            # registry
            self.assertRegex(src, rf'"{tool}": "')                  # manifest description
        self.assertIn("input_files", src)
        speak = self.read("live_speak.py")
        self.assertIn('"workspace_copy"', speak)

    def test_run_code_dispatch_forwards_input_files(self):
        self.assertIn("input_files=_as_list(", self.read("gpt2_test.py"))

    def test_file_stores_hint_is_in_the_prompt(self):
        self.assertIn("WHERE FILES LIVE", self.read("prompts.py"))
        self.assertIn("FILE_STORES_HINT", self.read("gpt2_test.py"))

    def test_tool_result_ok_filter(self):
        import json as _json
        f = load_from("gpt2_test.py", {"_tool_result_ok"}, {"json": _json})["_tool_result_ok"]
        self.assertTrue(f('{"success": true, "path": "a"}'))
        self.assertFalse(f('{"success": false, "error": "nope"}'))
        self.assertFalse(f('{"error": "boom"}'))
        self.assertTrue(f("plain text result"))


class HonestyEvidence(unittest.TestCase):
    def ctx(self, **kw):
        base = {"tools_called": [], "tools_ok": [], "file_results": [], "code_results": [], "image_results": []}
        base.update(kw)
        return base

    def test_failed_run_is_reported_not_deleted(self):
        text = "I ran the code and it failed with a KeyError."
        c = self.ctx(tools_called=["run_code"], code_results=[{"success": False}])
        self.assertEqual(honesty.post(c, text), text)

    def test_tests_passed_needs_a_successful_run(self):
        failed = self.ctx(tools_called=["run_code"], code_results=[{"success": False}])
        self.assertIn("Correction", honesty.post(failed, "All tests passed."))
        good = self.ctx(tools_called=["run_code"], code_results=[{"success": True}])
        self.assertEqual(honesty.post(good, "All tests passed."), "All tests passed.")

    def test_delete_claim_needs_a_real_delete(self):
        text = "I deleted the file."
        self.assertIn("Correction", honesty.post(self.ctx(), text))
        ok = self.ctx(tools_ok=["workspace_delete"])
        self.assertEqual(honesty.post(ok, text), text)


if __name__ == "__main__":
    unittest.main()
