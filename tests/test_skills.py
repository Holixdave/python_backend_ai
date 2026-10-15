"""Offline tests for skills/ and the removed saved-docs auto-search.
Run:  python -m unittest discover -s tests -v"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import skills
from skills import honesty, brevity, clarify_first


def ctx(**kw):
    base = {"prompt": "", "tools_called": [], "tools_ok": [], "file_results": [],
            "code_results": [], "image_results": [], "sources": []}
    base.update(kw)
    return base


class Loader(unittest.TestCase):
    def test_all_skills_load_in_priority_order(self):
        names = [m.NAME for m in skills.load_skills(force=True)]
        self.assertEqual(names, ["honesty", "brevity", "clarify_first"])

    def test_pre_contains_rules(self):
        out = skills.run_pre(ctx(prompt="hello"))
        self.assertIn("HONESTY RULES", out)
        self.assertIn("BREVITY RULES", out)

    def test_broken_skill_never_kills_reply(self):
        class Bad:
            NAME = "bad"; PRIORITY = 1
            def pre(self, c): raise RuntimeError("boom")
            def post(self, c, a): raise RuntimeError("boom")
        old = list(skills._CACHE)
        skills._CACHE[:] = [Bad()] + old
        try:
            self.assertIn("HONESTY", skills.run_pre(ctx()))
            self.assertEqual(skills.run_post(ctx(), "plain answer"), "plain answer")
        finally:
            skills._CACHE[:] = old


class Honesty(unittest.TestCase):
    def test_false_file_claim_removed_and_noted(self):
        out = honesty.post(ctx(), "I have updated the following files. Here is the plan.")
        self.assertNotIn("updated the following files", out)
        self.assertIn("Here is the plan.", out)
        self.assertIn("Correction:", out)
        self.assertIn("no file was actually created or changed", out)

    def test_false_download_claim_removed(self):
        out = honesty.post(ctx(), "Done. You can download your updated project file above.")
        self.assertNotIn("download", out)
        self.assertIn("no file was actually delivered", out)

    def test_false_run_and_ready_claims_removed(self):
        out = honesty.post(ctx(), "The tests passed. Your backend is now ready for deployment.")
        self.assertNotIn("ready for deployment", out)
        self.assertNotIn("tests passed", out.lower().replace("correction", ""))
        self.assertIn("Correction:", out)

    def test_true_claims_kept(self):
        c = ctx(tools_ok=["workspace_write", "workspace_run"], file_results=[{"filename": "a.py"}],
                code_results=[{"stdout": "ok"}])
        text = "I created the file. The tests passed. Download it above."
        self.assertEqual(honesty.post(c, text), text)

    def test_code_blocks_untouched(self):
        text = "Example:\n```python\n# I have created the file\nprint('tests passed')\n```"
        self.assertEqual(honesty.post(ctx(), text), text)

    def test_normal_answer_untouched(self):
        text = "Lagos is the largest city in Nigeria."
        self.assertEqual(honesty.post(ctx(), text), text)


class Brevity(unittest.TestCase):
    def test_generic_closer_removed(self):
        out = brevity.post(ctx(), "The fix is in main.py.\n\nIs there anything else you need to configure?")
        self.assertEqual(out, "The fix is in main.py.")

    def test_specific_followup_kept(self):
        text = "Done.\n\nWant me to add the giving page next?"
        self.assertEqual(brevity.post(ctx(), text), text)


class ClarifyFirst(unittest.TestCase):
    def test_vague_build_triggers(self):
        for p in ["create website", "build me an app", "make a church website", "please build a landing page"]:
            self.assertTrue(clarify_first.pre(ctx(prompt=p)), p)

    def test_detailed_or_normal_does_not(self):
        for p in ["build a church website with a prayer form and sermon page",
                  "why is my website slow", "yooo", "create a python script that renames files"]:
            self.assertFalse(clarify_first.pre(ctx(prompt=p)), p)


class SavedDocsAutoSearchRemoved(unittest.TestCase):
    def test_classifier_has_no_user_docs_type(self):
        # gpt2_functions needs Firebase/API keys at import, so check the source.
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "gpt2_functions.py"), encoding="utf-8") as f:
            src = f.read()
        start = src.index("def _fallback_intent")
        end = src.index("def classify_intent")
        self.assertNotIn("user_docs", src[start:end])
        self.assertNotIn('"user_docs", "none"', src)
        from prompts import INTENT_SYSTEM_PROMPT
        self.assertNotIn("user_docs", INTENT_SYSTEM_PROMPT)

    def test_no_user_docs_branch_left_in_flow(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "gpt2_test.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn('search_type"] == "user_docs"', src)
        self.assertNotIn("Checking your saved files", src)


if __name__ == "__main__":
    unittest.main()
