"""'What did the script change?' diffs. Run: python -m unittest tests.test_filediff -v"""
import os, sys, tempfile, types, unittest, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("WORKSPACE_ROOT", os.path.join(tempfile.mkdtemp(prefix="fdtest-"), "ws"))
os.environ.setdefault("WORKSPACE_PERSIST", "1")
if "gpt2_functions" not in sys.modules:
    stub = types.ModuleType("gpt2_functions")
    stub._upload_bytes_to_supabase = lambda uid, name, data, ctype: f"https://fake/{uid}/{name}"
    sys.modules["gpt2_functions"] = stub
import gpt2_filediff as fd


class Diff(unittest.TestCase):
    def test_edit_shows_minus_plus_and_counts(self):
        a, r, d = fd.file_diff('{\n  "debug": true,\n  "port": 8080\n}\n', '{\n  "debug": false,\n  "port": 8080\n}\n')
        self.assertEqual((a, r), (1, 1))
        lines = d.split("\n")
        self.assertEqual(lines[0], "@@ line 1 @@")
        self.assertIn('-   "debug": true,', lines)
        self.assertIn('+   "debug": false,', lines)

    def test_new_file_is_all_plus_and_capped(self):
        a, r, d = fd.file_diff(None, "\n".join(f"l{i}" for i in range(100)))
        self.assertEqual((a, r), (100, 0))
        self.assertTrue(d.startswith("+ l0"))
        self.assertIn("75 more lines", d.split("\n")[-1])

    def test_far_apart_changes_become_separate_hunks(self):
        old = "\n".join(str(i) for i in range(50))
        new = old.replace("\n3\n", "\nX\n").replace("\n45\n", "\nY\n")
        _, _, d = fd.file_diff(old, new)
        self.assertEqual(d.count("@@ line"), 2)

    def test_binary_and_big_are_not_text(self):
        self.assertIsNone(fd.text_of_bytes(b"\x89PNG\x00\x01"))
        self.assertIsNone(fd.text_of_bytes(b"a" * (fd.MAX_TEXT_BYTES + 1)))
        self.assertIsNone(fd.text_of_bytes("é".encode("latin-1")))
        self.assertEqual(fd.text_of_bytes("héllo".encode()), "héllo")

    def test_format_block(self):
        ch = [fd.change_entry("a.py", "x = 1\n", "x = 2\n"), fd.change_entry("b.md", None, "# hi\n")]
        s = fd.format_changes(ch)
        self.assertEqual(s.split("\n")[0], fd.SEPARATOR)
        self.assertIn("=== a.py (edited)", s)
        self.assertIn("=== b.md (new)", s)
        self.assertIn("+ # hi", s)
        self.assertEqual(fd.format_changes([]), "")
        many = [fd.change_entry(f"f{i}.txt", None, "x") for i in range(9)]
        self.assertIn("and 3 more files", fd.format_changes(many))


class RunCodeOutputs(unittest.TestCase):
    def test_overwritten_output_gets_a_diff_and_new_one_is_new(self):
        import gpt2_workspace as w
        import gpt2_sandbox as sb
        uid = "fduser"
        w.workspace_write(uid, "outputs/report.md", "# Report\nold line\n")
        run_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(run_dir, "outputs"))
        open(os.path.join(run_dir, "outputs", "report.md"), "w").write("# Report\nnew line\n")
        open(os.path.join(run_dir, "outputs", "fresh.txt"), "w").write("hello\n")
        open(os.path.join(run_dir, "outputs", "pic.bin"), "wb").write(b"\x00\x01\x02")
        res = {e["path"]: e for e in sb._collect_outputs_into_workspace(run_dir, uid)}
        self.assertEqual(res["outputs/report.md"]["change"]["status"], "edited")
        self.assertIn("- old line", res["outputs/report.md"]["change"]["diff"])
        self.assertEqual(res["outputs/fresh.txt"]["change"]["status"], "new")
        self.assertNotIn("change", res["outputs/pic.bin"])


class Preview(unittest.TestCase):
    def test_run_code_and_workspace_run_previews_append_the_block(self):
        import gpt2_native as n
        change = fd.change_entry("cfg.json", '{"a": 1}\n', '{"a": 2}\n')
        rc = json.dumps({"success": True, "stdout": "done", "saved_to_workspace": [{"path": "cfg.json", "size": 9, "change": change}]})
        out = n.tool_output_preview("run_code", True, rc)
        self.assertTrue(out.startswith("done\n\n" + fd.SEPARATOR))
        self.assertIn("=== cfg.json (edited)", out)
        wr = json.dumps({"success": True, "stdout": "", "changes": [change]})
        out2 = n.tool_output_preview("workspace_run", True, wr)
        self.assertTrue(out2.startswith(fd.SEPARATOR))
        plain = json.dumps({"success": True, "stdout": "42", "saved_to_workspace": []})
        self.assertEqual(n.tool_output_preview("run_code", True, plain), "42")


if __name__ == "__main__":
    unittest.main()
