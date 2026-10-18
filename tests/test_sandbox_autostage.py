import os, tempfile, unittest, wave

os.environ.setdefault("WORKSPACE_ROOT", tempfile.mkdtemp())
import gpt2_workspace as w
import gpt2_sandbox as sb


def _final(gen):
    return [e for e in gen if e.get("type") == "code_result"][-1]


class AutoStage(unittest.TestCase):
    def setUp(self):
        root = w._user_root("autostage")
        os.makedirs(os.path.join(root, "outputs"), exist_ok=True)
        with wave.open(os.path.join(root, "outputs", "tone.wav"), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(b"\x01\x00" * 800)

    CODE = ("import wave\n"
            "with wave.open('{src}','rb') as a:\n    p=a.getparams(); fr=a.readframes(a.getnframes())\n"
            "with wave.open('outputs/new.wav','wb') as o:\n    o.setparams(p); o.writeframes(fr+b'\\x00\\x00'*1600)\n")

    def test_workspace_path_without_input_files(self):
        r = _final(sb.run_code(self.CODE.format(src="outputs/tone.wav"), userid="autostage"))
        self.assertTrue(r["success"], r["stderr"])
        self.assertEqual([f["path"] for f in r["saved_to_workspace"]], ["outputs/new.wav"])

    def test_bare_name_found_in_workspace(self):
        r = _final(sb.run_code(self.CODE.format(src="tone.wav"), userid="autostage"))
        self.assertTrue(r["success"], r["stderr"])

    def test_unchanged_copy_not_resaved(self):
        r = _final(sb.run_code("print(open('outputs/tone.wav','rb').read()[:4])", userid="autostage"))
        self.assertEqual(r["saved_to_workspace"], [])



class StrayAndDeliver(unittest.TestCase):
    def test_file_written_next_to_script_is_kept_and_deliverable_by_bare_name(self):
        r = _final(sb.run_code("open('made.txt','w').write('hi')", userid="stray"))
        self.assertTrue(r["success"])
        self.assertEqual([f["path"] for f in r["saved_to_workspace"]], ["outputs/made.txt"])
        import gpt2_output as go
        self.assertTrue(os.path.isfile(go._find_workspace_file("stray", "made.txt")))


if __name__ == "__main__":
    unittest.main()
