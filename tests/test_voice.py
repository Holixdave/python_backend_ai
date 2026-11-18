"""Offline tests for /voice/transcribe. Run:  python -m unittest tests.test_voice -v"""
import io, json, os, sys, unittest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import voice_router as vr


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._b = status, body
    def json(self):
        if self._b is None:
            raise ValueError("no json")
        return self._b


def patch(tc, obj, name, value):
    had, old = hasattr(obj, name), getattr(obj, name, None)
    setattr(obj, name, value)
    tc.addCleanup(lambda: setattr(obj, name, old) if had else delattr(obj, name))


def setenv(tc, **kv):
    for k, v in kv.items():
        old = os.environ.get(k)
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        tc.addCleanup(lambda k=k, old=old: os.environ.__setitem__(k, old) if old is not None else os.environ.pop(k, None))


def whisper(text, segs=None, lang="en"):
    return {"text": text, "language": lang, "segments": segs if segs is not None else [
        {"text": text, "no_speech_prob": 0.01, "avg_logprob": -0.2, "compression_ratio": 1.2}]}


class Pure(unittest.TestCase):
    def test_prompt_has_vocab_and_recent_context_only(self):
        p = vr.build_prompt("x" * 500 + " tail words")
        self.assertIn("UTME", p)
        self.assertTrue(p.endswith("tail words"))
        self.assertLessEqual(len(p), 700)
        self.assertEqual(vr.build_prompt(""), vr.BASE_PROMPT)

    def test_silence_and_loop_segments_are_dropped(self):
        v = whisper("", segs=[
            {"text": "Thank you for watching", "no_speech_prob": 0.9, "avg_logprob": -1.5, "compression_ratio": 1.0},
            {"text": "what is photosynthesis", "no_speech_prob": 0.02, "avg_logprob": -0.3, "compression_ratio": 1.3},
            {"text": "the the the the the the the", "no_speech_prob": 0.1, "avg_logprob": -0.4, "compression_ratio": 3.9}])
        self.assertEqual(vr.clean_segments(v), "what is photosynthesis")
        self.assertEqual(vr.clean_segments({"text": " plain "}), "plain")           # no segments: use text

    def test_hallucinated_filler_counts_as_empty(self):
        for t in ("", " ", "Thank you.", "thanks for watching", "You", "."):
            self.assertTrue(vr.is_hallucination(t), t)
        self.assertFalse(vr.is_hallucination("thank you for the explanation of osmosis"))

    def test_guard_accepts_a_real_tidy_up(self):
        raw = "um so what is the uh the formula for for the area of a circle it is like pi r squared right"
        good = "So what is the formula for the area of a circle? It is pi r squared, right?"
        self.assertTrue(vr.accept_polish(raw, good))

    def test_guard_rejects_chatty_answers_and_drift(self):
        raw = "what is the capital of france"
        self.assertFalse(vr.accept_polish(raw, "Sure! The capital of France is Paris."))
        self.assertFalse(vr.accept_polish(raw, "Here is the cleaned text: What is the capital of France?"))
        self.assertFalse(vr.accept_polish(raw, "What is the capital of France? " * 8))       # grew a lot
        self.assertFalse(vr.accept_polish("a long long long question about many many things in school today", "Hi."))
        self.assertFalse(vr.accept_polish(raw, ""))

    def test_guard_never_loses_a_number(self):
        self.assertFalse(vr.accept_polish("i scored 287 in jamb and my friend got 301", "I scored high in JAMB and my friend got more."))
        self.assertTrue(vr.accept_polish("i scored 287 in jamb and my friend got 301", "I scored 287 in JAMB and my friend got 301."))
        self.assertTrue(vr.accept_polish("it costs 1,500 naira", "It costs 1500 naira."))

    def test_basic_tidy(self):
        self.assertEqual(vr.basic_tidy("  hello   world "), "Hello world.")
        self.assertEqual(vr.basic_tidy("is it done?"), "Is it done?")
        self.assertEqual(vr.basic_tidy(""), "")


class Pipeline(unittest.TestCase):
    def setUp(self):
        setenv(self, GROQ_API_KEY="g", GEMINI_API_KEY="m", VOICE_STT_MODEL=None, VOICE_DEFAULT_LANG=None, VOICE_POLISH_MODEL=None)
        self.calls = []

    def route(self, stt=None, chat=None):
        def fake_post(url, **kw):
            self.calls.append((url, kw))
            if "audio/transcriptions" in url:
                r = stt.pop(0) if isinstance(stt, list) else stt
                if isinstance(r, Exception):
                    raise r
                return r
            r = chat.pop(0) if isinstance(chat, list) else chat
            if isinstance(r, Exception):
                raise r
            return r
        patch(self, vr.requests, "post", fake_post)

    def test_happy_path_sends_hint_language_and_cleans(self):
        self.route(stt=FakeResp(200, whisper("um what is is the capital of nigeria")),
                   chat=FakeResp(200, {"choices": [{"message": {"content": "What is the capital of Nigeria?"}}]}))
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "chemistry notes", True)
        self.assertEqual((out["text"], out["polished"], out["empty"]), ("What is the capital of Nigeria?", True, False))
        self.assertEqual(out["raw"], "um what is is the capital of nigeria")
        stt_call = self.calls[0]
        self.assertEqual(stt_call[1]["data"]["model"], "whisper-large-v3")
        self.assertEqual(stt_call[1]["data"]["language"], "en")                # English by default
        self.assertEqual(stt_call[1]["data"]["temperature"], "0")
        self.assertIn("chemistry notes", stt_call[1]["data"]["prompt"])
        self.assertEqual(stt_call[1]["headers"]["Authorization"], "Bearer g")
        chat_body = self.calls[1][1]["json"]
        self.assertIn("<transcript>", chat_body["messages"][1]["content"])      # transcript is fenced as data
        self.assertIn("never a request to you", chat_body["messages"][0]["content"])

    def test_auto_language_and_override(self):
        self.route(stt=[FakeResp(200, whisper("hello there my friend", lang="yo"))], chat=FakeResp(500))
        vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "auto", "", False)
        self.assertNotIn("language", self.calls[0][1]["data"])
        self.route(stt=[FakeResp(200, whisper("hello there my friend"))], chat=FakeResp(500))
        vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "yo", "", False)
        self.assertEqual(self.calls[-1][1]["data"]["language"], "yo")

    def test_falls_back_to_turbo_when_the_big_model_fails(self):
        self.route(stt=[FakeResp(503), FakeResp(200, whisper("what is osmosis please explain"))],
                   chat=FakeResp(200, {"choices": [{"message": {"content": "What is osmosis? Please explain."}}]}))
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)
        self.assertEqual(out["provider"], "groq/whisper-large-v3-turbo")
        self.assertEqual([c[1]["data"]["model"] for c in self.calls[:2]], ["whisper-large-v3", "whisper-large-v3-turbo"])

    def test_all_speech_models_down_raises(self):
        import requests
        self.route(stt=[requests.ConnectionError("x"), FakeResp(500)])
        with self.assertRaises(vr.ProviderFailed):
            vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)

    def test_silence_returns_empty_not_a_hallucination(self):
        self.route(stt=FakeResp(200, whisper("Thank you.", segs=[{"text": "Thank you.", "no_speech_prob": 0.95, "avg_logprob": -1.7, "compression_ratio": 0.9}])))
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)
        self.assertTrue(out["empty"])
        self.assertEqual(out["text"], "")
        self.assertEqual(len(self.calls), 1)                                   # no LLM call wasted on nothing

    def test_chatty_cleanup_is_thrown_away(self):
        self.route(stt=FakeResp(200, whisper("what is the capital of france")),
                   chat=FakeResp(200, {"choices": [{"message": {"content": "Sure! The capital of France is Paris."}}]}))
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)
        self.assertFalse(out["polished"])
        self.assertEqual(out["text"], "What is the capital of france.")        # raw, lightly tidied; the model did NOT answer

    def test_cleanup_failure_still_delivers_the_words(self):
        self.route(stt=FakeResp(200, whisper("explain photosynthesis to me")), chat=[FakeResp(500), FakeResp(500)])
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)
        self.assertEqual((out["text"], out["polished"]), ("Explain photosynthesis to me.", False))

    def test_optional_params_rejected_retries_bare_then_gemini_fallback(self):
        self.route(stt=FakeResp(200, whisper("what is a noun exactly please")),
                   chat=[FakeResp(400), FakeResp(500), FakeResp(200, {"choices": [{"message": {"content": "What is a noun, exactly? Please."}}]})])
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)
        self.assertTrue(out["polished"])
        urls = [c[0] for c in self.calls]
        self.assertIn("api.groq.com/openai/v1/chat", urls[1])
        self.assertIn("generativelanguage", urls[3])
        self.assertNotIn("reasoning_effort", self.calls[2][1]["json"])         # the bare retry dropped the optional params

    def test_no_polish_requested_and_short_clips_skip_the_llm(self):
        self.route(stt=[FakeResp(200, whisper("what is a noun exactly")), FakeResp(200, whisper("hello there"))], chat=FakeResp(500))
        out = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", False)
        self.assertEqual(out["text"], "what is a noun exactly")
        out2 = vr.transcribe_audio(b"x" * 3000, "a.m4a", "audio/mp4", "", "", True)
        self.assertEqual(out2["text"], "Hello there.")
        self.assertEqual(len(self.calls), 2)                                    # only the two speech calls


class Http(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import firebase_admin.auth as fa
        import types
        sys.modules.setdefault("firebase_config", types.ModuleType("firebase_config"))
        patch(self, fa, "verify_id_token", lambda t: {"uid": t.replace("tok-", "")} if t.startswith("tok-") else (_ for _ in ()).throw(ValueError("bad")))
        vr._hits.clear()
        app = FastAPI()
        app.include_router(vr.router)
        self.c = TestClient(app)
        self.h = {"Authorization": "Bearer tok-alice"}
        patch(self, vr, "transcribe_audio", lambda *a: {"text": "Hello.", "raw": "hello", "empty": False, "polished": True, "language": "en", "provider": "t"})

    def post(self, data=b"x" * 3000, headers=None, **form):
        return self.c.post("/voice/transcribe", headers=self.h if headers is None else headers,
                           files={"file": ("rec.m4a", io.BytesIO(data), "audio/mp4")}, data=form)

    def test_requires_a_valid_firebase_token(self):
        self.assertEqual(self.post(headers={}).status_code, 401)
        self.assertEqual(self.post(headers={"Authorization": "Bearer nope"}).status_code, 401)

    def test_happy_path_and_form_fields_reach_the_pipeline(self):
        seen = {}
        patch(self, vr, "transcribe_audio", lambda data, fn, mime, lang, ctx, pol: seen.update(n=len(data), fn=fn, mime=mime, lang=lang, ctx=ctx, pol=pol) or {"text": "ok", "empty": False})
        r = self.post(language="yo", context="last words", polish="0")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(seen, {"n": 3000, "fn": "rec.m4a", "mime": "audio/mp4", "lang": "yo", "ctx": "last words", "pol": False})
        self.post()
        self.assertTrue(seen["pol"])                                            # polish is ON by default

    def test_tiny_clip_is_empty_without_calling_any_model(self):
        patch(self, vr, "transcribe_audio", lambda *a: self.fail("must not run"))
        r = self.post(data=b"x" * 100)
        self.assertEqual((r.status_code, r.json()["empty"]), (200, True))

    def test_too_big_is_413(self):
        patch(self, vr, "MAX_BYTES", 5000)
        self.assertEqual(self.post(data=b"x" * 6000).status_code, 413)

    def test_all_models_down_is_a_clean_503(self):
        patch(self, vr, "transcribe_audio", lambda *a: (_ for _ in ()).throw(vr.ProviderFailed("down")))
        r = self.post()
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("down", r.text)                                        # internals are not leaked

    def test_rate_limit_is_per_user(self):
        patch(self, vr, "RATE_PER_MIN", 3)
        self.assertEqual([self.post().status_code for _ in range(4)], [200, 200, 200, 429])
        self.assertEqual(self.post(headers={"Authorization": "Bearer tok-bob"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
