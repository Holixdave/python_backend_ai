"""Offline tests for the search-query gate. Run:  python -m unittest discover -s tests -v"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_query import build_search_query, is_meaningful_query, needs_web_search


class SearchGate(unittest.TestCase):
    def test_real_lookups_pass(self):
        for q in ["dollar to naira exchange rate today", "who won the match", "bitcoin price",
                  "Eagles vs Chelsea score", "jamb 2026 registration date"]:
            self.assertTrue(is_meaningful_query(q), q)

    def test_statements_and_filler_rejected(self):
        for q in ["", "hmm", "ok thanks that is good", "i think you are wrong about that",
                  "you are doing it again man", "the thing about it that you said",
                  "can you explain why the function returns none after the loop ends",
                  "model searching rubbish stuff when it does not know"]:
            self.assertFalse(is_meaningful_query(q), q)

    def test_long_pasted_sentence_rejected(self):
        s = "this is a very long sentence the user typed about nothing in particular at all"
        self.assertFalse(is_meaningful_query(s, s))

    def test_filler_removal_keeps_real_words(self):
        # regression: plain str.replace("man","") used to produce "hu"/"chester"
        self.assertEqual(build_search_query("dude can you find me human resources manager in Manchester abeg"),
                         "human resources manager in manchester")
        self.assertEqual(build_search_query("search for tuna shake recipe pls"), "tuna shake recipe")

    def test_keyword_fallback_not_fired_by_code_chat(self):
        self.assertFalse(needs_web_search("find the bug in my code"))
        self.assertFalse(needs_web_search("i think you should search sometimes"))
        self.assertTrue(needs_web_search("latest news on tinubu"))


if __name__ == "__main__":
    unittest.main()
