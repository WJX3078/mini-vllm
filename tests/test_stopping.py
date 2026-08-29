"""Incremental stop checking (P1): StopChecker unit tests.

Fast path (pre-encoded token sequences) and fallback (bounded-window text
decode) must agree on when generation terminates, cut BEFORE the stop
string, keep EOS as the final token, and never re-decode the whole history
(window is bounded by the longest stop sequence, not the output length).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from minivllm.stopping import StopChecker


class CharTok:
    """Character-level stub: encode splits chars, decode joins them."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_eos_kept_as_final_token():
    ck = StopChecker(None, [], eos_token_ids=[0])
    assert ck.check([5, 6, 0], ignore_eos=False) == 3
    assert ck.check([5, 6, 0], ignore_eos=True) is None
    assert ck.check([5, 6, 7], ignore_eos=False) is None


def test_stop_token_sequence_suffix_fast_path():
    ck = StopChecker(CharTok(), ["XYZ"], [])
    # generation ends exactly when the encoded stop completes as a suffix
    assert ck.check(list(b"abcXY"), ignore_eos=True) is None
    cut = ck.check(list(b"abcXYZ"), ignore_eos=True)
    assert cut == len(b"abc")            # keep everything before the stop


def test_stop_string_split_across_many_tokens():
    """A 5-char stop string completes token by token; the text fallback must
    catch it and cut before its first character."""
    ck = StopChecker(CharTok(), ["hello"], [])
    ids = list(b"say hello")
    cut = ck.check(ids, ignore_eos=True)
    assert cut == len(b"say ")


def test_no_false_positive_from_window_edge():
    ck = StopChecker(CharTok(), ["zzz"], [])
    assert ck.check(list(b"azazaza"), ignore_eos=True) is None


def test_window_is_bounded():
    """The decode window scales with the stop length, NOT with the history:
    a 10k-token history checks only the tail."""
    ck = StopChecker(CharTok(), ["STOP"], [])
    assert ck.window < 32
    long_history = list(b"A" * 10_000) + list(b"STOP")
    cut = ck.check(long_history, ignore_eos=True)
    assert cut == 10_000


def test_earliest_cut_wins():
    class MultiTok(CharTok):
        pass
    ck = StopChecker(MultiTok(), ["ab", "bcd"], [])
    ids = list(b"xab")            # "ab" completes before "bcd" could
    cut = ck.check(ids, ignore_eos=True)
    assert cut == 1


def test_none_tokenizer_disables_stop_strings():
    ck = StopChecker(None, ["stop"], eos_token_ids=[9])
    assert ck.stop_token_seqs == []
    assert ck.check([1, 2, 9], ignore_eos=False) == 3
    assert ck.check([1, 2, 3], ignore_eos=False) is None


def test_engine_integration_incremental_stop():
    """End-to-end on the plain engine: a stop string stops generation at the
    right token count and trims the text."""
    from helpers import make_tiny_engine, random_prompts

    class Tok:
        """token id -> single char; ids are chars so decode is exact."""
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) if 0 <= i < 0x110000 else "?" for i in ids)

    eng, _ = make_tiny_engine(seed=0)
    eng.tokenizer = Tok()
    prompt = random_prompts(1, min_len=8, max_len=8, seed=61)[0]
    params_stops = ["END"]
    from minivllm.sequence import SamplingParams
    p = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True,
                       stop=params_stops)
    out = eng.generate([prompt], p, use_tqdm=False)[0]
    ids = out.outputs[0]["token_ids"]
    # generation must finish well before max_tokens only if the stop string
    # actually occurred; assert the invariant: no stop string in final text,
    # and if we stopped early the stop token seq was completed then removed
    text = out.outputs[0]["text"]
    assert "END" not in text
    # either the stop string never appeared (length cap) or it was cut
    assert len(ids) <= 64
    # regenerate with EOS ignored and find where END would complete
    p2 = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    out2 = eng.generate([prompt], p2, use_tqdm=False)[0]
    joined2 = Tok().decode(out2.outputs[0]["token_ids"])
    if "END" in joined2:
        assert ids == out2.outputs[0]["token_ids"][:joined2.index("END") + 1]
        assert len(ids) < 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
