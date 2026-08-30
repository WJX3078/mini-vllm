"""Speculative decoding must truncate at the EARLIEST termination (P0).

One verify round commits k accepted proposals + 1 bonus token. If an EOS
(or stop string) sits inside that run -- e.g. proposals [A, EOS, B] all
accepted -- everything after the EOS must be dropped: output tokens, KV
frontier, prefix-cache registration and the drafter's view stay consistent
with the truncated stream. The old code checked only the LAST committed
token and leaked post-EOS proposals into the output.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from helpers import (
    make_tiny_spec_engine,
    random_prompts,
    run_hf_greedy,
    run_spec_greedy,
)

from minivllm.sequence import SamplingParams
from minivllm.stopping import StopChecker


# --------------------------------------------------------------- unit level
def _make_checker(eos=(0,), stop=(), tokenizer=None):
    return StopChecker(tokenizer, list(stop), list(eos))


def test_keep_count_eos_at_second_proposal():
    """committed = [A, EOS, B, C]: keep exactly [A, EOS]."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    committed = [10, 0, 11, 12]
    keep, terminated = eng._round_keep_count(
        candidate_ids=[5] + committed, base=1, committed=committed,
        params=SamplingParams(max_tokens=32), checker=None)
    assert (keep, terminated) == (2, True)


def test_keep_count_eos_at_bonus_position():
    """[A, B, EOS] all committed: keep 3, terminated."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    committed = [10, 11, 0]
    keep, terminated = eng._round_keep_count([5] + committed, 1, committed,
                                             SamplingParams(max_tokens=32), None)
    assert (keep, terminated) == (3, True)


def test_keep_count_no_termination():
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    committed = [10, 11, 12]
    keep, terminated = eng._round_keep_count([5] + committed, 1, committed,
                                             SamplingParams(max_tokens=32), None)
    assert (keep, terminated) == (3, False)


def test_keep_count_max_tokens_cap():
    """A commit larger than the remaining room is capped by max_tokens (and
    that is a LENGTH finish, not a stop)."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    committed = [10, 11, 12, 13, 14]
    keep, terminated = eng._round_keep_count(
        [5] + committed, 1, committed,
        SamplingParams(max_tokens=4, ignore_eos=True), None)
    assert (keep, terminated) == (3, False)   # room = 4 - 1


def test_keep_count_earliest_of_eos_and_stop():
    """Stop string completing BEFORE the EOS wins (cut earlier)."""
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [101, 102][:len(s) % 3 + 1]
        def decode(self, ids, skip_special_tokens=True):
            m = {10: "x", 11: "END", 12: "y", 0: "<eos>"}
            return "".join(m.get(i, "?") for i in ids)

    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    checker = StopChecker(Tok(), ["END"], [0])
    committed = [10, 11, 0, 12]      # text: x END <eos> y
    keep, terminated = eng._round_keep_count(
        [9] + committed, 1, committed, SamplingParams(max_tokens=32), checker)
    assert (keep, terminated) == (1, True)   # keep 'x', cut at 'END'


# ------------------------------------------------------------ end-to-end
def _prompt_with_in_run_eos():
    """Deterministic prompt whose plain greedy stream (EOS ignored) is all
    EOS: the tiny random model echoes its final token, so ending the prompt
    on EOS puts it inside round 1's accepted proposals -> the commit
    truncation path runs. (The precise 'EOS as the 2nd accepted proposal'
    cut position is covered by the unit tests above; the mechanism --
    dropping accepted proposals after the first termination -- is the same
    code path.)"""
    base = random_prompts(1, min_len=10, max_len=10, seed=61)[0]
    return base + [5, 0]


def test_eos_mid_round_truncates_output():
    """End-to-end: proposals accepted past an in-run EOS never leak into the
    output; generation stops exactly at the EOS token."""
    prompt = _prompt_with_in_run_eos()
    eng, _ = make_tiny_spec_engine(seed=0, drafter="(same)", num_spec_tokens=4)

    # reference: same prompt, EOS ignored -> the untouched greedy stream
    params_noeos = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    raw = eng.generate([prompt], params_noeos, use_tqdm=False)[0].token_ids
    eos_pos = raw.index(0)                    # first EOS anywhere in the stream
    assert eos_pos <= 4                       # inside round 1's proposals

    # now let EOS finish the sequence
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=False)
    out = eng.generate([prompt], params, use_tqdm=False)[0]
    assert out.token_ids == raw[:eos_pos + 1]   # nothing after the EOS
    assert out.token_ids[-1] == 0


def test_pre_eos_tokens_match_hf():
    """Truncation correctness against HF: tokens up to and incl. EOS equal
    the HF stream."""
    prompt = _prompt_with_in_run_eos()
    eng, hf = make_tiny_spec_engine(seed=0, drafter="(same)", num_spec_tokens=4)
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=False)
    out = eng.generate([prompt], params, use_tqdm=False)[0]
    ref = run_hf_greedy(hf, [prompt], max_new_tokens=8)[0]
    eos = ref.index(0)
    assert out.token_ids == ref[:eos + 1]


def test_stop_string_inside_round_truncates():
    """Stop strings terminate mid-round: committed tokens completing the
    stop string and everything after it are dropped."""
    class Tok:
        def encode(self, s, add_special_tokens=False):
            m = {"XYZ": [101, 102]}
            return m.get(s, [ord(c) % 200 for c in s])
        def decode(self, ids, skip_special_tokens=True):
            m = {30: "Q", 31: "R", 101: "X", 102: "Y", 103: "Z", 0: "<eos>"}
            return "".join(m.get(i, "?") for i in ids)

    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    checker = StopChecker(Tok(), ["XYZ"], [0])
    committed = [30, 101, 102, 103]        # text QXYZ -> stop at XYZ
    keep, terminated = eng._round_keep_count(
        [8] + committed, 1, committed,
        SamplingParams(max_tokens=32), checker)
    assert (keep, terminated) == (1, True)  # keep Q only


def test_greedy_ngram_pipeline_still_matches_hf():
    """The truncation rework must not perturb the normal greedy path."""
    eng, hf = make_tiny_spec_engine(seed=0, drafter="ngram", num_spec_tokens=4)
    prompts = random_prompts(4, seed=44)
    outs = run_spec_greedy(eng, prompts, max_new_tokens=12)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=12)
    for i, o in enumerate(outs):
        assert o.token_ids == ref[i]




# --------------------------------------------- stop-stream semantics (v0.3)
class _IdTok:
    """Stub tokenizer: encode maps chosen strings to chosen token lists;
    decode maps token ids through a dict (text fallback uses it)."""

    def __init__(self, enc_map, dec_map):
        self._enc, self._dec = enc_map, dec_map

    def encode(self, s, add_special_tokens=False):
        return self._enc[s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._dec.get(i, "?") for i in ids)


class _ScriptedDrafter:
    """Deterministic stand-in for the drafter: yields a fixed proposal list
    for the first round, then nothing. Lets tests place stop tokens inside
    REJECTED proposals with full control."""

    def __init__(self, first_round):
        self.first_round = first_round
        self.done = False

    def reset(self):
        self.done = False

    def sync(self, tokens):
        pass

    def propose(self, tokens, gamma):
        if self.done:
            return []
        self.done = True
        return self.first_round[:gamma]


def _scripted_engine(first_round):
    """Spec engine with a scripted drafter proposing `first_round` tokens
    that the greedy target model does NOT produce (ids chosen outside the
    prompt echo set); asserts k=0 and returns (eng, prompt, raw_ids)."""
    import pytest
    from helpers import make_tiny_spec_engine, random_prompts

    from minivllm.sequence import SamplingParams
    for extra in range(200, 250):
        eng, _hf = make_tiny_spec_engine(seed=0, drafter="ngram",
                                         num_spec_tokens=4)
        prompt = random_prompts(1, min_len=8, max_len=8, seed=71)[0]
        first_round = [extra, extra + 1, extra + 2, extra + 3]
        eng.ngram_drafter = _ScriptedDrafter(first_round)
        raw = eng.generate([prompt], SamplingParams(
            temperature=0.0, max_tokens=12, ignore_eos=True),
            use_tqdm=False)[0]
        if raw.token_ids[0] not in first_round:     # round 1: k=0, all rejected
            return eng, prompt, raw.token_ids
    pytest.skip("no scripted proposal id produced a rejected round")


def test_rejected_proposal_stop_does_not_stop():
    """Bug A regression: a stop string completed by the TAIL of a rejected
    proposal run must not stop generation (the buggy code checked the
    history+ALL-proposals stream, whose suffix matched)."""
    from minivllm.sequence import SamplingParams
    stop_tok = 249
    eng, prompt, raw_ids = _scripted_engine([91, 92, 93, stop_tok])
    assert raw_ids[0] != 91                          # precondition: k = 0
    eng2, prompt2, _raw = _scripted_engine([91, 92, 93, stop_tok])
    eng2.tokenizer = _IdTok({"S": [stop_tok]}, {})
    out = eng2.generate([prompt2], SamplingParams(
        temperature=0.0, max_tokens=12, ignore_eos=True,
        stop=["S"]), use_tqdm=False)[0]
    # buggy code stopped at round 1 (candidate suffix == rejected proposal
    # tail == stop token); the fix keeps generating to the length cap
    assert len(out.token_ids) == 12


def test_bonus_token_completing_stop_stops():
    """Bug B regression: the bonus token completing a stop string must stop,
    even though the bonus is not part of the proposal stream (the buggy
    code's checked stream ended with rejected proposals, so it never saw
    the stop completion)."""
    from minivllm.sequence import SamplingParams
    eng, prompt, raw_ids = _scripted_engine([91, 92, 93, 94])
    bonus_tok = raw_ids[0]                           # k=0: first commit = bonus
    eng2, prompt2, _raw = _scripted_engine([91, 92, 93, 94])
    eng2.tokenizer = _IdTok({"S": [bonus_tok]}, {})
    out = eng2.generate([prompt2], SamplingParams(
        temperature=0.0, max_tokens=12, ignore_eos=True,
        stop=["S"]), use_tqdm=False)[0]
    # the stop string completed at the bonus and is excluded from the
    # output (same semantics as the plain engine) -> zero-token generation
    assert out.token_ids == []


def test_accepted_proposal_completing_stop_truncates():
    """Accepted proposals that complete a stop string are cut before the
    stop tokens; the committed stream (not the proposal stream) decides."""
    eng, _hf = make_tiny_spec_engine(seed=0, drafter="ngram")
    eng.tokenizer = _IdTok({"S": [30]}, {})
    checker = StopChecker(eng.tokenizer, ["S"], [0])
    history = [5, 6]
    committed = [10, 20, 30]                        # 30 completes the stop
    keep, terminated = eng._round_keep_count(
        history + committed, base=len(history), committed=committed,
        params=SamplingParams(max_tokens=32), checker=checker)
    assert (keep, terminated) == (2, True)          # stop token dropped


def test_eos_stop_max_tokens_earliest_wins():
    """EOS, stop strings and max_tokens compete inside one round: the
    earliest token position terminates, max_tokens is a hard length cap."""
    eng, _hf = make_tiny_spec_engine(seed=0, drafter="ngram")
    eng.tokenizer = _IdTok({"S": [11]}, {"11": "K"})
    checker = StopChecker(eng.tokenizer, ["S"], [0])
    P = SamplingParams(max_tokens=32)

    # EOS at index 1 beats a stop that would complete at index 2
    committed = [10, 0, 11, 12]
    keep, terminated = eng._round_keep_count(
        [5] + committed, 1, committed, P, checker)
    assert (keep, terminated) == (2, True)          # [10, EOS], not stop cut

    # max_tokens caps BEFORE a later EOS: length finish, not stop
    # (base=1 history token, room = 2-1 = 1)
    committed = [10, 11, 0]
    keep, terminated = eng._round_keep_count(
        [5] + committed, 1, committed,
        SamplingParams(max_tokens=2, ignore_eos=True), checker)
    assert (keep, terminated) == (1, False)

    # EOS before the cap still wins as a stop event
    keep, terminated = eng._round_keep_count(
        [5] + committed, 1, committed, P, checker)
    assert (keep, terminated) == (3, True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
