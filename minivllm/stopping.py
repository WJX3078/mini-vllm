"""Incremental stop-condition checking.

Checking stop strings by re-decoding a sequence's WHOLE output after every
token is O(T^2) tokenizer work over a generation. This module checks
termination using only a bounded tail of the stream:

* fast path: stop strings are pre-encoded to token-id sequences once; a
  generation terminates when ``output_ids[-len(stop):] == stop``.
* fallback: because tokenizers merge across token boundaries, a stop string
  may complete without any single token sequence matching. We decode only
  the last ``window`` tokens (window >= 2*max_stop_token_len, so any stop
  string shorter than the window completes strictly inside it) and search
  the text. Windowed decode can tokenize the left edge slightly differently
  from full-history decode; the generous window + "walk back until gone"
  truncation keeps this off the common path and bounded.

``check`` returns the number of output tokens to KEEP when generation should
stop (truncating everything from the earliest terminating token on), or
None to continue. Speculative decoding reuses the same helper so an EOS or
stop string in the middle of an accepted proposal run cuts the commit
exactly there -- tokens after termination never reach the output.
"""
from collections.abc import Sequence


class StopChecker:
    def __init__(self, tokenizer, stop_strings: Sequence[str],
                 eos_token_ids: Sequence[int] | None = None):
        self.tokenizer = tokenizer
        self.stop_strings = list(stop_strings)
        self.eos_token_ids = set(eos_token_ids or ())
        self.stop_token_seqs: list[list[int]] = []
        self.window = 8
        if tokenizer is not None and self.stop_strings:
            for s in self.stop_strings:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    self.stop_token_seqs.append(list(ids))
            self.window = 2 * max((len(x) for x in self.stop_token_seqs),
                                  default=0) + 4

    def _stop_cut_token_level(self, output_ids: list[int]) -> int | None:
        """Fast path: suffix match of pre-encoded stop sequences / EOS."""
        cut: int | None = None
        for ids in self.stop_token_seqs:
            n = len(ids)
            if len(output_ids) >= n and output_ids[-n:] == ids:
                cut = len(output_ids) - n if cut is None else min(cut, len(output_ids) - n)
        return cut

    def _stop_cut_text_level(self, output_ids: list[int]) -> int | None:
        """Fallback: decode a bounded window, locate the stop string's
        character position, and map it back to a token count so the output
        is cut BEFORE the stop string starts (vLLM text semantics)."""
        if self.tokenizer is None or not self.stop_strings:
            return None
        start = max(0, len(output_ids) - self.window)
        window_ids = output_ids[start:]
        text = self.tokenizer.decode(window_ids)
        hits = [text.find(s) for s in self.stop_strings]
        hits = [h for h in hits if h >= 0]
        if not hits:
            return None
        hit = min(hits)
        # keep the longest token prefix whose decoded text ends at or before
        # the stop string's first character (BPE prefixes can re-tokenize
        # slightly differently; the generous window keeps this exact for any
        # stop string shorter than the window in the common case)
        kept = 0
        for k in range(1, len(window_ids) + 1):
            if len(self.tokenizer.decode(window_ids[:k])) <= hit:
                kept = k
            else:
                break
        return start + kept

    def check(self, output_ids: list[int], ignore_eos: bool = False,
              eos_token_ids: Sequence[int] | None = None) -> int | None:
        """Terminate? Return how many output tokens to keep, else None.

        EOS terminates AFTER the EOS token itself (vLLM convention: the EOS
        token ends the sequence; here it is kept as the final token).
        Stop strings terminate BEFORE the tokens that complete them.
        """
        eos = self.eos_token_ids if eos_token_ids is None else set(eos_token_ids)
        if not ignore_eos and eos and output_ids[-1] in eos:
            return len(output_ids)
        cut = self._stop_cut_token_level(output_ids)
        if cut is None:
            cut = self._stop_cut_text_level(output_ids)
        return cut
