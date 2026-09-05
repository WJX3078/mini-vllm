"""Incremental detokenization for streaming (v0.4).

Streaming must not re-decode the whole output every token (O(T^2) tokenizer
work), and naive "decode each new token alone" is wrong for byte-level BPE:
a multi-byte UTF-8 character spans 2-4 tokens and a text delta may be
invalid until its last byte arrives.

Algorithm (bounded window, stable-boundary verification):

* state: `emitted_tokens` (tokens whose text has been fully emitted),
  `emitted_chars`, and a sliding window start `ws` with
  `ws_head_chars = len(decode(ids[:ws]))`.
* on push: everything except the LAST token is considered closed. Its text
  is computed on a small window `decode(ids[ws:closed])`; the delta is the
  slice after the already-emitted chars. The window slides forward only
  when the boundary is verified stable (`decode(ids[new_ws:])` starts with
  the head we would drop), so a BPE merge across the boundary can never
  corrupt output.
* the final push flushes everything, including the held-back last token.

Tests cover ASCII, CJK (3-byte), emoji (4-byte), split-surrogate-free
replacement chars, and byte-level stub tokenizers.
"""
from __future__ import annotations


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._emitted_tokens = 0          # tokens fully emitted
        self._emitted_chars = 0           # chars emitted so far
        self._ws = 0                      # window start (token index)
        self._ws_head_chars = 0           # len(decode(ids[:ws]))
        self._finished = False

    def push(self, new_token_ids: list[int], final: bool = False) -> str:
        """Feed newly generated tokens; return the newly decoded text."""
        if self._finished:
            return ""
        self._ids.extend(new_token_ids)
        total = len(self._ids)
        if total == 0:
            return ""
        if final:
            text = self._decode(self._ws, total)
            delta = text[self._emitted_chars - self._ws_head_chars:] \
                if self._emitted_chars >= self._ws_head_chars else text
            delta = text[max(0, self._emitted_chars - self._ws_head_chars):]
            self._emitted_chars = self._ws_head_chars + len(text)
            self._emitted_tokens = total
            self._finished = True
            return delta

        closed = total - 1                # hold back the last token
        if closed <= self._emitted_tokens:
            return ""
        self._maybe_slide_window(closed)
        window = self._decode(self._ws, closed)
        start = self._emitted_chars - self._ws_head_chars
        if len(window) < start:
            return ""                     # boundary not stable yet: hold
        delta = window[start:]
        if delta.endswith("\ufffd"):
            # incomplete multi-byte character: hold the tail
            delta = delta[:-1]
            if not delta:
                return ""
        self._emitted_chars += len(delta)
        self._emitted_tokens = closed
        return delta

    # ------------------------------------------------------------------ internals
    def _decode(self, start: int, stop: int) -> str:
        return self._tokenizer.decode(self._ids[start:stop])

    def _maybe_slide_window(self, closed: int):
        """Slide the window start forward while (a) >= 48 chars are already
        emitted past it and (b) the boundary is verified stable against a
        bounded lookahead (BPE merges are local in practice)."""
        while self._ws < self._emitted_tokens and \
                self._emitted_chars - self._ws_head_chars >= 48 and \
                self._ws + 1 <= closed:
            head = self._decode(self._ws, self._ws + 1)
            rest = self._decode(self._ws + 1, min(closed + 8, len(self._ids)))
            if not rest.startswith(head):
                break                     # merge across the boundary: stop
            self._ws_head_chars += len(head)
            self._ws += 1
