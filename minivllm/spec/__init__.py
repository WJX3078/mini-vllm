"""Speculative decoding: draft-then-verify at no quality cost.

Standard loop (Leviathan et al. 2023 / Chen et al. 2023):

  1. DRAFT: a cheap drafter (n-gram lookup or a small model) proposes
     `gamma` tokens autoregressively.
  2. VERIFY: the target model runs ONE forward over
     [last accepted token] + [gamma proposals] and produces a distribution
     at every position.
  3. ACCEPT: walk the proposals; greedy mode accepts while the target's
     argmax agrees; sampling mode uses the rejection-sampling rule
     accept with prob min(1, p(x)/q(x)), and on rejection resample from
     norm(max(p - q, 0)). The output distribution is *exactly* the target's.
  4. The token after the last accepted proposal (the "bonus") comes free
     from the same verify forward.

Each round yields k+1 tokens for one target forward (+ gamma cheap draft
forwards), so latency improves proportionally to the acceptance rate.

KV-cache bookkeeping (the tricky part, all handled here):
  * target KV covers the accepted prefix minus the pending bonus token;
  * draft KV is rewound/extended to match the accepted prefix each round;
  * rejected proposals leave stale slots in exclusive blocks that are simply
    overwritten by the next round -- blocks are only registered in the
    prefix cache up to the last *accepted* token, so garbage is never shared.
"""
