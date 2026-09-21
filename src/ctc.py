"""CTC forced alignment: where in the audio does each character of a known text sit.

Used by the admin text input (app.py), where the acoustic transcript is
supplied instead of decoded: Wav2Vec2 still runs on the audio, and this
Viterbi pass places the given characters on its frames, so the waveform
markers land where the sounds actually are.

Pure Python over a small matrix (frames × distinct tokens), so it is
testable without torch.
"""


def forced_align(log_probs, tokens, blank):
    """Viterbi CTC alignment → [(first_frame, last_frame + 1)] per token.

    ``log_probs[t][k]`` is the log-probability of token id ``k`` at frame
    ``t`` (any indexable rows; only the ids in ``tokens`` and ``blank`` are
    read). Raises ValueError if the audio has too few frames for the text.
    """
    n_frames = len(log_probs)
    ext = [blank]
    for tok in tokens:
        ext += [tok, blank]
    n_states = len(ext)
    need = len(tokens) + sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    if not tokens or n_frames < need:
        raise ValueError(f"{n_frames} frames cannot hold {len(tokens)} tokens")

    neg = float("-inf")
    score = [neg] * n_states
    score[0] = log_probs[0][ext[0]]
    if n_states > 1:
        score[1] = log_probs[0][ext[1]]
    back = []
    for t in range(1, n_frames):
        row = log_probs[t]
        new, choice = [neg] * n_states, [0] * n_states
        for s in range(n_states):
            best, arg = score[s], s
            if s >= 1 and score[s - 1] > best:
                best, arg = score[s - 1], s - 1
            if (s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]
                    and score[s - 2] > best):
                best, arg = score[s - 2], s - 2
            if best > neg:
                new[s] = best + row[ext[s]]
                choice[s] = arg
        score = new
        back.append(choice)

    # end in the last token or the trailing blank
    s = n_states - 1 if score[-1] >= score[-2] else n_states - 2
    path = [s]
    for choice in reversed(back):
        s = choice[s]
        path.append(s)
    path.reverse()

    spans = [None] * len(tokens)
    for t, s in enumerate(path):
        if s % 2:                        # odd states are tokens
            i = s // 2
            spans[i] = (spans[i][0], t + 1) if spans[i] else (t, t + 1)
    return spans
