# Demo take data (`docs/assets/demo_take.json`)

One analysed utterance — the README demo, sentence 8 — exported as data so
that a page can draw it without hand transcription. The screenshots listed
in `source.screenshots` show the same stored record.

```bash
python tools/export_demo_take.py --record 9 --out docs/assets/demo_take.json
```

The exporter calls no model and no API. It reads the stored ASR outputs,
recomputes everything else deterministically, and exits without writing if
the score, surface form, IPA or error tags differ from what was stored.

## Schema `phonemejp2kr.demo_take/1`

| Field | Meaning |
|---|---|
| `generated_at` | UTC time of export |
| `source.record_id`, `source.recorded_at` | the history record the take comes from |
| `source.screenshots` | captures of the same record |
| `source.input` | `"recording"`, or for the admin text input `{tts_text, voice, acoustic_text}` — the script, the Edge TTS voice, and the transcript placed on the audio by forced alignment |
| `versions` | app version, git commit (`-dirty` if `src/` had changes), kiwipiepy, ASR model ids, coaching LLM id |
| `target.text` / `.surface` / `.ipa` | sentence, standard pronunciation (G2P), IPA |
| `channels.heard` | Whisper: `text`, `ipa`, `scored: false` |
| `channels.acoustic` | Wav2Vec2-CTC: `text` (no spaces), `ipa`, `scored: true`, `stripped_edge_noise` |
| `channels.acoustic.katakana` | `text`, `is_channel: false`, `method` — a rule-based notation of the acoustic channel, not a channel |
| `score` | `value` (0–100), `definition`, `calibrated: false` |
| `words[]` | one entry per target word, in order (below) |
| `limitations[]` | what the data cannot show |

Each `words[]` entry:

| Field | Meaning |
|---|---|
| `index`, `target`, `surface` | position, the word as written, its standard pronunciation |
| `heard` | Whisper's segment aligned to this word |
| `acoustic` | Wav2Vec2's segment aligned to this word (cut by the scoring alignment) |
| `katakana` | rule-based katakana of `acoustic`; the entries concatenate to `channels.acoustic.katakana.text` |
| `acoustic_errors[]` | scored tags: `tag`, `label_ja`, `ref`, `hyp`, `timestamp_s` (when the CTC timing exists), `explanation_ja` |
| `heard_errors[]` | the same classifier run on Whisper — for display, **not scored** |

`ref`/`hyp` are surface-form jamo; `""` marks a deletion (`hyp`) or an
insertion (`ref`). `explanation_ja` is a fixed template per tag
(`src/labels.py`), never LLM text. How the words are cut is described in
[DECISIONS.md](DECISIONS.md), decision 11.
