# Morphology-Aware G2P Engine Enhancement

This plan outlines the strategy to integrate a morphological analyzer (`kiwipiepy`) into Phomene's deterministic G2P pipeline. This upgrade will resolve the known phonological limitations (e.g., `맛없다`, `꽃잎`, `밟다`) that currently require context beyond raw spelling.

## User Review Required

> [!WARNING]
> **New Dependency**: This will add `kiwipiepy` to the project's `requirements.txt`. It is a fast, C++ based morphological analyzer with a Python wrapper. It adds ~50MB to the environment. Please confirm this addition.

## Open Questions

None at this stage. The logic elegantly piggybacks on the existing robust pipeline.

## Proposed Changes

We will introduce a `_preprocess_morphology()` hook that runs *before* the main rule pipeline. By analyzing the POS (Part-of-Speech) tags of adjacent syllables, we can mutate the text characters at boundaries to trigger the correct behavior in the downstream deterministic pipeline.

### Core Logic Implementation

#### [MODIFY] [src/g2p.py](file:///C:/Users/Baek/Phomene/src/g2p.py)
1. **Import & Init**: Instantiate `kiwipiepy.Kiwi()` lazily or at module level.
2. **Morphological Preprocessor**: Add `_preprocess_morphology(text: str) -> str`. It will tokenize words and apply three specific boundary mutations:
   - **ㄴ-Insertion (ㄴ 첨가)**: If Token1 ends in a consonant and Token2 is a substantive starting with `ㅣ, ㅑ, ㅕ, ㅛ, ㅠ`, we mutate Token2's onset to `ㄴ` (e.g., `꽃` + `잎` -> `꽃` + `닢`). The pipeline will naturally handle `꽃닢` -> `꼳닙` -> `꼰닙`.
   - **Stem Tensification (어간 경음화)**: If Token1 is a Verb/Adj stem ending in `ㄴ, ㅁ, ㄼ, ㄾ, ㄺ` and Token2 is an Eomi starting with `ㄱ, ㄷ, ㅅ, ㅈ`, we tensify Token2's onset (e.g., `신` + `다` -> `신` + `따`).
   - **Substantive Liaison Exception (의미 경계 연음 예외)**: If Token1 ends in a consonant and Token2 is a substantive starting with `ㅏ, ㅓ, ㅗ, ㅜ, ㅟ`, we force-neutralize Token1's coda (e.g., `맛` + `없다` -> `맏` + `없다`). The pipeline handles `맏없다` -> `마덥따`.
3. **Pipeline Integration**: Call `_preprocess_morphology(text)` at the very beginning of the `to_surface(text)` and `to_jamo_sequence(text)` functions.

#### [MODIFY] [tests/test_g2p.py](file:///C:/Users/Baek/Phomene/tests/test_g2p.py)
Add test cases to ensure the new morphology-aware rules work perfectly:
- `맛없다` -> `마덥따`
- `꽃잎` -> `꼰닙`
- `물약` -> `물략`
- `신다` -> `신따`
- `넓게` -> `널께`
- `읽고` -> `일꼬`
- `밟다` -> `밥따`

#### [MODIFY] [requirements.txt](file:///C:/Users/Baek/Phomene/requirements.txt)
- Add `kiwipiepy` to the list of dependencies.

## Verification Plan

### Automated Tests
- Run `pytest tests/test_g2p.py -v` to ensure the newly added test cases pass.
- The 51/51 held-out test suite will run in CI to ensure no regressions occur on standard, non-morphological rules.

### Manual Verification
- Run the Streamlit app locally (`streamlit run app.py`).
- Input the edge cases (`맛없다`, `꽃잎이`, `신다`) into the UI and verify that the IPA, surface form, and Jamo Diff Table reflect the correct morphological boundaries.
