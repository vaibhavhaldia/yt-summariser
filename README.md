# 🎓 yt-summariser

> Turn 2-hour YouTube lectures into focused Markdown research notes —
> free, local, no API keys, built to last.

`yt-summariser` downloads a video's transcript, scores every segment across
four independent signals, detects topic boundaries, and produces a structured
Markdown report with a TL;DR, chapter summaries, timestamped highlights,
cross-chapter insights, and actionable takeaways. Unlike cloud summarisers it
requires no API key, runs entirely on your machine, grounds every claim in the
actual transcript, and queries Wikipedia on-demand only for terms the speaker
left unexplained. ok.

**Designed for daily use over 2–3 years.** See the
[Maintenance Guide](#-maintenance-guide-2-3-years) for the exact update
checklist.

---

## Table of Contents

1. [What you get](#-what-you-get-per-video)
2. [How it works](#-how-it-works)
3. [Quick Start](#-quick-start)
4. [All Options](#️-all-options)
5. [Example Output](#-example-output)
6. [Architecture](#-architecture)
7. [Dependencies](#-dependencies)
8. [Design Principles](#-design-principles)
9. [Maintenance Guide — 2–3 Years](#-maintenance-guide-2-3-years)
10. [Known Limitations](#-known-limitations)

---

## ✨ What you get (per video)

| Output | Description |
|---|---|
| `<video_id>.md` | Structured Markdown: TL;DR · chapters · highlights · insights · takeaways |
| Highlight Segments | Timestamped table of top moments with source badges (📄 / 🎙️ / 🖼️) |
| Chapter Detection | Automatic topic boundaries with YouTube deep-links |
| Wikipedia Context | On-demand definitions for concepts the speaker left unexplained |
| `<video_id>_highlights.mp4` | Clipped highlight reel (skip with `--no-video`) |

---

## 🏗️ How it works

### Transcript tiers (best available wins, cheapest tried first)

| Tier | Source | Accuracy | Cost |
|---|---|---|---|
| 0 | YouTube manual captions (creator-uploaded) | ~98 % | Free, instant |
| 1 | YouTube auto-generated captions | ~85–95 % | Free, instant |
| 2 | Whisper ASR local fallback | ~80–90 % | Free, ~1–3 min/hour |
| 3 | Frame OCR via easyocr (weak segments only) | ~88–92 % on slides | Free, ~2 s/frame |
| 4 | moondream2 visual caption (when OCR finds nothing) | ~70–80 % | Free, needs `transformers` |

A segment is **weak** (triggers Tier 3/4) when it has fewer than 5 words,
less than 0.5 words per second, or text like `[Music]` / `[Applause]`.

### Synthesis stack

| Stage | Tool | Fallback |
|---|---|---|
| Scoring | TF-IDF · MiniLM cosine · word density · audio energy | TF-IDF + density |
| Chapter detection | Cosine similarity shifts on MiniLM embeddings | Single chapter |
| Chapter titles | BART `facebook/bart-large-cnn` `max_length=10` | TF-IDF top-4 terms |
| LLM synthesis | Mistral-7B-Instruct Q4_K_M via `llama-cpp-python` | Sumy LSA + KeyBERT |
| Wikipedia | REST API, ~3–8 calls/video, `[[LOOKUP: term]]` only | Silently skipped |

**Wikipedia is a dictionary, not an editor.** It is consulted only when
Mistral emits a `[[LOOKUP: term]]` marker, meaning the transcript itself left
a concept unexplained. It never adds missing topics or criticises the speaker.

---

## 🚀 Quick Start

```bash
# 1. Install core dependencies (fast, always required)
pip install yt-dlp scikit-learn

# 2. Install recommended dependencies (one-time, models downloaded on first run)
pip install -r requirements.txt

# 3. Markdown report only (fastest, no models needed)
python -m summarise "https://youtube.com/watch?v=VIDEO_ID" \
    --no-llm --no-enrich --no-video

# 4. Full pipeline with highlight video (default)
python -m summarise "https://youtube.com/watch?v=VIDEO_ID"

# 5. Multiple videos in parallel
python -m summarise \
    "https://youtube.com/watch?v=VIDEO_A" \
    "https://youtube.com/watch?v=VIDEO_B" \
    --workers 2

# 6. Entire playlist
python -m summarise "https://youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"

# 7. Custom output length (12 minutes of highlights)
python -m summarise "https://youtube.com/watch?v=VIDEO_ID" \
    --target-mins 12

# 8. Save to a specific folder
python -m summarise "https://youtube.com/watch?v=VIDEO_ID" \
    --output-dir ~/research/ml-lectures
```

Reports are written to `./output/<video_id>.md` by default. Re-running the
same URL is a no-op: if `<video_id>.md` already exists the video is skipped.

---

## ⚙️ All Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `urls` | positional (1+) | — | YouTube video or playlist URLs |
| `--target-pct` | float | `0.10` | Highlight duration as fraction of video length |
| `--target-mins` | float | `None` | Override `--target-pct` with exact minutes |
| `--no-video` | flag | off | Skip ffmpeg clipping, produce `.md` only |
| `--no-llm` | flag | off | Skip Mistral-7B; use Sumy/KeyBERT extractive fallback |
| `--no-enrich` | flag | off | Skip frame OCR and visual captioning |
| `--no-captions` | flag | off | Run OCR but skip moondream2 visual captioning |
| `--output-dir` | str | `output` | Directory for `.md` files and `.mp4` clips |
| `--whisper-model` | str | `base` | Whisper model: `tiny` `base` `small` `medium` `large` |
| `--workers` | int | `1` | Parallel threads for batch processing (max 15 videos) |
| `--loglevel` | str | `warning` | Verbosity: `debug` `info` `warning` |

---

## 📄 Example Output

Snippet from a hypothetical video on *Attention Is All You Need* (Vaswani et al.):

```markdown
# Attention Is All You Need — Explained

| | |
|---|---|
| **URL** | [https://youtu.be/iDulhoQ2pro](https://youtu.be/iDulhoQ2pro) |
| **Duration** | 1h 26m 12s |
| **Processed** | 2024-06-15T14:32:07 |
| **Transcript** | 📄 auto — 1847 segments total |
| **Highlights** | 9 clips · 8m 43s |

---

## 🎯 TL;DR

The Transformer replaces recurrence with self-attention [00:04:11], achieving
state-of-the-art translation while training fully in parallel.  Multi-head
attention lets the model attend to different representation subspaces jointly
[00:09:33]. Positional encodings using sine/cosine substitute for the
sequential inductive bias previously provided by RNNs [00:21:58].

---

## 🏷️ Key Concepts

`self-attention` `multi-head attention` `positional encoding`
`scaled dot-product` `encoder-decoder` `layer normalisation` `BLEU score`

---

## 📖 Chapters

### Chapter 1 · Motivation & RNN Limitations · [[00:00:00]](https://youtu.be/iDulhoQ2pro?t=0)
The speaker explains why LSTMs struggle with long-range dependencies and
prevent efficient parallelisation during training [00:02:44].

### Chapter 2 · Scaled Dot-Product Attention · [[00:07:18]](https://youtu.be/iDulhoQ2pro?t=438)
Attention is a weighted sum of values where weights are computed from query-key
compatibility [00:08:55]. The 1/√d_k scaling prevents gradients from vanishing
in high-dimensional spaces [00:11:02].

---

## ✂️ Highlight Segments

| # | Time | Score | Source | Excerpt |
|---|---|---|---|---|
| 1 | [00:08:55](https://youtu.be/iDulhoQ2pro?t=535) | 0.91 | 📄 auto | Attention is a weighted sum of the values where the weight is a compatibility function of the query with the key... |
| 2 | [00:19:47](https://youtu.be/iDulhoQ2pro?t=1187) | 0.87 | 📄 auto | Instead of a single attention function we found it beneficial to project queries, keys and values h times... |
| 3 | [00:43:12](https://youtu.be/iDulhoQ2pro?t=2592) | 0.79 | 🖼️ ocr | [ON SCREEN: BLEU Score Table — Transformer (big): 41.0 EN-DE, 43.9 EN-FR] |

---

## 💡 Actionable Takeaways

1. Apply 1/√d_k scaling before softmax to avoid gradient saturation.
2. Start with h=8 attention heads; ablations show degradation below h=4.
3. Add positional encodings before the first encoder layer.

---

## 📚 References

**self-attention** — a mechanism that relates different positions of a single
sequence to compute a representation of that sequence.
· [Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning))
```

---

## 🧩 Architecture

```
YouTube URL
    │
    ├─ Tier 0/1: yt-dlp captions ──────────► Segments        [transcript.py]
    ├─ Tier 2:   Whisper (fallback) ─────────►
    │
    ├─ Tier 3/4: Frame OCR + Caption ────────► EnrichedSegs   [enricher.py]
    │            (weak segments only)
    │            easyocr · moondream2
    │
    ├─ Scoring ───────────────────────────────────────────    [scorer.py]
    │   TF-IDF · MiniLM cosine · word density · ffmpeg energy
    │
    ├─ Chapter Detection ─────────────────────────────────    [segmenter.py]
    │   cosine similarity shifts · BART titles
    │
    ├─ Greedy Selection ──────────────────────────────────    [selector.py]
    │   fill target_duration · merge adjacent clips
    │   └─► SelectedSegments
    │
    ├─ Mistral-7B Synthesis ──────────────────────────────    [synthesiser.py]
    │   llama-cpp-python · grounded · timestamp-cited
    │   Fallback: Sumy LSA + KeyBERT
    │   Emits [[LOOKUP: X]] markers for unknown terms
    │
    ├─ Wikipedia ─────────────────────────────────────────    [references.py]
    │   on-demand REST API · ~3–8 calls/video
    │   resolves [[LOOKUP: X]] markers ONLY
    │
    ├─ ffmpeg Clipping ───────────────────────────────────    [clipper.py]
    │   yt-dlp stream URL · -c copy · concat demuxer
    │   └─► <video_id>_highlights.mp4
    │
    └─ Markdown Reporter ─────────────────────────────────    [reporter.py]
        deep-links · source badges · injected references
        └─► <video_id>.md
```

---

## 📦 Dependencies

### Python packages

| Package | Purpose | Required? |
|---|---|---|
| `yt-dlp` | Subtitle download, stream URL, playlist expansion, audio extraction | ✅ Required |
| `scikit-learn` | TF-IDF scoring, chapter title fallback, concept extraction | ✅ Required |
| `sentence-transformers` | MiniLM embeddings: semantic scoring + chapter detection | ⭐ Recommended |
| `transformers` | BART chapter titles, moondream2 visual captioning | 🔵 Optional |
| `torch` | Required by `transformers` and `sentence-transformers` | 🔵 Optional |
| `accelerate` | Speeds up `transformers` inference | 🔵 Optional |
| `llama-cpp-python` | Mistral-7B-Instruct Q4_K_M local inference | ⭐ Recommended |
| `huggingface-hub` | Auto-downloads GGUF model on first use | 🔵 Optional |
| `openai-whisper` | Local ASR when no YouTube captions exist | 🔵 Optional |
| `easyocr` | OCR on video frames for weak segments | 🔵 Optional |
| `sumy` | LSA extractive summariser (LLM fallback) | 🔵 Optional |
| `keybert` | Key concept extraction (LLM fallback) | 🔵 Optional |

### System tools (must be on `PATH`)

| Tool | Purpose | Install |
|---|---|---|
| `ffmpeg` | Audio analysis, frame extraction, clipping, concat | `brew install ffmpeg` / `apt install ffmpeg` |
| `yt-dlp` binary | Subtitle + video download | `pip install yt-dlp` or [yt-dlp releases](https://github.com/yt-dlp/yt-dlp/releases) |

---

## 💡 Design Principles

**1. Transcript is the only authority**
The LLM cannot add information not present in the transcript. Every factual
claim must end with a timestamp citation (`[00:14:22]`). The prompt explicitly
forbids hallucination. The transcript is capped at 6 000 chars, prioritising
the highest-scored segments.

**2. Wikipedia is a dictionary**
Wikipedia is consulted on-demand and only when the LLM emits `[[LOOKUP: term]]`
— meaning the transcript itself left a concept unexplained. It never adds
topics the speaker did not discuss, never corrects the speaker, never signals
gaps. Disambiguation pages and no-extract articles are silently discarded.

**3. Graceful degradation**
Every heavy dependency is optional and lazily imported. Missing packages emit
a `warnings.warn` and the pipeline continues with the next-best strategy. Even
with only `yt-dlp` + `scikit-learn`, the pipeline always produces a report.

---

## 🔧 Maintenance Guide — 2–3 Years

This section documents exactly what will need attention over the lifetime of
this tool, in order of likelihood. Nothing here requires code changes unless
explicitly noted.

---

### 🔴 HIGH — Will definitely need action

#### 1. `yt-dlp` — update every 2–4 months

**Why it breaks:** YouTube changes its internal API, webpage structure, and
signature algorithms regularly. yt-dlp is the only component that directly
talks to YouTube. When it breaks, subtitle downloads and stream URL resolution
both fail.

**Symptom:**
```
ERROR: [youtube] VIDEO_ID: Sign in to confirm you're not a bot
ERROR: Unable to extract video data
```

**Fix (30 seconds):**
```bash
pip install -U yt-dlp
```

**Or if using the binary directly:**
```bash
yt-dlp -U
```

**How to check without running a video:**
```bash
yt-dlp --version
yt-dlp "https://youtu.be/dQw4w9WgXcQ" --get-title --no-warnings
```

**In the desktop app:** Settings → [Check for yt-dlp update] button does this
automatically.

**Files affected:** `utils.py`, `transcript.py`, `clipper.py`
**Code change needed:** Almost never — yt-dlp's CLI interface is stable.
Only needed if yt-dlp removes or renames a flag used in these files.

---

#### 2. YouTube subtitle format — watch for VTT changes

**Why it breaks:** The VTT parser in `utils.py:_parse_vtt()` expects a
specific cue block structure. YouTube has changed this format before (adding
extra metadata lines, changing timing precision, adding new tag types).

**Symptom:** Reports are generated but all segments have empty text, or
TF-IDF scores are uniformly 0.0.

**Fix:** Check the raw `.vtt` file:
```bash
yt-dlp --write-auto-sub --sub-lang en --skip-download \
    --output "/tmp/test_%(id)s" "https://youtu.be/VIDEO_ID"
cat /tmp/test_VIDEO_ID.en.vtt | head -40
```

Compare the structure against what `_parse_vtt()` in `utils.py` expects.
The function is self-contained and well-commented — the fix is usually adding
a new skip condition for a new header line or a new tag pattern to
`_VTT_TAG_RE`.

**Files affected:** `utils.py` (function `_parse_vtt`, lines ~160–220)
**Code change:** Minor regex or skip-line update, 5–10 lines.

---

### 🟡 MEDIUM — Will likely need action within 2 years

#### 3. `llama-cpp-python` — update when Python or LLVM version changes

**Why it breaks:** `llama-cpp-python` compiles C++ extensions at install time.
A new Python version (e.g. 3.13 → 3.14) or a macOS Xcode update can break
the build.

**Symptom:**
```
ImportError: llama_cpp.so: cannot open shared object file
```
or a build failure during `pip install llama-cpp-python`.

**Fix:**
```bash
pip install -U llama-cpp-python --force-reinstall --no-cache-dir
```

On Apple Silicon (M1/M2/M3), use the Metal-accelerated build:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python \
    --force-reinstall --no-cache-dir
```

**Files affected:** `synthesiser.py` (function `_get_llm`)
**Code change needed:** Never — the `Llama()` constructor API has been
stable since v0.2. Only change if `llama-cpp-python` deprecates a parameter
(check their [CHANGELOG](https://github.com/abetlen/llama-cpp-python/blob/main/CHANGELOG.md)).

---

#### 4. Mistral-7B model — upgrade to better quantisation or newer version

**Why to upgrade:** Better models (Mistral v0.3, Mistral Nemo 12B, Llama 3.1
8B) provide higher-quality synthesis with the same or less RAM. Quantisation
improvements (Q5_K_M, Q6_K) offer better quality at similar file size.

**This requires zero code changes.** The GGUF format is stable. Just swap the
model file.

**How to upgrade:**
1. Download a new `.gguf` file from HuggingFace:
   - [TheBloke's GGUF collection](https://huggingface.co/TheBloke)
   - [bartowski's newer quantisations](https://huggingface.co/bartowski)
2. Point to it:
   - **Desktop app:** Settings → LLM Model Path → Browse
   - **CLI:** set `llm_model_path` in `synthesiser._get_llm()` or pass via
     the `model_path` argument to `synthesise()`

**Recommended upgrade candidates (2025–2027):**

| Model | Size | RAM needed | Quality vs current |
|---|---|---|---|
| `mistral-7b-instruct-v0.3.Q4_K_M.gguf` | 4.1 GB | 6 GB | Marginally better |
| `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` | 4.9 GB | 7 GB | Noticeably better |
| `Mistral-Nemo-Instruct-2407.Q4_K_M.gguf` | 7.7 GB | 10 GB | Significantly better |
| `Phi-3.5-mini-instruct.Q4_K_M.gguf` | 2.2 GB | 4 GB | Smaller, faster, similar |

**Files affected:** `synthesiser.py` (function `_get_llm`, lines ~90–120)
**Code change needed:** Only if the new model uses a different prompt format
than Mistral instruct (`[INST]...[/INST]`). Check the model card on HuggingFace.
If it uses ChatML format instead, update `_build_prompt()` accordingly.

---

#### 5. `sentence-transformers` + `all-MiniLM-L6-v2` — model may be superseded

**Why to upgrade:** The `all-MiniLM-L6-v2` model (80 MB) is from 2021.
Better small embedding models now exist.

**Recommended replacement (drop-in):**
```bash
# Replace all-MiniLM-L6-v2 with a 2024 model — same API, better quality
```

Change this one line in both `scorer.py` and `segmenter.py`:
```python
# Before
model = SentenceTransformer("all-MiniLM-L6-v2")

# After (2024 — better quality, same size)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Or (2024 — slightly larger but notably better)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
```

**Files affected:** `scorer.py` (function `_score_semantic`, ~1 line),
`segmenter.py` (function `_encode_texts`, ~1 line)

---

### 🟢 LOW — Unlikely to need action, but document anyway

#### 6. `easyocr` — update if OCR quality degrades on new slide styles

**Symptom:** OCR returns garbage or empty strings on slides that look visually
clear.

**Fix:**
```bash
pip install -U easyocr
```

EasyOCR occasionally releases improved models for English. The API
(`reader.readtext()`) has been stable since v1.0.

**Files affected:** `enricher.py` (function `_run_ocr`)

---

#### 7. `transformers` + moondream2 — update if captioning model is deprecated

The `vikhyatk/moondream2` model on HuggingFace is actively maintained as of
2024. If it is deprecated or removed:

1. Find a replacement on HuggingFace with `pipeline("image-to-text")`:
   - `nlpconnect/vit-gpt2-image-captioning` (smaller, faster)
   - `Salesforce/blip-image-captioning-base` (better quality)

2. Change one line in `enricher.py`:
```python
# Before
_caption_pipeline = hf_pipeline("image-to-text", model="vikhyatk/moondream2")

# After
_caption_pipeline = hf_pipeline("image-to-text",
                                  model="Salesforce/blip-image-captioning-base")
```

**Files affected:** `enricher.py` (function `_run_caption`, ~1 line)

---

#### 8. Wikipedia REST API — extremely stable, no action expected

The Wikipedia REST API (`/api/rest_v1/page/summary/{title}`) has been stable
since 2015. Wikipedia's terms of service for automated access require only a
descriptive `User-Agent` header, which is already set in `references.py`.

The only scenario requiring a fix: Wikimedia changes the JSON response schema.
**Symptom:** `WikiReference.summary` is always empty despite API returning 200.

**Fix:** Inspect the raw response:
```python
import urllib.request, json
url = "https://en.wikipedia.org/api/rest_v1/page/summary/Attention_(machine_learning)"
with urllib.request.urlopen(url) as r:
    print(json.dumps(json.loads(r.read()), indent=2))
```

Then update `_fetch_wiki()` in `references.py` to use the new field names.

---

#### 9. `BART` (`facebook/bart-large-cnn`) — update if HuggingFace deprecates

This model has been on HuggingFace since 2020 and is extremely widely used.
Deprecation is unlikely. If it happens, replace with:
```python
# In segmenter.py, function _generate_chapter_title:
_bart_pipeline = hf_pipeline("summarization",
                               model="facebook/bart-large-xsum")  # smaller, punchier
# or
_bart_pipeline = hf_pipeline("summarization",
                               model="sshleifer/distilbart-cnn-12-6")  # 50% faster
```
---
```
┌─────────────────────────────────────────────────┐
│  🎓 yt-summariser                    [_][□][X]  │
├──────────────┬──────────────────────────────────┤
│              │                                  │
│  📚 History  │   ┌─ Add Video ──────────────┐   │
│  ─────────── │   │ URL: [________________]  │   │
│  Today       │   │ [▶ Analyse]  [⚙ Options] │   │
│  • Video 1   │   └──────────────────────────┘   │
│  • Video 2   │                                  │
│              │   ┌─ Progress ───────────────┐   │
│  Yesterday   │   │ ████████░░░ Stage 6/10   │   │
│  • Video 3   │   │ Detecting chapters...    │   │
│              │   └──────────────────────────┘   │
│  This week   │                                  │
│  • Video 4   │   ┌─ Output ─────────────────┐   │
│  • Video 5   │   │ [📄 Open Report]         │   │
│              │   │ [🎬 Open Video]          │   │
│  🔍 Search   │   │ [📁 Open Folder]         │   │
│  [_______]   │   └──────────────────────────┘   │
│              │                                  │
│  🏷️ Topics   │   ┌─ Preview ────────────────┐   │
│  ml (12)     │   │ # Attention Is All You   │   │
│  python (8)  │   │ ## TL;DR                 │   │
│  systems (5) │   │ The paper introduces...  │   │
│              │   │ ## Chapters              │   │
│              │   │ ### 1. Motivation [▶]    │   │
└──────────────┴──────────────────────────────────┘
```

---

### 📅 Recommended Update Schedule

| Frequency | Task |
|---|---|
| **Every 2–4 months** | `pip install -U yt-dlp` |
| **Every 6 months** | `pip install -U yt-dlp scikit-learn sentence-transformers easyocr` |
| **Annually** | Review model quality — consider swapping to a better GGUF |
| **On Python upgrade** | Reinstall `llama-cpp-python` from scratch |
| **When a stage breaks** | Check the symptom table above, fix the identified file |

---

### 🧪 Quick Health Check

Run this after any update to confirm the pipeline still works end-to-end:

```bash
# Uses a short video (Rick Astley — 3.5 min) to test all stages quickly
python -m summarise "https://youtu.be/dQw4w9WgXcQ" \
    --no-video \
    --no-enrich \
    --output-dir /tmp/yt_health_check

# Expected: /tmp/yt_health_check/dQw4w9WgXcQ.md created with content
cat /tmp/yt_health_check/dQw4w9WgXcQ.md | head -20
```

If the file is created with a non-empty TL;DR and at least one highlight
segment, the pipeline is healthy.

---

## ⚠️ Known Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| English only | `--sub-lang en` hardcoded; TF-IDF stop words are English | Edit `utils.py` line ~170 to change language code |
| Mistral-7B RAM | Needs ~6 GB RAM minimum | Use `--no-llm` flag or swap to `Phi-3.5-mini` (4 GB) |
| Frame enrichment speed | ~2 s/frame on CPU; many weak segments add minutes | Use `--no-enrich` to skip |
| Clip boundary drift | `-c copy` snaps to nearest keyframe, up to ~2 s drift | Unavoidable without re-encoding; use timestamps in `.md` instead |
| Max 15 videos/run | Hard cap in `summarise.py` to prevent runaway downloads | Run the tool multiple times for large playlists |
| No GPU acceleration | All models run on CPU by default | For llama-cpp: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python` (macOS) |
