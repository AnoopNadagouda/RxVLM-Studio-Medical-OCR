# Data_TO_KG_Verified

**Goal:** Train a model to generate Cypher queries from natural language questions. To avoid format overfitting, every schema across all 512,940 samples is normalized into one canonical representation — so the model learns query generation, not format memorization.

Unified graph schema dataset — all 512,940 answerable samples across train/val/test normalized into a single format, plus 170,979 injected unanswerable samples for classifier training.

## Unified Format

```
Nodes: Label(prop: TYPE, ...), ...; Relationships: Source->REL->Target, ...
```

Every schema across all sources (Neo4j Text2Cypher 2024/2025, QALD-9-Plus, Mind the Query, SpCQL, Wikidata/DBpedia) uses this format.

## Two-Stage Inference Pipeline

The system uses a two-stage architecture: a lightweight BERT classifier gates the T5 generator.

1. **BERT classifier** (`google/bert_uncased_L-4_H-512_A-8`) — predicts whether a question is answerable given the schema. Trained on 25% unanswerable ratio.
2. **T5Gemma-270M generator** — generates Cypher for answerable questions. Trained on ~1.3–2.5% unanswerable ratio to preserve Cypher quality.

At inference time, BERT filters unanswerable questions before T5 sees them. T5 still sees a small number of unanswerable examples (BERT false negatives + random floor) so it can reject what BERT misses.

## Training Pipeline

### Pipeline ordering

`inject_unanswerable.py` and `build_t5_unanswerable.py` are **not** competing tools. They target different models in the two-stage pipeline and must run in a specific order:

```
1. normalize_schemas.py     →  splits/{train,val,test}_normalized.parquet  (answerable only)
2. inject_unanswerable.py   →  splits/{train,val,test}_normalized.parquet  (25% unanswerable, for BERT)
3. split_answerable.py      →  splits/{train,val,test}_answerable.parquet  (answerable subset, for T5 val/test)
4. train BERT classifier    →  checkpoints/bert_classifier/                (using *_normalized.parquet)
5. build_t5_unanswerable.py →  splits/train_t5.parquet                     (answerable + ~1.3-2.5% unanswerable, for T5)
6. train T5 generator       →  checkpoints/t5gemma/                        (using train_t5.parquet + val/test_answerable.parquet)
```

Notes:

- `inject_unanswerable.py` is for **BERT** (25% ratio) — run BEFORE BERT training.
- `build_t5_unanswerable.py` is for **T5** (low ratio) — run AFTER BERT training (needs the BERT checkpoint).
- `split_answerable.py` must run BEFORE `build_t5_unanswerable.py` (T5 needs the answerable base).
- Running `inject_unanswerable.py` does NOT affect T5 training data — T5 uses `train_t5.parquet`, not `*_normalized.parquet`.

### Step 1: Normalize schemas
- **`normalize_schemas.py`** — Converts 10+ schema formats (markdown, JSON Neo4j, compact, Wikidata, graph_schema_text, Python repr, SQL JSON, etc.) into the unified format. 0 parse failures.
- **`verify_schema.py`** — Validates every schema matches the unified format. 512,940/512,940 pass, 0 old-format pattern hits.

### Step 2: Inject unanswerable rows (for BERT)
- **`inject_unanswerable.py`** — Injects unanswerable rows into normalized parquets at 25% ratio. Pairs questions with incompatible schemas. Total: 170,979 unanswerable rows (154,426 train / 8,312 val / 8,241 test).
- Output: `splits/{train,val,test}_normalized.parquet` (683,919 total rows)

### Step 3: Train BERT classifier
- **`classifier.py`** — Trains BERT classifier on 25% unanswerable parquets. Uses class-weighted loss and auto-tuned batch size based on VRAM.
- Checkpoint: `./checkpoints/classifier_best/`

### Step 4: Build T5 training data (combined approach)
- **`split_answerable.py`** — Extracts answerable-only rows from 25% parquets. Output: `splits/{train,val,test}_answerable.parquet` (512,940 total).
- **`build_t5_unanswerable.py`** — Builds T5's small unanswerable set:
  1. Runs trained BERT on unanswerable train rows → collects false negatives (BERT predicts answerable, truth is unanswerable)
  2. Random samples 5,000 from remaining unanswerable (deduped against false negatives)
  3. Combines: false_negatives × 2 (oversampled) + random_5,000
  4. Appends to answerable base → `splits/train_t5.parquet`
- Expected T5 unanswerable ratio: ~1.3–2.5% (preserves Cypher generation quality)

### Step 5: Train T5 generator
- **`train.py`** — Two-phase curriculum training on `splits/train_t5.parquet`. Phase 1: simpler queries, Phase 2: complex. Validates on `splits/val_answerable.parquet` (pure Cypher quality).
- Uses bf16 with GPU capability check, LoRA, auto-tuned `max_input_length=1024` / `max_target_length=320`.

## Dataset Statistics

| Split | Answerable | Unanswerable (25%) | Total |
|-------|-----------|-------------------|-------|
| Train | 463,278 | 154,426 | 617,704 |
| Val | 24,938 | 8,312 | 33,250 |
| Test | 24,724 | 8,241 | 32,965 |
| **Total** | **512,940** | **170,979** | **683,919** |

T5 trains on answerable + ~6K–12K combined unanswerable (~469K–475K total).

## Sample

`sample/` contains proportional subsets: 90 train / 5 val / 5 test rows.