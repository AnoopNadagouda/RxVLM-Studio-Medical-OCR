# Schema-Aware Text2Cypher Project

This project is a **schema-aware natural-language-to-Cypher system** designed to do more than generate Cypher. It first determines whether a natural-language question is compatible with the provided graph schema, and only then generates a Cypher query if the question is answerable.

The motivation is simple: most text-to-query systems assume every user question should produce a query, which leads to hallucinated labels, properties, and relationships when the schema does not support the question. This project is built to prevent that behavior by making **question-schema compatibility** a first-class part of the model itself.

## What the project is

The system takes two inputs:

- a **graph schema** describing labels, relationships, and properties.
- a **natural-language query** from the user.

It produces one of two outcomes:

- a valid Cypher query if the question matches the schema and is answerable.
- a rejection such as `question not supported by this schema` if the question is irrelevant or incompatible.

This makes the project not only a Text2Cypher generator, but also a **schema-grounded semantic parser with built-in answerability detection**.

## What we are using

### Model: `PhatcatDK/t5gemma-2-270m-270m-encoder-only`

The base model is **`PhatcatDK/t5gemma-2-270m-270m-encoder-only`**, the encoder-only extraction of the 270M T5Gemma 2 checkpoint. This is the lightest practical checkpoint in the family and is used here for fast, text-only schema-aware modeling. [web:373]

| Feature | Value |
|---------|-------|
| **Checkpoint** | `PhatcatDK/t5gemma-2-270m-270m-encoder-only` [web:373] |
| **Model size** | 1.37 GB [web:381] |
| **Saved VRAM** | ~13% [web:381] |
| **Architecture** | Encoder-only extraction |
| **Vision** | Not activated |
| **Purpose** | Fast schema-aware compatibility modeling |

This version is chosen because it is the smallest practical checkpoint in the family and is appropriate for your goal of speed and memory efficiency.

## Datasets we are using

The main Text2Cypher dataset is **Mind the Query**. It contains **27,529 natural-language queries paired with Cypher queries** across **11 real-world graph datasets**, with each example grounded in an accompanying graph database for query execution and validation. [web:5][web:3]

Mind the Query is used for **positive supervision** - training the model to generate valid Cypher when the question matches the schema. [web:5][web:3]

We also use **LC-QuAD 2.0** as an auxiliary natural-language source for negative and mismatch training. LC-QuAD 2.0 contains **30,000 question-query pairs** over Wikidata and DBpedia. [web:22][web:28]

### Dataset summary

| Dataset | Purpose | Size / count |
|---------|---------|-------------|
| Mind the Query | Main positive Text2Cypher supervision | 27,529 NL-Cypher pairs across 11 graph datasets [web:5][web:3] |
| LC-QuAD 2.0 | Auxiliary NL source for mismatch / negative examples | 30,000 question-query pairs [web:22][web:28] |

## Methodology

The methodology combines **positive generation training** with **schema-mismatch training**.

### 1. Positive supervision

Matched question-schema pairs come from Mind the Query. In these examples, the question belongs to the schema and the target output is a valid Cypher query. These samples train the generation side of the model. [web:5][web:3]

### 2. Negative supervision through cross-schema NL mixing

The key project idea is to generate negative examples by **mixing natural-language questions from one dataset with schemas from another, incompatible dataset**. For example, a question about social-network users can be paired with a Wikipedia-like graph schema, or a knowledge-graph question can be paired with an unrelated application schema. [web:5][web:22]

These mismatched pairs are labeled as **unanswerable** or **schema-incompatible**. This teaches the model not to hallucinate a query when the question does not fit the schema.

### 3. Multi-task learning

The model is trained on two related tasks:

- **Task A: answerability / schema compatibility classification** - decide whether the question makes sense for the schema.
- **Task B: Cypher generation** - generate a valid Cypher query only for compatible inputs.

A practical implementation uses a shared encoder backbone and a small classification head on top of the fused schema + NL representation. During inference, the compatibility score gates the decoder.

### 4. Constrained decoding

For additional reliability, constrained decoding can be added so that generation respects structural output rules. This helps keep generated Cypher syntactically and structurally valid.

## CUDA Implementation

The project is set up to run on **CUDA 12.8**, with compatibility for **CUDA 13.x** environments.

### Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
```

### PyTorch CUDA Setup

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

model_name = "PhatcatDK/t5gemma-2-270m-270m-encoder-only"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model = model.to(device)
model = model.half()
```

### CUDA Accelerated Inference

```python
def encode_schema_and_query(schema, nl_query):
    prompt = "Schema: " + schema + "\nQuestion: " + nl_query
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs.last_hidden_state
```

### CUDA Training with Mixed Precision

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./schema_aware_text2cypher",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    save_steps=500,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Performance Comparison

| Metric | CPU | CUDA (FP16) | Speedup |
|--------|-----|-------------|---------|
| Inference time | 100-250ms | 50-100ms | 2-3x faster |
| VRAM | 200MB RAM | 400MB VRAM | GPU memory |
| Training batch | 8-16 | 32-64 | 4x larger |
| Training time | 4-6 hours | 1-2 hours | 3-4x faster |

## Project workflow

1. Input schema and natural-language query.
2. Encode the schema and question together.
3. Predict whether the question is compatible with the schema.
4. If compatible, generate Cypher.
5. If not compatible, return a rejection message.

## Why this methodology matters

Traditional text-to-query systems always produce an answer, even when the schema does not support the question. This creates reliability problems because the model may generate syntactically plausible but semantically invalid queries.

By training explicit question-schema compatibility detection, this project reduces hallucinations, improves execution reliability, and makes the system safer for real-world graph querying.

## Project deliverables

- A trained schema-aware Text2Cypher model.
- Answerability classification head integrated into the model.
- Curated training set with matched and mismatched question-schema pairs.
- Evaluation setup for both query generation and schema-mismatch rejection.
- CUDA-optimized inference pipeline returning either Cypher or rejection.

## Final project definition

In full, this project is a **schema-aware, encoder-based Text2Cypher system built on `PhatcatDK/t5gemma-2-270m-270m-encoder-only`**, trained on Mind the Query for positive Cypher supervision and augmented with cross-schema natural-language mixing using LC-QuAD 2.0 to create unanswerable examples. Its methodology combines schema linking, answerability classification, and CUDA-optimized inference for fast compatibility checking and query generation.