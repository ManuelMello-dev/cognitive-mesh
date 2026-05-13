# Cognitive Mesh — Z³ Language Neural Network

Cognitive Mesh is a language-first Z³ neural runtime for interaction, testing,
and continuous cognitive development. The system keeps the core mesh
domain-agnostic while using real language streams as the default proving source.

## Current Focus

The active direction is language training and chatbox-driven testing. The default
runtime loads `LanguageStreamPlugin`, which reads real text from
`LANGUAGE_TRAINING_CORPUS_PATH` or `LANGUAGE_TRAINING_TEXT` and emits generic
observations under `language:corpus`. The browser dashboard includes a chatbox
that posts to `/api/chat`; each chat or analysis message is also queued as a
`language:chat` observation so interaction becomes part of the test stream.

| Area | Runtime Behavior |
|---|---|
| Default stream | `LanguageStreamPlugin` |
| Corpus input | `LANGUAGE_TRAINING_CORPUS_PATH` or `LANGUAGE_TRAINING_TEXT` |
| Chatbox testing | `/api/chat` returns a response and queues `language:chat` observations |
| Direct ingestion | `/api/ingest` accepts generic observations |
| Legacy markets | Optional only with `ENABLE_MARKET_PLUGIN=1` |

## Run Locally

```bash
pip install -r requirements-minimal.txt
export LANGUAGE_TRAINING_TEXT="Your real language corpus or transcript text goes here."
python main.py
```

Open the configured service port and use the Z³ Language Neural Console chat tab
for interaction and testing.

## Useful Environment Variables

| Variable | Purpose |
|---|---|
| `DISABLE_LANGUAGE_PLUGIN` | Set to `1` to disable the default language stream |
| `LANGUAGE_TRAINING_CORPUS_PATH` | Path to a real text corpus file |
| `LANGUAGE_TRAINING_TEXT` | Inline real text corpus for quick tests |
| `LANGUAGE_TRAINING_BATCH_SIZE` | Number of corpus units emitted per fetch cycle |
| `ENABLE_MARKET_PLUGIN` | Enables the legacy financial stream |
| `ENABLE_MARKET_CONTEXT_PLUGINS` | Enables optional legacy financial context plugins |

## HTTP Interaction

```bash
curl -X POST http://localhost:8081/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Test this Z³ language interaction.","mode":"chat"}'
```

The response includes `language_ingested` to confirm whether the chat message was
queued into the language testing stream.
