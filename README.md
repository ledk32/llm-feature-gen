# LLM Feature Gen

**LLM Feature Gen** is a Python library for discovering and generating interpretable features from unstructured data with large language models.

It helps you:
- discover human-interpretable features from images, text, tabular data, and video
- turn model outputs into structured JSON artifacts
- generate feature values from raw multimodal inputs for downstream models
- export per-class CSVs that are ready for analysis or modeling

## How It Works

The library supports a two-step workflow:

1. **Discover features** from a dataset and save them as JSON in `outputs/`.
2. **Generate feature values** for each file or row using the discovered feature schema.

## Supported Inputs

### Discovery

- Images: `.jpg`, `.jpeg`, `.png`
- Text: `.txt`, `.md`, `.pdf`, `.docx`, `.html`
- Tabular: `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json`
- Video: `.mp4`, `.mov`, `.avi`, `.mkv`

### Generation

- Images, text, tabular files, and videos are supported through the same folder-based pipeline.
- Generation expects a root folder with one subfolder per class, for example `images/hotpot/` and `images/vase/`.

### Optional Parser Dependencies

The base install covers the core package, but some formats need extra packages at runtime:

- `.pdf`: `pypdf`
- `.docx`: `python-docx`
- `.html`: `beautifulsoup4`
- `.xlsx`: `openpyxl`
- `.xls`: `xlrd`
- `.parquet`: `pyarrow` or `fastparquet`

For video audio extraction, you also need the `ffmpeg` system binary available on your machine.

## Project Structure

```text
llm-feature-gen/
├─ src/
│  ├─ llm_feature_gen/
│  │  ├─ __init__.py
│  │  ├─ discover.py
│  │  ├─ generate.py
│  │  ├─ providers/
│  │  │  ├─ local_provider.py
│  │  │  └─ openai_provider.py
│  │  ├─ prompts/
│  │  │  ├─ image_discovery_prompt.txt
│  │  │  ├─ image_generation_prompt.txt
│  │  │  ├─ text_discovery_prompt.txt
│  │  │  └─ text_generation_prompt.txt
│  │  └─ utils/
│  │     ├─ image.py
│  │     ├─ text.py
│  │     └─ video.py
│  └─ tests/
│     ├─ conftest.py
│     ├─ test_discover_more.py
│     ├─ test_discovery.py
│     ├─ test_generation.py
│     ├─ test_providers.py
│     └─ test_utils_and_prompts.py
├─ outputs/
├─ pyproject.toml
└─ README.md
```

## Installation

Install from PyPI:

```bash
pip install llm-feature-gen
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

If you need non-core document or tabular formats:

```bash
pip install pypdf python-docx beautifulsoup4 openpyxl xlrd pyarrow
```

## Environment Setup

Create a `.env` file in the project root.

### OpenAI API

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_name
OPENAI_AUDIO_MODEL=whisper-1
```

### Azure OpenAI

```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_GPT41_DEPLOYMENT_NAME=your_chat_deployment
AZURE_OPENAI_WHISPER_DEPLOYMENT=your_audio_deployment
```

If `AZURE_OPENAI_ENDPOINT` is set, the provider automatically uses Azure OpenAI. Otherwise it falls back to the standard OpenAI API.

## Discovery Examples

### Discover Features from Images

```python
from llm_feature_gen.discover import discover_features_from_images

result = discover_features_from_images(
    image_paths_or_folder="discover_images",
    as_set=True,
)

print(result)
```

This reads all supported images in `discover_images/`, sends them as a joint set to the provider, and saves the result to `outputs/discovered_image_features.json`.

Example output:

```json
{
  "proposed_features": [
    {
      "feature": "has visible handle",
      "description": "Some objects include handles, while others do not.",
      "possible_values": ["present", "absent"]
    },
    {
      "feature": "color tone",
      "description": "Objects vary between metallic, earthy, and bright palettes.",
      "possible_values": ["metallic", "earthy", "bright", "dark"]
    }
  ]
}
```

### Discover Features from Text

```python
from llm_feature_gen.discover import discover_features_from_texts

result = discover_features_from_texts(
    texts_or_file="discover_texts",
    as_set=True,
)

print(result)
```

This loads all supported text documents in `discover_texts/`, extracts raw text, and saves the result to `outputs/discovered_text_features.json`.

### Discover Features from Tabular Data

```python
from llm_feature_gen.discover import discover_features_from_tabular

result = discover_features_from_tabular(
    file_or_folder="discover_tabular",
    text_column="text",
    as_set=True,
)

print(result)
```

This loads supported tabular files, reads the `text` column, and saves the result to `outputs/discovered_tabular_features.json`.

Example output:

```json
{
  "proposed_features": [
    {
      "feature": "overall sentiment",
      "description": "Rows differ in whether they express favorable or unfavorable opinions.",
      "possible_values": ["positive", "negative", "mixed"]
    },
    {
      "feature": "focus of the review",
      "description": "Some rows focus on performance, others on plot, visuals, or general quality.",
      "possible_values": ["performance", "plot", "visuals", "general quality"]
    }
  ]
}
```

### Discover Features from Videos

```python
from llm_feature_gen.discover import discover_features_from_videos

result = discover_features_from_videos(
    videos_or_folder="discover_videos",
    as_set=True,
    num_frames=5,
    use_audio=True,
)

print(result)
```

This extracts key frames, optionally transcribes audio, and saves the result to `outputs/discovered_video_features.json`.

## Generation Example

After discovery, you can generate feature values for each class folder.

```python
from llm_feature_gen.generate import generate_features_from_images

csv_paths = generate_features_from_images(
    root_folder="images",
    discovered_features_path="outputs/discovered_image_features.json",
    merge_to_single_csv=True,
)

print(csv_paths)
```

With a folder layout like this:

```text
images/
├─ hotpot/
└─ vase/
```

the command writes per-class CSVs such as `outputs/hotpot_feature_values.csv` and `outputs/vase_feature_values.csv`. If `merge_to_single_csv=True`, it also creates `outputs/all_feature_values.csv`.

The same workflow is available for other modalities:

```python
from llm_feature_gen.generate import (
    generate_features_from_images,
    generate_features_from_tabular,
    generate_features_from_texts,
    generate_features_from_videos,
)
```

## Running Tests

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Useful commands:

```bash
pytest -vv
pytest src/tests/test_discovery.py
```

Tests use fake providers and temporary directories, so they do not require OpenAI or Azure credentials.
