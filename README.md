# GhostPilot

Desktop interview copilot for Windows: **invisible overlay** + **ASR → text LLM** + **Alt+P screenshot → multimodal LLM**, with a local **RAG knowledge base**.

## Architecture

```mermaid
flowchart TD
  subgraph UI[Overlay Windows]
    ASR_UI[ASR Overlay Q/A]
    V_UI[Vision Overlay vision answer]
  end

  subgraph Audio[Audio / ASR Pipeline]
    AC[WASAPI Loopback Capture\n(silence padding)]
    WD[Audio Watchdog\n(auto restart)]
    AZ[Azure Speech (partial/final)]
    SEG[Partial Segmenter\n(punct / timeout)]
    RAG[RAGManager\n(knowledge/ chunks)]
    TEXTLLM[Text LLM\n(OpenAI-compatible)]
  end

  subgraph Vision[Vision Pipeline (Independent)]
    HKP[Hotkey Alt+P]
    CAP[AreaCapture (Qt overlay)\n→ JPEG compress]
    VLLM[Vision LLM\n(Gemini via google-genai)]
  end

  AC --> AZ --> SEG --> TEXTLLM --> ASR_UI
  WD --> AC
  RAG --> TEXTLLM

  HKP --> CAP --> VLLM --> V_UI
```

## Quick start (Conda on Windows)

```powershell
conda create -n ghost-pilot python=3.12 -y
conda activate ghost-pilot

pip install -r requirements.txt
```

Create `.env` from `.env.example`, then run:

```powershell
python .\main.py
```

## Configuration

All config can be set via:
- **`.env`** (loaded at startup)
- **Settings UI** (writes `config.json`, which takes precedence)

### Required keys

- **Azure ASR**:
  - `SPEECH_KEY`
  - `SPEECH_REGION` (or `ENDPOINT`)
- **Vision (Gemini)**:
  - `GEMINI_API_KEY`
  - `VISION_MODEL=gemini-...`

### Hotkeys (two overlays)

- **Screenshot (Vision)**:
  - `SCREENSHOT_HOTKEY` (default `alt+p`)
- **ASR overlay interaction (click-through ↔ draggable)**:
  - `ASR_INTERACTION_HOTKEY` / `ASR_INTERACTION_HOTKEY_BACKUP`
- **Vision overlay interaction (click-through ↔ draggable)**:
  - `VISION_INTERACTION_HOTKEY` / `VISION_INTERACTION_HOTKEY_BACKUP`
- **Safety (force both overlays click-through)**:
  - `FORCE_STEALTH_HOTKEY` / `FORCE_STEALTH_HOTKEY_BACKUP`

### Local knowledge base (RAG)

Put your resume/cheatsheets/notes under `knowledge/` (default).

- `KNOWLEDGE_DIR=knowledge`
- `KNOWLEDGE_PATTERNS=*.md,*.txt`

On startup the app will:
- read those files
- chunk them
- embed + index in memory (if `sentence-transformers` is available)
- inject top matches into the text LLM prompt

## Notes / troubleshooting

- **PyQt6 DLL load failed**: prefer installing PyQt/Qt via conda-forge (`conda install -c conda-forge pyqt=6 qt-main`) and make sure VC++ 2015-2022 x64 runtime is installed.
- **Alt+P / Vision failures**: vision pipeline is isolated; failures should show only in the Vision overlay and not stop ASR.

