# Bird species identifier (catalog #1168)

Random-forest bird classifier from Eckroth’s *Python AI Projects for Beginners*, Chapter 2 (CUB-200-2011 attributes).

## Data (one-time)

```powershell
Set-Location "c:\Users\emmak\Desktop\Content\CodingProjects\PortProjects\demos\bird-species"
python scripts/fetch_cub_metadata.py
```

Downloads text metadata only (~70 MB attribute file; no bird images).

## Run the web demo (port 3031)

```powershell
Set-Location "c:\Users\emmak\Desktop\Content\CodingProjects\PortProjects\demos\bird-species\web"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 3031
```

Open http://localhost:3031 — first load trains and caches the model (~1–2 minutes).

Provenance: [BOOK_DIRECTIONS.md](./BOOK_DIRECTIONS.md).
