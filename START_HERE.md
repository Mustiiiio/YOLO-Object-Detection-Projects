# Start here

1. Read `README.md` and the two project READMEs.
2. Install dependencies into a virtual environment: `python -m pip install -r requirements.txt`.
3. Place your own test input outside the repository or under each project's `data/` (ignored by Git).
4. Run `python tools/run_inference.py --project mine --source PATH` or use `--project shape`.
5. Inspect annotated media in `outputs/`.
6. Before publishing, inspect `docs/reproducibility_and_limitations.md`, confirm model provenance and decide whether weights should live on GitHub or in Releases/LFS.

The original scripts are included as historical source, **not** guaranteed plug-and-play scripts. They refer to missing input media and model filenames (`yolomW.pt`, `best263D2.pt`). The new CLI lets you select the submitted weights without rewriting those sources.
