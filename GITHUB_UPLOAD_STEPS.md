# GitHub upload steps

The repository is initialized on `main`, verified, and staged. No author
identity, commit, remote, or push is created automatically.

Before public upload:

1. complete `PUBLICATION_METADATA_REQUIRED.md`;
2. approve `LICENSE_STATUS.md` and create `LICENSE` from `LICENSE.template`;
3. add valid `CITATION.cff` metadata;
4. update any human-only status text, then refresh the root manifest;
5. rerun `python verify_release.py --full`; and
6. create an empty GitHub repository without generated starter files.

Then:

```bash
git status --short
git config user.name "ACCOUNTABLE AUTHOR"
git config user.email "AUTHOR EMAIL"
python refresh_release_manifest.py --confirm-accountable-metadata
git add -A
python verify_release.py --full
git diff --cached --check
git commit -m "Release reproducible ShiftTitan and EviAudio Q1 studies"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git remote -v
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` before changing
anything. Never commit or push credentials, provider-controlled data, audio,
or unlicensed model weights.
