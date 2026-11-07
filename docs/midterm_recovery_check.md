# Original Midterm backup recovery check

Checked on November 8, 2026 (Asia/Qyzylorda), after the owner supplied
[Nura007/itune_ml](https://github.com/Nura007/itune_ml) as the backup location.
This was a read-only GitHub investigation. No branch, commit, workflow or remote file was changed.

**Result: the original private dataset and saved split were not recovered.** The repository and
downloadable research artifacts preserve the Midterm source, aggregate results and presentation.
They do not contain the original raw API responses, cleaned app-level data, split manifest,
detailed predictions or fitted models. The Endterm continuity limitation therefore remains.

## Repository checks

- GitHub reported `main` as the default branch. Its complete recursive tree was checked at
  [33e7f44a79418dba8b09ac8f53bf4f457a17052b](https://github.com/Nura007/itune_ml/tree/33e7f44a79418dba8b09ac8f53bf4f457a17052b).
- The only other listed branch was `codex/midterm`. Its complete recursive tree was checked at
  [5f98d789d224299be1a9c1a3008622bb7fd5f470](https://github.com/Nura007/itune_ml/tree/5f98d789d224299be1a9c1a3008622bb7fd5f470).
- Both trees contained only `data/README.md` under `data/`. Neither contained private data,
  saved split files, fitted models or a backup ZIP.
- The [repository releases endpoint](https://api.github.com/repos/Nura007/itune_ml/releases?per_page=100)
  returned no releases.
- Commit history queries for `data/splits/split.csv` on `main` and `data/raw` on
  `codex/midterm` both returned empty results.

## Workflow artifact checks

The workflow-run listing returned 15 runs, including four runs named `Approved Midterm experiment`.
The artifact listing for pilot run [37684942987](https://github.com/Nura007/itune_ml/actions/runs/37684942987)
was empty. Each of the other three research runs had one artifact named `midterm-public-results`:

| Run | Artifact ID | ZIP size, bytes | Inspection |
| --- | --- | ---: | --- |
| [Original experiment: 37685277008](https://github.com/Nura007/itune_ml/actions/runs/37685277008) | 11511460388 | 1,183,688 | Downloaded; all 21 members inspected |
| [Report refresh: 37686294665](https://github.com/Nura007/itune_ml/actions/runs/37686294665) | 11511796796 | 1,175,980 | Downloaded; all 21 members inspected |
| [Main branch: 37759529123](https://github.com/Nura007/itune_ml/actions/runs/37759529123) | 11540949875 | 1,175,983 | Downloaded; all 21 members inspected |

Each ZIP contained 13 aggregate report files, four figures, the Midterm PPTX and PDF, the slide
contact image, and `notebooks/midterm.ipynb`. No ZIP contained the missing private files.
The [workflow at the original source revision](https://github.com/Nura007/itune_ml/blob/ae4e81218cf5039db10ee6e9c380423f7008807e/.github/workflows/research.yml)
explicitly uploaded these public output paths only.

Downloaded ZIP SHA-256 values matched the corresponding GitHub artifact digests:

```text
11511460388  733f44ba49d7a20ea417b73404b63295d2e3debffb9a0355198df17a1b5ac18c
11511796796  eababfc6431f8ef20d34852d9e48de0efe4fd90d6c396bfb497180b0478ca8de
11540949875  3aae38c9584a6025ef5a7b8e4177c934f0ead662f81c8b86eb0338f8292e8628
```

All three archived provenance files identify the original dataset fingerprint:

```text
5c4169bb4bf7cf5f8ae8c14110dc8aef6948a645657b869ff77c21c4631bc8fd
```

This confirms which experiment the aggregate reports describe. A fingerprint cannot reconstruct
the app records or original train/test assignments. Downloads remain in the ignored
`work/midterm_recovery/` directory. This check establishes the contents of the inspected GitHub
locations; it does not establish whether an independent backup exists elsewhere.
