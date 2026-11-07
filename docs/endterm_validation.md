# Endterm validation record

Verified on November 8, 2026, Asia/Qyzylorda.

- `ruff check src tests scripts`: passed.
- `pytest --basetemp work/pytest-endterm-final-20261108`: **57 passed**. Only dependency
  deprecation warnings from matplotlib/pyparsing were emitted.
- The team-metadata revision also passed all 57 tests. Notebook edits changed Markdown only;
  all seven executed code cells and their original outputs were preserved.
- `notebooks/endterm.ipynb`: all **7 code cells executed**, with zero error outputs.
  Execution refitted all eight models and produced the empirical comparison from the real API snapshot.
- Dataset: 2,179 eligible apps; 1,743 train and 436 test; all models share saved split IDs.
- Private-run integrity checks reconstruct the prepared dataset from checksummed raw responses,
  verify dataset and split hashes, confirm frozen snapshot metadata and all eight fitted model files.
- Publication check: **58 aggregate artifacts validated**, with the Git index unchanged.
- Report: **11 PDF pages** visually inspected, without clipped or overlapping content.
- Presentation: **9 PDF slides** visually inspected; proposed notes total **420 seconds**.
  This timing is a plan, not a claim that any student has rehearsed or delivered the talk.

The experiment-time source hashes remain unchanged. The private archive additionally records
final packaging-time source hashes, including later documentation, reporting and validation
guard edits. No post-test model, hyperparameter, threshold or feature change was made.

The owner supplied **Nurtilek Koblandy, Erkebulan Zhuparbay and Mansur Ryskali**, group
**SE-2423**, as a three-person team. Proposed future speaking roles total 420 seconds:
Nurtilek, slides 1-3 (130 seconds); Erkebulan, slides 4-6 (155 seconds); Mansur, slides 7-9
(135 seconds). These assignments are not evidence of past work or completed rehearsal.

Remaining requirements are substantive: the original Midterm snapshot/split cannot be recovered
from this checkout, actual student contributions remain unprovided, instructor acceptance of
three students is unverified against the rubric's two-partner wording, and the individual oral
defense must be performed by the students. No complete rubric compliance or grade is claimed.
