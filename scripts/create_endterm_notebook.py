"""Create the reproducible Endterm notebook; run it with the project's Python kernel."""
from pathlib import Path

import nbformat


def create(root):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md("# App Store rating classification: Endterm\n\n"
                "This notebook reruns the real frozen API experiment. It performs no network collection. "
                "The November snapshot is a **separate experiment**: the original Midterm raw data and "
                "split were lost. The original-data continuity criterion remains unresolved. "
                "All eight methods below share the saved split of this new snapshot.\n\n"
                "Use the private submission archive to obtain the data. Select the project virtual "
                "environment, then Restart and Run All. Runtime is several minutes on a typical CPU."),
             code("from pathlib import Path\nimport sys, json\nimport pandas as pd\n"
                  "from IPython.display import display, Markdown, Image\n"
                  "ROOT = Path.cwd()\n"
                  "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
                  "sys.path.insert(0, str(ROOT / 'src'))\n"
                  "from appstore_ml.endterm import experiment\n"
                  "OUT = ROOT / 'outputs/endterm'\n"
                  "RUN = ROOT / 'work/endterm-2026-11-08'\n"
                  "assert (RUN / 'data/raw').is_dir(), 'Restore the private frozen run before execution.'"),
             md("## Research question and locked analysis\n\n"
                "Can seven metadata inputs distinguish rating >=4.5 from <4.5 among existing US apps "
                "with at least 50 ratings? Five numeric features and two categorical features enter "
                "fold-fitted preprocessing. Ratings, rating counts, identifiers and text are excluded. "
                "F1 for class 1 is primary, with AUC, balanced accuracy and class-0 recall to expose "
                "imbalance. Forest and boosting use nested CV (5 outer / 3 inner folds). "
                "The MLP has fixed settings and training-only learning diagnostics. "
                "No hyperparameter or threshold choice uses test rows."),
             code("provenance = experiment(RUN, OUT, new_experiment=True, render=False)\n"
                  "print('Data status:', provenance['data_kind'])\n"
                  "print('Experiment:', provenance['experiment_id'])\n"
                  "print('Saved split:', provenance['train_n'], '/', provenance['test_n'])\n"
                  "print('CV-selected model:', provenance['selected_model'])"),
             md("## Data, continuity and model selection"),
             code("for name in ['cleaning_audit.json', 'continuity.json', 'model_selection.json']:\n"
                  "    display(Markdown('### ' + name))\n"
                  "    display(json.loads((OUT / 'reports' / name).read_text()))"),
             md("## One comparison on the saved split\n\nCV spread is sample SD, not a confidence interval. "
                "For tuned ensembles, outer folds evaluate the complete tuning procedure. "
                "Winning CV scores may still be optimistic after model-family selection."),
             code("display(pd.read_csv(OUT / 'reports/comparison.csv').round(4))\n"
                  "display(pd.read_csv(OUT / 'reports/tuning_results.csv').round(4))"),
             md("## PCA structure and its predictive check\n\n"
                "Descriptive PCA uses training rows only. The SVM + PCA comparison refits PCA inside "
                "every CV fold. Two-dimensional density does not establish natural app categories."),
             code("display(json.loads((OUT / 'reports/unsupervised.json').read_text()))\n"
                  "display(pd.read_csv(OUT / 'reports/pca_profiles.csv'))\n"
                  "display(Image(filename=str(OUT / 'figures/pca_variance.png')))\n"
                  "display(Image(filename=str(OUT / 'figures/pca_map.png')))"),
             md("## MLP learning\n\nThe diagnostic split is within training. Its imputer, encoder and "
                "scaler exclude the diagnostic validation rows. The fixed epoch count is not tuned on "
                "these curves or on test performance."),
             code("display(json.loads((OUT / 'reports/mlp_diagnostics.json').read_text()))\n"
                  "learning = pd.read_csv(OUT / 'reports/mlp_learning_curve.csv')\n"
                  "display(learning.iloc[[0, -1]])\n"
                  "learning.set_index('epoch').filter(like='loss').plot(title='Training-only MLP diagnostic')"),
             md("## Real errors across all models\n\n"
                "Public examples use coarse descriptions. App IDs, exact ratings, descriptions and "
                "detailed predictions are retained only in the private archive."),
             code("display(Image(filename=str(OUT / 'figures/confusion_matrices.png')))\n"
                  "display(Image(filename=str(OUT / 'figures/roc_curves.png')))\n"
                  "display(pd.read_csv(OUT / 'reports/error_cases_anonymized.csv'))\n"
                  "display(pd.read_csv(OUT / 'reports/paired_errors.csv'))\n"
                  "display(pd.read_csv(OUT / 'reports/error_profiles.csv'))"),
             md("## Interpretation, limitations and Final plan\n\n"
                "Use the generated Endterm report for the complete result interpretation. "
                "A convenience sample of established apps does not establish pre-launch prediction "
                "or causal explanations. Developer overlap and temporal drift remain limitations. "
                "For Final, investigate retained app descriptions linked privately by trackId, only "
                "after confirming permission or choosing a suitably licensed replacement. Compare "
                "train-fold TF-IDF plus a linear model with metadata-only and combined models, with "
                "the same frozen IDs and training-only selection. Keep raw text private. "
                "Do not reuse this examined test set for unlimited adaptive model choices.\n\n"
                "AI use: Codex assisted with code, tests, collection execution, analysis and writing. "
                "Students must record their real contributions and explain every part independently. "
                "No student names, hours or instructor exceptions are inferred.")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    target = root / "notebooks/endterm.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    return target


if __name__ == "__main__":
    print(create(Path(__file__).resolve().parents[1]))
