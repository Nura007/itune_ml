"""Readable seven-minute PDF presentation generated from measured aggregates."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from .common import ROOT

INK = "#123047"
BLUE = "#176B83"
GOLD = "#B56624"
GRAY = "#536775"
W, H = 960, 540


def build_slides(public_dir):
    out = Path(public_dir)
    team = json.loads((ROOT / "config/team.json").read_text(encoding="utf-8"))
    speakers = {number: student["name"] for student in team["students"] for number in student["slides"]}
    names = "; ".join(student["name"] for student in team["students"])
    reports = out / "reports"
    read = lambda name: json.loads((reports / name).read_text(encoding="utf-8"))
    provenance, selection = read("provenance.json"), read("model_selection.json")
    audit, pca, mlp = read("cleaning_audit.json"), read("unsupervised.json"), read("mlp_diagnostics.json")
    cv = pd.read_csv(reports / "cv_metrics.csv").set_index("model")
    test = pd.read_csv(reports / "test_metrics.csv").set_index("model")
    paired = pd.read_csv(reports / "paired_errors.csv").set_index("model")
    selected = selection["selected_model"]
    earlier = selection["best_earlier_model"]
    chosen = test.loc[selected]
    delta = chosen.f1 - test.loc["Dummy", "f1"]
    pca_delta = cv.loc["SVM + PCA", "cv_f1_mean"] - cv.loc["SVM", "cv_f1_mean"]
    timings = [35, 45, 50, 50, 55, 50, 55, 45, 35]
    pdf = canvas.Canvas(str(out / "Endterm_Presentation.pdf"), pagesize=(W, H))
    pdf.setTitle("Endterm: App Store rating classification")
    pdf.setAuthor(names + "; " + team["group"] + "; Codex assistance disclosed")
    notes = ["# Seven-minute speaker notes\n",
             f"Team: {names}. Group: {team['group']}.\n\n"
             "These are assigned responsibilities and proposed speaking roles, not completed-work claims. "
             "Total planned time: 420 seconds. Every student must understand all slides. "
             "Allow five additional minutes for questions. The rubric describes two partners; "
             "instructor acceptance of the supplied three-person team is unverified.\n",
             "| Student | Assigned responsibility | Slides | Seconds |\n"
             "| --- | --- | --- | --- |\n" + "\n".join(
                 f"| {student['name']} | {student['responsibility']} | "
                 f"{student['slides'][0]}-{student['slides'][-1]} | {student['speaking_seconds']} |"
                 for student in team["students"]) + "\n"]

    def text(value, x, y, width, height, size=21, color=INK, bold=False):
        style = ParagraphStyle("slide", fontName="Helvetica-Bold" if bold else "Helvetica",
                               fontSize=size, leading=size*1.24, textColor=HexColor(color))
        paragraph = Paragraph(str(value), style)
        _, used = paragraph.wrap(width, height)
        if used > height + .1:
            raise ValueError(f"Slide text overflows: {value[:80]} ({used}>{height})")
        paragraph.drawOn(pdf, x, H-y-used)

    def start(number, title):
        pdf.setFillColor(HexColor("#F7F9FA"))
        pdf.rect(0, 0, W, H, fill=1, stroke=0)
        pdf.setFillColor(HexColor(BLUE))
        pdf.rect(0, H-12, W, 12, fill=1, stroke=0)
        text(title, 45, 38, 870, 88, 31, bold=True)
        pdf.setStrokeColor(HexColor("#D9E2E8"))
        pdf.line(45, 37, 915, 37)
        text(f"Separate November 2026 experiment  /  {number}/9", 45, 511, 700, 20, 11, GRAY)
        text(f"{speakers[number]}  {timings[number-1]}s", 690, 511, 230, 20, 11, GRAY)

    def note(number, title, words):
        elapsed = sum(timings[:number-1])
        notes.append(f"## {number}. {title}\n\nSpeaker: {speakers[number]}; {timings[number-1]} seconds; "
                     f"{elapsed//60}:{elapsed%60:02d}-{(elapsed+timings[number-1])//60}:"
                     f"{(elapsed+timings[number-1])%60:02d}\n\n{words}\n")
        pdf.showPage()

    def picture(name, x, y, width, height):
        pdf.drawImage(str(out / "figures" / name), x, H-y-height,
                      width=width, height=height, preserveAspectRatio=True, anchor="c", mask="auto")

    start(1, "Can app metadata identify highly rated apps?")
    text("ENDTERM  /  MACHINE LEARNING ALGORITHMS", 48, 137, 850, 35, 15, BLUE, True)
    text("Seven inputs. Eight models.<br/>One frozen test set in the new experiment.",
         48, 192, 840, 135, 34, bold=True)
    text(f"{escape(selected)} selected by training CV. Test F1 {chosen.f1:.3f}; "
         f"class-0 recall {chosen.specificity:.1%}.", 48, 365, 850, 75, 23, BLUE)
    text(f"{escape(names)}<br/>Group {escape(team['group'])}", 48, 439, 850, 48, 16, GRAY)
    note(1, "Question and main result",
         f"Our question is whether seven metadata attributes can identify highly rated existing US App Store apps. "
         f"Highly rated means at least 4.5 stars, and every app has at least 50 ratings. "
         f"We compared eight methods. Training cross-validation selected {selected}. "
         f"Its test F1 was {chosen.f1:.3f}, but class-zero recall was only {chosen.specificity:.1%}. "
         "That second number matters because the positive class dominates this sample.")

    start(2, "Continuity is the unresolved requirement")
    text("October Midterm", 48, 145, 380, 40, 24, BLUE, True)
    text("2,029 apps<br/>Original private snapshot and split were lost.<br/>Public aggregates survive.",
         48, 205, 390, 180, 23)
    text("November Endterm", 510, 145, 400, 40, 24, BLUE, True)
    text(f"{audit['clean_rows']:,} real API apps<br/>{provenance['train_n']:,} train / "
         f"{provenance['test_n']:,} test<br/>Raw data and split now retained privately.",
         510, 205, 400, 180, 23)
    text("New snapshot: no claim of improvement on the original holdout", 48, 424, 850, 55, 22, GOLD, True)
    note(2, "Data and continuity",
         "The original cloud run retained public results but lost its private app records and split manifest. "
         "A checksum cannot reconstruct those missing rows. We therefore state a continuity gap rather than "
         f"claiming a complete continuation. The separately collected November dataset has {audit['clean_rows']:,} "
         f"eligible apps, split into {provenance['train_n']:,} training and {provenance['test_n']:,} test rows. "
         "All baselines and new models use this same new split. Raw response checksums, split IDs and models "
         "are now retained in the private submission. Restoring the original archive is still needed to meet "
         "the rubric's same-data condition. No instructor exception has been assumed.")

    start(3, "Every choice is made inside training data")
    rows = [("Inputs", "Price, size, languages, release year, update recency, genre, age category"),
            ("Outer CV", "Five shared stratified folds compare all eight methods"),
            ("Inner CV", "Three folds tune forest and boosting inside each outer training fold"),
            ("Final test", "Evaluate after CV selection; keep default decision thresholds")]
    for index, (label, detail) in enumerate(rows):
        y = 145 + index*76
        text(label, 48, y, 170, 50, 22, BLUE, True)
        text(detail, 245, y, 650, 63, 22)
    note(3, "Fair evaluation",
         "We kept the original seven raw inputs. User ratings define the label, and rating counts define "
         "eligibility, but neither enters a model. Identifiers and text are also excluded. Imputation, scaling "
         "and one-hot encoding are fitted within training folds. Every method uses the same five outer folds. "
         "For each ensemble, a three-fold inner search chooses parameters without seeing the outer validation "
         "rows. A final search on all training rows produces the fitted ensemble. Positive-class F1 remains "
         "primary, while AUC, balanced accuracy and class-zero recall check discrimination. The test set is "
         "used only after selection, and thresholds are unchanged.")

    start(4, "One comparison includes every baseline")
    headers = ["Model", "CV F1 +/- SD", "Test F1", "AUC", "Bal. acc."]
    positions = [48, 325, 550, 675, 795]
    widths = [270, 210, 115, 110, 120]
    for x, width, label in zip(positions, widths, headers):
        text(label, x, 142, width, 36, 17, BLUE, True)
    for i, name in enumerate(cv.index):
        v, t = cv.loc[name], test.loc[name]
        values = [name + (" *" if name == selected else ""),
                  f"{v.cv_f1_mean:.3f} +/- {v.cv_f1_std:.3f}", f"{t.f1:.3f}",
                  f"{t.roc_auc:.3f}", f"{t.balanced_accuracy:.3f}"]
        for x, width, value in zip(positions, widths, values):
            text(escape(value), x, 184+i*33, width, 32, 17, bold=name==selected)
    text("* Selected by CV. Spread is sample SD. All test rows are identical.",
         48, 461, 850, 25, 14, GRAY)
    note(4, "Complete comparison",
         f"The table compares the majority baseline, three earlier classifiers, two tuned ensembles, a PCA "
         f"ablation and the MLP. {selected} has the highest mean outer CV F1 and is marked with an asterisk. "
         f"Its test F1 difference from Dummy is {delta:+.4f}. A high F1 alone is not strong evidence of useful "
         "classification: the majority model receives a high value while finding no lower-rated apps. "
         "The spread shown here is the sample standard deviation across five folds, not a confidence interval. "
         "We did not change the winner after inspecting test AUC or balanced accuracy.")

    start(5, "Small, explicit searches control complexity")
    for i, name in enumerate(["Random Forest", "Gradient Boosting"]):
        x = 48 + i*465
        text(name, x, 143, 420, 42, 25, BLUE, True)
        params = selection["selected_parameters"][name]
        settings_text = "<br/>".join(escape(k.replace("model__", "")) + ": " + escape(str(v))
                                    for k, v in params.items())
        text(settings_text, x, 205, 405, 155, 21)
        text(f"CV F1 {cv.loc[name, 'cv_f1_mean']:.3f}<br/>Test F1 {test.loc[name, 'f1']:.3f}",
             x, 374, 405, 75, 24, BLUE, True)
    note(5, "Tuned ensembles",
         "Both ensembles use eight prespecified parameter combinations. The random forest always uses "
         "100 trees and searches depth, minimum leaf size and the number of candidate features. Larger leaves "
         "and restricted depth reduce variance. Boosting searches 75 or 150 rounds, learning rates of 0.03 or "
         "0.1, and depths one or two. This balances correction strength against overfitting. The selected "
         "full-training settings are shown here; outer folds may select different settings. "
         f"Random forest achieved outer CV F1 {cv.loc['Random Forest', 'cv_f1_mean']:.3f}, and boosting "
         f"achieved {cv.loc['Gradient Boosting', 'cv_f1_mean']:.3f}. These are estimates of the tuning "
         "procedures, not the optimistic best scores from the final search.")

    start(6, "PCA describes variation, then tests its value")
    picture("pca_variance.png", 30, 140, 535, 300)
    text(f"{pca['components_for_90_percent']} components retain 90% variance.<br/><br/>"
         f"First two: {pca['first_two_explained_variance']:.1%}.<br/><br/>"
         f"PCA - SVM CV F1:<br/>{pca_delta:+.4f}", 595, 151, 320, 264, 23, BLUE)
    text("Numeric inputs standardized; one-hot categories stay 0/1. No claim of natural clusters.",
         48, 458, 855, 28, 14, GRAY)
    loads = pca["dominant_loadings"]["PC1"][:2]
    load_names = ", ".join(v["feature"].replace("numeric__", "").replace("categorical__", "") for v in loads)
    note(6, "Unsupervised structure",
         f"Training-only PCA requires {pca['components_for_90_percent']} components to retain 90 percent of "
         f"the transformed variance. The first two retain {pca['first_two_explained_variance']:.1%}. "
         f"The largest PC1 loadings include {load_names}. The report connects signed loadings to median "
         "feature profiles and two coarse examples of actual training apps. Numeric attributes are "
         "standardized, while categorical indicators remain zero or one, so this is an explicit weighting "
         "choice. A density map is not proof of real groups. To test usefulness, we refit PCA inside each "
         f"SVM training fold. The change in mean CV F1 was {pca_delta:+.4f}. PCA maximizes variance, not "
         "the ability to distinguish ratings, so compression need not improve prediction.")

    start(7, "The MLP learning check reveals overfitting")
    picture("mlp_learning.png", 30, 140, 560, 290)
    text("32 + 16 hidden units<br/>ReLU / Adam / 120 epochs", 610, 147, 300, 88, 23, BLUE, True)
    text(f"Validation-loss minimum:<br/>epoch {mlp['best_observed_validation_epoch']}<br/><br/>"
         f"MLP CV F1: {cv.loc['MLP', 'cv_f1_mean']:.3f}<br/>"
         f"Earlier best: {escape(earlier)}<br/>{cv.loc[earlier, 'cv_f1_mean']:.3f}",
         610, 262, 300, 185, 20)
    note(7, "Neural network learning",
         "Our MLP has hidden layers of 32 and 16 ReLU units, trained with Adam at learning rate 0.001 "
         "and L2 regularization 0.001. It uses a fixed 120 epochs. The learning curves come from a separate "
         "80/20 split inside the training partition, with preprocessing fitted only on the diagnostic "
         f"training subset. Validation log loss was lowest at epoch {mlp['best_observed_validation_epoch']} "
         f"and then increased by {mlp['validation_loss_above_observed_minimum']:.3f}, while training loss "
         "fell. This supports an overfitting diagnosis. We did not retrospectively select that epoch. "
         f"The MLP's CV F1 was {cv.loc['MLP', 'cv_f1_mean']:.3f}, compared with "
         f"{cv.loc[earlier, 'cv_f1_mean']:.3f} for the CV-selected earlier reference, {earlier}. "
         "More capacity did not automatically solve the metadata limitation.")

    start(8, "Error patterns matter more than a high F1")
    text(escape(selected) + " on the holdout", 48, 145, 470, 45, 25, BLUE, True)
    labels = [("Lower-rated detected", f"{int(chosen.tn)} / {int(chosen.tn+chosen.fp)}"),
              ("False positives", str(int(chosen.fp))), ("False negatives", str(int(chosen.fn)))]
    for i, (label, value) in enumerate(labels):
        text(label, 48, 215+68*i, 330, 40, 22)
        text(value, 395, 215+68*i, 140, 40, 27, GOLD, True)
    text("All models: matrices + actual errors<br/><br/>"
         "Boundary cases and coarse metadata describe associations.<br/><br/>"
         "They do not explain why users assigned ratings.", 585, 155, 315, 275, 23)
    row = paired.loc[selected] if selected in paired.index else None
    pair_text = (f"Against SVM, the selected model alone corrected {int(row.model_only_correct)} apps "
                 f"while SVM alone corrected {int(row.svm_only_correct)}. " if row is not None else "")
    note(8, "Errors and trade-offs",
         f"The selected model identified {int(chosen.tn)} of {int(chosen.tn+chosen.fp)} lower-rated apps. "
         f"It made {int(chosen.fp)} false positives and {int(chosen.fn)} false negatives. "
         + pair_text + "The report includes confusion matrices for every model and actual anonymized error "
         "examples, with exact records retained privately. Some errors lie near the 4.5 threshold, where a "
         "small rating difference changes the label. Coarse metadata can describe these cases but cannot "
         "establish why users liked or disliked an app. Different models trade positive recall against "
         "recognizing lower-rated apps.")

    start(9, "Conclusions and a testable Final plan")
    text("Metadata offers limited discrimination", 48, 143, 850, 48, 27, BLUE, True)
    text("Query sampling, imbalance, developer overlap and snapshot drift limit generalization.",
         48, 209, 850, 66, 24)
    text("Final: permission-checked descriptions linked to the same apps. Compare text-only, "
         "metadata-only and combined models using training-fold TF-IDF.", 48, 309, 850, 100, 23)
    text("Codex assistance disclosed. Real contributions and individual explanations belong to the students.",
         48, 454, 850, 30, 14, GRAY)
    note(9, "Conclusions and Final plan",
         "The answer is cautious: metadata provides limited discrimination in this convenience sample, "
         "and high positive-class F1 hides important failures. For Final, the proposed source is retained "
         "app descriptions linked privately by trackId. We must confirm permitted use or obtain suitable "
         "licensed text before training. We will compare text-only, metadata-only and combined models "
         "using fold-fitted TF-IDF and training-only selection. Codex assisted implementation and writing. "
         "Every student must confirm their completed contributions, rehearse, and explain the entire project.")
    pdf.save()
    notes.append("## Evidence and citations\n\n"
                 "All numerical statements come from outputs/endterm/reports; this deck is regenerated by "
                 "appstore_ml.endterm_slides. The attached Endterm Rubric was read directly (four pages).\n\n"
                 "- Apple Search API: https://performance-partners.apple.com/search-api\n"
                 "- Source robots policy: https://itunes.apple.com/robots.txt\n"
                 "- Nested CV: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html\n"
                 "- MLP: https://scikit-learn.org/stable/modules/neural_networks_supervised.html\n"
                 "- PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca\n")
    (out / "speaker_notes.md").write_text("\n".join(notes), encoding="utf-8")
    return out / "Endterm_Presentation.pdf"


if __name__ == "__main__":
    from .common import ROOT
    print(build_slides(ROOT / "outputs/endterm"))
