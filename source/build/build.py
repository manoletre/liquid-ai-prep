import json, os, subprocess, shutil, sys
ROOT = "/workspace/ml-tutor"; REPO = "/workspace/liquid-ai-prep"
STEPS = f"{ROOT}/manim/steps"
# Rendered 1080p30 videos live in manim/media/videos/<src>/1080p30 (src = the Manim file name without .py).
def VID(src):
    base = f"{ROOT}/manim/media/videos/{src}"
    for q in ("1080p30", "720p30", "480p15"):
        if os.path.isdir(f"{base}/{q}"): return f"{base}/{q}"
    return f"{base}/1080p30"
TPL = open(f"{ROOT}/build/template.html").read()

SYM_ALL = {
 "pi":["π_θ · policy","Your LLM with weights θ: a probability for every response / next token."],
 "piold":["π_old · sampling policy","Frozen snapshot of the model that generated the batch."],
 "x":["x · prompt (state)","What the model is conditioned on."],
 "y":["y · response (action)","A generated completion, i.e. a token sequence."],
 "R":["R · reward","Score from a reward function or reward model."],
 "A":["A · advantage","How much better than expected. Positive → more likely, negative → less likely."],
 "b":["b · baseline","Typical reward for this prompt. Subtracting it cuts variance without bias."],
 "rho":["ρ · ratio","π_θ(y|x) / π_old(y|x). 1.5 = now 50% more likely than when sampled."],
 "eps":["ε · clip range","How far ρ may move from 1 before the gradient is cut (typically 0.2)."],
 "alpha":["α · learning rate","Step size of each update."],
 "mu":["μ · group mean","Average reward of the G responses to one prompt: GRPO's baseline."],
 "sigma":["σ · group std","Std of rewards inside one prompt's group (scale_rewards=\"group\")."],
 "beta":["β · KL weight","Penalty for drifting from a reference model. 0 = no reference model."],
 "H":["H · entropy","Average next-token uncertainty of the policy (nats). Lower = more peaked."],
 "TP":["TP · true positive","Actually positive and flagged. A caught case."],
 "FP":["FP · false positive","Actually negative but flagged. A false alarm (type I error)."],
 "FN":["FN · false negative","Actually positive but not flagged. A miss (type II error)."],
 "TN":["TN · true negative","Actually negative and not flagged. Correctly left alone."],
 "prec":["P · precision","TP / (TP + FP). Of everything flagged, the share that is real."],
 "rec":["R · recall (TPR, sensitivity)","TP / (TP + FN). Of all real positives, the share you caught."],
 "F1":["F1 score","Harmonic mean of precision and recall: 2PR / (P + R)."],
 "thr":["t · threshold","Flag if score ≥ t. Raising t → fewer flags → precision up, recall down."],
 "wS":["S · substitutions","Reference word replaced by a different word."],
 "wD":["D · deletions","Reference word missing from the ASR output."],
 "wI":["I · insertions","Extra word in the ASR output that isn't in the reference."],
 "wN":["N · reference length","Number of words in the reference transcript (not the hypothesis)."],
 "lambda":["λ · regularization strength","Weight on the L2/L1 penalty. Bigger λ → stronger shrink / more sparsity."],
 "sig":["σ(z) · sigmoid","1/(1+e^{−z}). Squashes a score to a probability in (0,1)."],
 "loss":["L · loss","Training objective; often −log likelihood under a noise model."],
 "grad":["∇L · gradient","Vector of partial derivatives; direction of steepest ascent of L."],
 "w":["w · weights","Model parameters updated by gradient descent."],
 "eta":["η · learning rate","Step size in w ← w − η∇L. Too big diverges; too small crawls."],
 "p":["p · predicted probability","σ(z) in logistic regression; compared to label y in the gradient (p−y)x."],
 "surp":["−log p · surprise","Cost of an outcome you gave probability p: how many halvings below 1 (bits)."],
 "Pd":["P · true distribution","What really happens: the data or the labels."],
 "Qd":["Q · model distribution","What the model predicts."],
 "CE":["H(P,Q) · cross-entropy","Average surprise when the truth is P and you predict Q. The classification loss."],
 "HP":["H(P) · entropy","Average surprise when you predict with the truth itself. 0 when certain, log K when uniform."],
 "KL":["D_KL(P‖Q) · KL divergence","H(P,Q) − H(P): extra surprise from using Q instead of P. Always ≥ 0, not symmetric."],
 "yhat":["ŷ · prediction","What the model outputs for one example; the truth y misses it by some noise y − ŷ."],
 "noise":["σ · noise width","Standard deviation of the Gaussian noise around the prediction. Doesn't change which model wins."],
 "mse":["MSE · mean squared error","(1/N) Σ (y − ŷ)². The −log likelihood under Gaussian noise, up to constants."],
 "z":["z · score (logit)","w·x (+ b). Any real number; the log-odds ln(p/(1−p))."],

 "trn":["train set","Data gradient descent fits on. Its score can't tell memorizing from learning."],
 "val":["validation (dev) set","Held-out data for every choice: λ, learning rate, epochs, which model."],
 "tst":["test set","Held-out data measured once at the very end. The number you report."],
 "kf":["k · folds","k-fold CV: k trainings, each validated on the one fold it didn't train on."],
 "leak":["leakage","Information the model couldn't have at prediction time reaching training (test data, the future, the label)."],
 "delta":["δ · delta","∂L/∂(node): how much the loss moves per small nudge of that node. Passed backward."],
}
MACROS = {"\\pit":["pi","\\pi_\\theta"],"\\piold":["piold","\\pi_{\\text{old}}"],"\\X":["x","x"],"\\Y":["y","y"],"\\Rw":["R","R"],
 "\\Adv":["A","A"],"\\bb":["b","b"],"\\rr":["rho","\\rho"],"\\eps":["eps","\\epsilon"],"\\lr":["alpha","\\alpha"],"\\gmu":["mu","\\mu"],
 "\\gsig":["sigma","\\sigma"],"\\bt":["beta","\\beta"],"\\Hent":["H","H"],"\\th":["pi","\\theta"],"\\grad":[None,"\\nabla_\\theta"],"\\E":[None,"\\mathbb{E}"],
 "\\TP":["TP","TP"],"\\FP":["FP","FP"],"\\FN":["FN","FN"],"\\TN":["TN","TN"],"\\Prec":["prec","P"],"\\Rec":["rec","R"],"\\Fone":["F1","F_1"],
 "\\thr":["thr","t"],"\\Sw":["wS","S"],"\\Dw":["wD","D"],"\\Iw":["wI","I"],"\\Nw":["wN","N"]}

def enc(src, dst, poster_t, poster):
    if os.path.exists(dst) and os.path.exists(poster) and os.path.getmtime(dst) > os.path.getmtime(src):
        return  # already encoded from this render
    subprocess.run(["ffmpeg","-v","error","-y","-i",src,"-c:v","libx264","-pix_fmt","yuv420p","-profile:v","high","-crf","23","-movflags","+faststart","-an",dst],check=True)
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(poster_t),"-i",dst,"-frames:v","1","-vf","scale=1280:-1","-q:v","4",poster],check=True)

def build(L, outdir, encode=True):
    os.makedirs(f"{outdir}/media", exist_ok=True)
    chapters = []
    for c in L["chapters"]:
        j = json.load(open(f"{STEPS}/{c['scene']}.json"))
        keys = [s["key"] for s in j["steps"]]
        assert keys == [s[0] for s in c["steps"]], (c["scene"], keys)
        name = c["file"]
        if encode:
            enc(f"{VID(c['src'])}/{c['scene']}.mp4", f"{outdir}/media/{name}.mp4", j["steps"][0]["end"] - 0.2, f"{outdir}/media/{name}-poster.jpg")
        steps = []
        for (k, title, html, *_), t in zip(c["steps"], j["steps"]):
            steps.append({"key": k, "title": title, "html": html, "start": t["start"], "end": t["end"]})
        chapters.append({"title": c["title"], "short": c["short"], "video": f"media/{name}.mp4", "poster": f"media/{name}-poster.jpg",
                         "steps": steps, "check": c["check"], "hook": c.get("hook"), "duration": j["duration"]})
    for c in chapters: assert c["hook"] and c["check"], ("every chapter needs a hook (prediction) and a check (practice)", c["title"])
    data = {"id": L["id"], "title": L["title"], "h1": L["h1"], "sub": L["sub"], "drill": L.get("drill", []), "key": L["key"], "sym": {k: SYM_ALL[k] for k in SYM_ALL}, "macros": MACROS,
            "chapters": chapters, "summary": L["summary"]}
    html = TPL.replace("__TITLE__", L["title"]).replace("__H1__", L["h1"]).replace("__SUB__", L["sub"]).replace("__DATA__", json.dumps(data, ensure_ascii=False))
    open(f"{outdir}/index.html", "w").write(html)
    print("built", outdir, [len(c["steps"]) for c in chapters])

sys.path.insert(0, f"{ROOT}/build")
from content import LESSONS
if __name__ == "__main__":
    # usage: python build.py 3        (build lesson 3)   |   python build.py 1 2 3
    for n in sys.argv[1:] or list(LESSONS):
        L = LESSONS[n]
        build(L, f"{REPO}/lessons/{L['slug']}")
        # drop media files that the lesson no longer uses
        used = {f"{c['file']}{ext}" for c in L["chapters"] for ext in (".mp4", "-poster.jpg")}
        md = f"{REPO}/lessons/{L['slug']}/media"
        for f in os.listdir(md):
            if f not in used: os.remove(f"{md}/{f}"); print("removed unused", f)
