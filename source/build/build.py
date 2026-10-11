import json, os, subprocess, shutil, sys
ROOT = "/workspace/ml-tutor"; REPO = "/workspace/liquid-ai-prep"
STEPS = f"{ROOT}/manim/steps"
VID = {"lesson01_scenes": f"{ROOT}/manim/media/videos/lesson01_scenes/1080p30", "all_scenes": f"{ROOT}/manim/media/videos/all_scenes/1080p30"}
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
}
MACROS = {"\\pit":["pi","\\pi_\\theta"],"\\piold":["piold","\\pi_{\\text{old}}"],"\\X":["x","x"],"\\Y":["y","y"],"\\Rw":["R","R"],
 "\\Adv":["A","A"],"\\bb":["b","b"],"\\rr":["rho","\\rho"],"\\eps":["eps","\\epsilon"],"\\lr":["alpha","\\alpha"],"\\gmu":["mu","\\mu"],
 "\\gsig":["sigma","\\sigma"],"\\bt":["beta","\\beta"],"\\Hent":["H","H"],"\\th":["pi","\\theta"],"\\grad":[None,"\\nabla_\\theta"],"\\E":[None,"\\mathbb{E}"]}

def enc(src, dst, poster_t, poster):
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
            enc(f"{VID[c['src']]}/{c['scene']}.mp4", f"{outdir}/media/{name}.mp4", j["steps"][0]["end"] - 0.2, f"{outdir}/media/{name}-poster.jpg")
        steps = []
        for (k, title, html, *rest), t in zip(c["steps"], j["steps"]):
            d = {"key": k, "title": title, "html": html, "start": t["start"], "end": t["end"]}
            if rest: d["resolves"] = rest[0]
            steps.append(d)
        chapters.append({"title": c["title"], "short": c["short"], "video": f"media/{name}.mp4", "poster": f"media/{name}-poster.jpg",
                         "steps": steps, "check": c["check"], "duration": j["duration"]})
    data = {"id": L["id"], "hook": L["hook"], "key": L["key"], "sym": {k: SYM_ALL[k] for k in SYM_ALL}, "macros": MACROS,
            "chapters": chapters, "summary": L["summary"]}
    html = TPL.replace("__TITLE__", L["title"]).replace("__H1__", L["h1"]).replace("__SUB__", L["sub"]).replace("__DATA__", json.dumps(data, ensure_ascii=False))
    open(f"{outdir}/index.html", "w").write(html)
    print("built", outdir, [len(c["steps"]) for c in chapters])

sys.path.insert(0, f"{ROOT}/build")
from content import L1, L2
if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2"]
    if "1" in which: build(L1, f"{REPO}/lessons/01-policy-gradient-to-ppo")
    if "2" in which: build(L2, f"{REPO}/lessons/02-explore-exploit-grpo")
