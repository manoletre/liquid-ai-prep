import json, os, subprocess, sys
REPO = "/workspace/liquid-ai-prep"
TPL = open(f"{REPO}/source/build/template.html").read()
STEPS = "/workspace/ml-tutor/manim/steps"
VID = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ml-tutor/manim/media/videos/l16_all/1080p30"
L15 = json.load(open("/tmp/l15/data.json"))
SYM = dict(L15["sym"]); MACROS = dict(L15["macros"])
SYM.update({
 "srw": ["scale_rewards · TRL GRPOConfig", "Which spread σ the advantage is divided by: \"group\" (default, each prompt's own σ), \"batch\" (one σ for the whole batch) or \"none\" (no division)."],
 "sgg": ["σ_group · group spread", "Standard deviation of the G rewards for ONE prompt: √(average of (r − μ_group)²). TRL uses the unbiased version (÷ (G − 1))."],
 "sgb": ["σ_batch · batch spread", "Standard deviation of ALL rewards in the batch (every prompt's G answers together), around the batch mean."],
 "mug": ["μ_group · group mean", "Average reward of one prompt's G answers. The baseline in every mode."],
 "zstd": ["frac_reward_zero_std", "TRL metric: share of rewards whose σ is 0 — i.e. groups where every answer scored the same and so learned nothing. Uses whichever σ scale_rewards picked."],
 "G": ["G · group size", "Number of answers sampled per prompt (num_generations). Your run: 8."],
 "pc": ["p · fraction correct", "For 0/1 rewards, share of a group's answers that are right. Its spread is σ = √(p(1 − p))."],
})
MACROS.update({"\\sgg": ["sgg", "\\sigma_{\\text{group}}"], "\\sgb": ["sgb", "\\sigma_{\\text{batch}}"], "\\mug": ["mug", "\\mu_{\\text{group}}"], "\\Gn": ["G", "G"], "\\pc": ["pc", "p"]})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content16 import L

def enc(src, dst, poster_t, poster):
    subprocess.run(["ffmpeg","-v","error","-y","-i",src,"-c:v","libx264","-pix_fmt","yuv420p","-profile:v","high","-crf","23","-movflags","+faststart","-an",dst],check=True)
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(poster_t),"-i",dst,"-frames:v","1","-vf","scale=1280:-1","-q:v","4",poster],check=True)

out = f"{REPO}/lessons/{L['slug']}"; os.makedirs(f"{out}/media", exist_ok=True)
chapters = []
for c in L["chapters"]:
    j = json.load(open(f"{STEPS}/{c['scene']}.json"))
    keys = [s["key"] for s in j["steps"]]
    assert keys == [s[0] for s in c["steps"]], (c["scene"], keys)
    enc(f"{VID}/{c['scene']}.mp4", f"{out}/media/{c['file']}.mp4", j["steps"][0]["end"] - 0.2, f"{out}/media/{c['file']}-poster.jpg")
    steps = [{"key": k, "title": t, "html": h, "start": s["start"], "end": s["end"]} for (k, t, h), s in zip(c["steps"], j["steps"])]
    chapters.append({"title": c["title"], "short": c["short"], "video": f"media/{c['file']}.mp4", "poster": f"media/{c['file']}-poster.jpg",
                     "steps": steps, "check": c["check"], "hook": c["hook"], "duration": j["duration"]})
data = {"id": L["id"], "title": L["title"], "h1": L["h1"], "sub": L["sub"], "drill": L["drill"], "key": L["key"], "sym": SYM, "macros": MACROS,
        "chapters": chapters, "summary": L["summary"]}
html = TPL.replace("__TITLE__", L["title"]).replace("__H1__", L["h1"]).replace("__SUB__", L["sub"]).replace("__DATA__", json.dumps(data, ensure_ascii=False))
open(f"{out}/index.html", "w").write(html)
print("built", out, [len(c["steps"]) for c in chapters], [c["duration"] for c in chapters])
