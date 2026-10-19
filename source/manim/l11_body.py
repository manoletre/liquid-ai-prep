# Lesson 11 · LoRA: low-rank adapters for fine-tuning (Liquid / post-training)
C_W = "#7aa2f7"; C_A = "#e0af68"; C_B = "#9ece6a"; C_SC = "#b392f0"
C_DR = "#f7768e"; C_OK = "#9ece6a"; C_NO = "#f7768e"; C_VA = "#e0af68"
C_KL = "#5fc9b6"; C_LK = "#ff9e64"; C_CL = "#f7768e"; C_AU = "#5fc9b6"; C_BP = "#ff9e64"

def rows11(*items, buff=0.3, size=28):
    out = VGroup()
    for it in items:
        if isinstance(it, tuple):
            out.add(Text(it[0], font_size=it[2] if len(it) > 2 else size, color=it[1]))
        else:
            out.add(Text(it, font_size=size))
    return out.arrange(DOWN, aligned_edge=LEFT, buff=buff)

def fit11(g, maxw=13.2):
    if g.width > maxw:
        g.scale_to_fit_width(maxw)
    return g

def mat_grid(rows, cols, cell=0.38, gap=0.04, color=C_W, opacity=0.85, label=None, fs=22):
    """Visual d×d / d×r matrix as a grid of small rectangles."""
    g = VGroup()
    for i in range(rows):
        for j in range(cols):
            r = Rectangle(
                width=cell, height=cell, stroke_color="#0e1117", stroke_width=2,
                fill_color=color, fill_opacity=opacity,
            ).move_to([(j - (cols - 1) / 2) * (cell + gap), ((rows - 1) / 2 - i) * (cell + gap), 0])
            g.add(r)
    if label is not None:
        lab = Text(label, font_size=fs, color=color).next_to(g, DOWN, buff=0.2)
        return VGroup(g, lab)
    return g

# ---------------------------------------------------------------- Ch1: Recall
class Recall11(StepScene):
    def construct(self):
        title = caption("Recall first · four ideas due today", 32).to_edge(UP, buff=0.35)
        # 1 KL asymmetry / CE = H + KL (Lesson 06) — new numbers
        self.begin("kl")
        self.play(FadeIn(title), run_time=0.7)
        h = Text("Lesson 6 · CE = H + KL, and KL is not symmetric", font_size=28, color=C_KL).next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(h), run_time=0.7)
        kl = rows11(
            ("H(P) = 0.9 bits,  H(P, Q) = 1.3 bits  →  KL = CE − H = 0.4 bits", C_KL, 27),
            ("P = [⅔, ⅓],  Q = [1, 0]  (made-up two-way)", MUTED, 25),
            ("D(P ‖ Q): P puts mass on outcome 2 where Q(2) = 0  →  ∞", C_NO, 27),
            ("D(Q ‖ P) = 1 · log₂(1 / ⅔) = log₂(3/2) ≈ 0.585 bits  (finite)", C_OK, 27),
            buff=0.32,
        ).shift(DOWN * 0.7)
        fit11(kl)
        for it in kl:
            self.play(FadeIn(it), run_time=0.65)
            self.wait(0.3)
        self.say("CE = H + KL; swapping P and Q can turn finite into ∞", C_KL, 26)
        self.end()
        # 2 train/val/test optimism (Lesson 10)
        self.begin("opt")
        self.play(FadeOut(VGroup(h, kl)), run_time=0.5)
        h2 = Text("Lesson 10 · you tried 8 learning rates on the SAME test set", font_size=27, color=C_LK).next_to(title, DOWN, buff=0.4)
        fit11(h2)
        self.play(FadeIn(h2), run_time=0.7)
        op = rows11(
            ("best of 8 lands at 64%  (made-up; true rate ≈ 55%)", C_NO, 28),
            ("you chose ON the test set → the winner partly won by luck", C_NO, 26),
            ("the test score is optimistic; the test set became validation", C_LK, 26),
            ("right: pick on a separate val/dev set, run test once, report that", C_OK, 26),
            buff=0.34,
        ).shift(DOWN * 0.5)
        fit11(op)
        for it in op:
            self.play(FadeIn(it), run_time=0.65)
            self.wait(0.3)
        self.say("every look that changes a decision leaks a little of the test set", C_LK, 26)
        self.end()
        # 3 PPO clip for A < 0 (concrete)
        self.begin("clip")
        self.play(FadeOut(VGroup(h2, op)), run_time=0.5)
        h3 = Text("Lesson 1 · PPO clip, bad response A = −2, ε = 0.2", font_size=28, color=C_CL).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h3), run_time=0.7)
        f = MathTex(
            r"\min\big(\rho A,\ \mathrm{clip}(\rho,\,0.8,\,1.2)\,A\big)",
            font_size=42,
        ).next_to(h3, DOWN, buff=0.35)
        self.play(Write(f), run_time=1.2)
        cl = rows11(
            ("ρ = 0.95:  ρA = −1.90,  clip·A = −1.90  → min −1.90  (inside, not clipped)", C_OK, 25),
            ("ρ = 0.6:   ρA = −1.20,  clip·A = 0.8·(−2) = −1.60  → min −1.60", C_NO, 25),
            ("the clipped term wins and is flat in ρ → gradient 0", C_NO, 26),
            buff=0.34,
        ).shift(DOWN * 1.15)
        fit11(cl)
        for it in cl:
            self.play(FadeIn(it), run_time=0.65)
            self.wait(0.3)
        self.say("for A < 0, the floor is 1 − ε: already rare enough for this batch", C_CL, 26)
        self.end()
        # 4 AUROC with a tie
        self.begin("auroc")
        self.play(FadeOut(VGroup(h3, f, cl)), run_time=0.5)
        h4 = Text("Lesson 3 · AUROC: positives {0.9, 0.5}, negatives {0.5, 0.2}", font_size=27, color=C_AU).next_to(title, DOWN, buff=0.4)
        fit11(h4)
        self.play(FadeIn(h4), run_time=0.7)
        au = rows11(
            ("2 × 2 = 4 pairs", MUTED, 25),
            ("0.9 beats 0.5 and 0.2  → 1 + 1 = 2", C_OK, 27),
            ("0.5 vs 0.5 tie → ½,  beats 0.2 → 1    total 1.5", C_VA, 27),
            ("AUROC = (2 + 1.5) / 4 = 3.5 / 4 = 0.875", C_AU, 30),
            buff=0.34,
        ).shift(DOWN * 0.55)
        for it in au:
            self.play(FadeIn(it), run_time=0.65)
            self.wait(0.3)
        self.say("ties count ½; chance a random positive outscores a random negative", C_AU, 26)
        self.end(2.2)

# ---------------------------------------------------------------- Ch2: What LoRA is
class WhatLora11(StepScene):
    def construct(self):
        title = caption("LoRA: freeze W, train a thin low-rank update", 32).to_edge(UP, buff=0.35)
        # 1 intuition
        self.begin("idea")
        self.play(FadeIn(title), run_time=0.7)
        idea = rows11(
            ("full fine-tune: every entry of a big weight matrix W moves", MUTED, 27),
            ("observation: the useful update ΔW often lives in a thin subspace", FG, 27),
            ("LoRA: keep W frozen, learn a low-rank ΔW instead", C_SC, 28),
            ("from Lesson 9: frozen W gets no ∂L/∂W stored — adapters do", C_BP, 26),
            buff=0.36,
        ).shift(UP * 0.4)
        fit11(idea)
        for it in idea:
            self.play(FadeIn(it), run_time=0.75)
            self.wait(0.35)
        self.say("most of the fine-tune change is low-rank — so only train that part", C_SC, 26)
        self.end()
        # 2 formula ΔW = B A
        self.begin("formula")
        self.play(FadeOut(idea), run_time=0.5)
        eq = MathTex(r"W' = W + BA", font_size=56).next_to(title, DOWN, buff=0.55)
        self.play(Write(eq), run_time=1.3)
        labs = rows11(
            ("W  : frozen pretrained weights (d_out × d_in)", C_W, 27),
            ("B  : d_out × r     A  : r × d_in     (trainable)", C_A, 27),
            ("rank r ≪ d  →  BA has rank at most r", C_SC, 27),
            buff=0.4,
        ).shift(DOWN * 0.6)
        fit11(labs)
        for it in labs:
            self.play(FadeIn(it), run_time=0.75)
            self.wait(0.35)
        self.say("forward: y = x Wᵀ + scale · x Aᵀ Bᵀ   (same as (W + BA)ᵀ)", FG, 24)
        self.end()
        # 3 param-count worked example
        self.begin("ex1")
        self.play(FadeOut(VGroup(eq, labs)), run_time=0.5)
        hd = Text("worked example 1 · one attention projection, d = 4096 (made-up)", font_size=26, color=MUTED).next_to(title, DOWN, buff=0.35)
        fit11(hd)
        self.play(FadeIn(hd), run_time=0.6)
        # visual: big square vs thin factors
        big = mat_grid(8, 8, cell=0.32, color=C_W, label="W : 4096² params", fs=22)
        thin_a = mat_grid(2, 8, cell=0.32, color=C_A, label="A : r × d", fs=20)
        thin_b = mat_grid(8, 2, cell=0.32, color=C_B, label="B : d × r", fs=20)
        big.move_to([-4.2, 0.3, 0])
        thin_b.move_to([0.2, 0.3, 0])
        thin_a.move_to([3.6, 0.3, 0])
        self.play(FadeIn(big), run_time=0.8)
        self.play(FadeIn(thin_b), FadeIn(thin_a), run_time=0.9)
        nums = rows11(
            ("full W:  4096 × 4096 ≈ 16.8 M trainable", C_W, 26),
            ("LoRA r = 8:  2 · 4096 · 8 = 65,536 trainable  (~250× fewer)", C_OK, 26),
            buff=0.3,
        ).shift(DOWN * 2.0)
        fit11(nums)
        for it in nums:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("trainable params ≈ 2 · d · r  (per adapted matrix)", C_OK, 28)
        self.end()
        # 4 edge: r = d recovers full rank
        self.begin("ex2")
        self.play(FadeOut(VGroup(hd, big, thin_a, thin_b, nums)), run_time=0.5)
        hd2 = Text("worked example 2 (edge) · what if r is huge?", font_size=28, color=C_NO).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd2), run_time=0.6)
        ed = rows11(
            ("r = d = 4096:  BA can be ANY d × d matrix (full rank)", C_NO, 27),
            ("trainable: 2 · 4096 · 4096 ≈ 33.5 M  — MORE than full FT of W", C_NO, 26),
            ("(two factors, not one) → LoRA's win disappears", MUTED, 25),
            ("practice: r stays small — 4, 8, 16, 32 in real GRPO runs", C_OK, 26),
            ("your world: LoRA r = 32 is a common GRPO default", C_SC, 26),
            buff=0.32,
        ).shift(DOWN * 0.35)
        fit11(ed)
        for it in ed:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("r ≪ d is the whole point; r = d is not \"better LoRA\"", C_NO, 26)
        self.end()
        # 5 faded
        self.begin("faded")
        self.play(FadeOut(VGroup(hd2, ed)), run_time=0.5)
        fd = rows11(
            ("your turn · d = 1024, r = 16  (one matrix)", MUTED, 26),
            ("full W trainable params = 1024 × 1024 = ?", FG, 28),
            ("LoRA trainable = 2 · 1024 · 16 = ?", C_AL, 28),
            buff=0.4,
        ).shift(UP * 0.2)
        for it in fd[:-1]:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.play(FadeIn(fd[-1]), Create(SurroundingRectangle(fd[-1], color=C_AL, buff=0.15)), run_time=0.8)
        self.say("multiply: 2 · d · r", C_AL, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch3: r, alpha, dropout
class Knobs11(StepScene):
    def construct(self):
        title = caption("Three knobs: rank r, alpha α, and dropout", 32).to_edge(UP, buff=0.35)
        # 1 overview
        self.begin("knobs")
        self.play(FadeIn(title), run_time=0.7)
        kb = rows11(
            ("r (rank)     → capacity of the update (how rich BA can be)", C_A, 27),
            ("α (alpha)    → strength of the update (how hard it pushes)", C_SC, 27),
            ("dropout      → random zeroing on the adapter path (regularize)", C_DR, 27),
            ("effective scale in the forward pass:  α / r", C_SC, 28),
            buff=0.38,
        ).shift(UP * 0.35)
        fit11(kb)
        for it in kb:
            self.play(FadeIn(it), run_time=0.75)
            self.wait(0.35)
        self.say("y = x Wᵀ + (α/r) · (x Aᵀ) Bᵀ", C_SC, 28)
        self.end()
        # 2 why α/r
        self.begin("scale")
        self.play(FadeOut(kb), run_time=0.5)
        sc = rows11(
            ("why divide by r?", MUTED, 26),
            ("A, B are init'd small; without scaling, bigger r → bigger BA", FG, 26),
            ("α/r keeps the initial update size stable when you change r", C_SC, 27),
            ("α = r  →  scale = 1  (plain BA, no extra gain)", C_OK, 27),
            ("α = 2r →  scale = 2  (adapter hits twice as hard)", C_A, 27),
            buff=0.34,
        ).shift(UP * 0.2)
        fit11(sc)
        for it in sc:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("tune α for strength; r for capacity — they are not the same knob", FG, 26)
        self.end()
        # 3 worked example 1: r=8 vs r=32
        self.begin("ex1")
        self.play(FadeOut(sc), run_time=0.5)
        hd = Text("worked example 1 · α fixed at 16 (made-up)", font_size=27, color=MUTED).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd), run_time=0.6)
        ex = rows11(
            ("r = 8,  α = 16  →  scale = 16/8 = 2", C_A, 28),
            ("r = 32, α = 16  →  scale = 16/32 = 0.5", C_SC, 28),
            ("same α, bigger r: MORE capacity, WEAKER per-step push", FG, 26),
            ("common recipe: set α = 2r  so scale stays 2 as you sweep r", C_OK, 26),
            ("(or α = r so scale = 1 — both are used in practice)", MUTED, 24),
            buff=0.32,
        ).shift(DOWN * 0.45)
        fit11(ex)
        for it in ex:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("changing r without touching α silently changes the learning dynamics", C_A, 24)
        self.end()
        # 4 worked example 2 edge: α = r and dropout
        self.begin("ex2")
        self.play(FadeOut(VGroup(hd, ex)), run_time=0.5)
        hd2 = Text("worked example 2 (edge) · α = r, and what dropout does", font_size=26, color=C_DR).next_to(title, DOWN, buff=0.35)
        fit11(hd2)
        self.play(FadeIn(hd2), run_time=0.6)
        ed = rows11(
            ("α = 32, r = 32  →  scale = 1  (edge: no extra gain)", C_OK, 27),
            ("dropout p = 0.05 on the LoRA path (PEFT default-ish)", C_DR, 26),
            ("each forward: randomly zero some adapter activations", C_DR, 26),
            ("forces the update not to rely on one lucky direction", FG, 26),
            ("too high (e.g. 0.5): adapter barely learns — underfit the task", C_NO, 26),
            buff=0.3,
        ).shift(DOWN * 0.4)
        fit11(ed)
        for it in ed:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("dropout regularizes the adapter; α/r sets its strength", C_DR, 26)
        self.end()
        # 5 faded
        self.begin("faded")
        self.play(FadeOut(VGroup(hd2, ed)), run_time=0.5)
        fd = rows11(
            ("your turn · r = 16, α = 32  (GRPO-flavoured)", MUTED, 26),
            ("scale = α / r = ?", C_AL, 30),
            ("if you bump r to 64 and leave α = 32, new scale = ?", C_AL, 28),
            buff=0.4,
        ).shift(UP * 0.15)
        self.play(FadeIn(fd[0]), run_time=0.6)
        self.wait(0.2)
        self.play(FadeIn(fd[1]), run_time=0.7)
        self.wait(0.3)
        self.play(FadeIn(fd[2]), Create(SurroundingRectangle(fd[2], color=C_AL, buff=0.15)), run_time=0.8)
        self.say("divide α by r each time", C_AL, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch4: When LoRA vs full fine-tune
class WhenLora11(StepScene):
    def construct(self):
        title = caption("When LoRA, when full fine-tune / more params", 32).to_edge(UP, buff=0.35)
        # 1 decision axes
        self.begin("axes")
        self.play(FadeIn(title), run_time=0.7)
        ax = rows11(
            ("four practical axes (interview checklist)", MUTED, 25),
            ("1. data size / how new is the skill?", FG, 27),
            ("2. risk of forgetting the base model", FG, 27),
            ("3. serving: many adapters vs one big model", FG, 27),
            ("4. memory / how many GPUs do you have?", FG, 27),
            buff=0.34,
        ).shift(UP * 0.25)
        fit11(ax)
        for it in ax:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("LoRA is a bet that a thin update is enough for this task", C_SC, 26)
        self.end()
        # 2 worked scenario 1: LoRA wins
        self.begin("ex1")
        self.play(FadeOut(ax), run_time=0.5)
        hd = Text("worked example 1 · LoRA is the right call", font_size=28, color=C_OK).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd), run_time=0.6)
        w1 = rows11(
            ("base: strong chat LLM; task: company FAQ tone (few thousand pairs)", FG, 25),
            ("small domain shift, base already knows language", C_OK, 26),
            ("want to keep general ability (less catastrophic forgetting)", C_OK, 26),
            ("ship 5 customer-specific adapters on one frozen backbone", C_OK, 26),
            ("1×24 GB GPU: full FT of 7B doesn't fit; LoRA does", C_OK, 26),
            ("→ LoRA r = 8…32, α ≈ 2r, light dropout", C_SC, 26),
            buff=0.28,
        ).shift(DOWN * 0.45)
        fit11(w1)
        for it in w1:
            self.play(FadeIn(it), run_time=0.65)
            self.wait(0.25)
        self.say("thin update + multi-tenant serving + memory → LoRA", C_OK, 26)
        self.end()
        # 3 worked scenario 2 edge: full FT / more params
        self.begin("ex2")
        self.play(FadeOut(VGroup(hd, w1)), run_time=0.5)
        hd2 = Text("worked example 2 (edge) · when LoRA is not enough", font_size=27, color=C_NO).next_to(title, DOWN, buff=0.4)
        fit11(hd2)
        self.play(FadeIn(hd2), run_time=0.6)
        w2 = rows11(
            ("task: teach a brand-new modality or a deep new capability", FG, 26),
            ("huge curated dataset; base model is a weak starting point", C_NO, 26),
            ("LoRA plateaus: val loss stuck while full FT keeps falling", C_NO, 26),
            ("options: raise r (16→64→256), adapt more modules,", C_A, 25),
            ("          or full / almost-full fine-tune if you can afford it", C_A, 25),
            ("Liquid-style post-training: start LoRA; escalate if capacity-bound", C_SC, 25),
            buff=0.28,
        ).shift(DOWN * 0.4)
        fit11(w2)
        for it in w2:
            self.play(FadeIn(it), run_time=0.65)
            self.wait(0.25)
        self.say("tiny domain shift → LoRA; big new skill → more params / full FT", C_NO, 26)
        self.end()
        # 4 decision summary table-ish
        self.begin("table")
        self.play(FadeOut(VGroup(hd2, w2)), run_time=0.5)
        tb = rows11(
            ("prefer LoRA when…", C_OK, 28),
            ("  small–medium data · keep base skills · many task adapters · tight VRAM", C_OK, 24),
            ("prefer more params / full FT when…", C_NO, 28),
            ("  large data · new capability · LoRA saturates · one model to ship", C_NO, 24),
            ("middle path: bigger r, more layers, or QLoRA (base quantized, adapters FP)", MUTED, 24),
            buff=0.3,
        ).shift(DOWN * 0.15)
        fit11(tb)
        for it in tb:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.3)
        self.say("say the tradeoff out loud in the interview, then pick", FG, 28)
        self.end()
        # 5 faded
        self.begin("faded")
        self.play(FadeOut(tb), run_time=0.5)
        fd = rows11(
            ("your turn · two scenarios (pick LoRA or full FT)", MUTED, 25),
            ("A: GRPO on math with 50k prompts, 1 GPU, keep chat ability", C_AL, 26),
            ("B: train a 1B model from a weak base on 2T new tokens", C_AL, 26),
            buff=0.4,
        ).shift(UP * 0.2)
        for it in fd[:-1]:
            self.play(FadeIn(it), run_time=0.7)
            self.wait(0.25)
        self.play(FadeIn(fd[-1]), Create(SurroundingRectangle(fd[-1], color=C_AL, buff=0.15)), run_time=0.8)
        self.say("A → LoRA; B → full (pre)training — say why", C_AL, 26)
        self.end(2.2)
