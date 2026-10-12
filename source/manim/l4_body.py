# Lesson 4 · ML fundamentals for pen-and-paper interviews (Guava MLE)

C_BV = "#7aa2f7"; C_OV = "#f7768e"; C_UN = "#e0af68"; C_RG = "#9ece6a"
C_L2 = "#b392f0"; C_L1 = "#7dcfff"; C_DO = "#ff9e64"; C_LOSS = "#e08fd0"
C_GD = "#c3e88d"; C_SIG = "#7aa2f7"; C_BP = "#f7768e"

# ---------------------------------------------------------------- Ch1: Metrics recall
class MetricsRecall(StepScene):
    def construct(self):
        title = caption("Lesson 3 weak spots · retrieval before new material", 32).to_edge(UP, buff=0.35)
        # 1 AUROC pairs
        self.begin("auroc")
        self.play(FadeIn(title), run_time=0.7)
        h = Text("AUROC = P(random + scores above random −)", font_size=30, color=C_GOOD).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h), run_time=0.7)
        pos = Text("positives  {0.9, 0.5}", font_size=28, color=C_GOOD).shift(LEFT * 3.2 + UP * 0.6)
        neg = Text("negatives  {0.7, 0.5, 0.2}", font_size=28, color=C_BAD).shift(RIGHT * 2.8 + UP * 0.6)
        self.play(FadeIn(pos), FadeIn(neg), run_time=0.8)
        rows = VGroup(
            Text("(0.9, 0.7) → 1    (0.9, 0.5) → 1    (0.9, 0.2) → 1", font_size=26),
            Text("(0.5, 0.7) → 0    (0.5, 0.5) → ½    (0.5, 0.2) → 1", font_size=26),
            Text("sum = 4.5   ·   6 pairs   ·   AUROC = 4.5/6 = 0.75", font_size=28, color=C_AL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(DOWN * 1.0)
        self.play(FadeIn(rows), run_time=1.2)
        self.say("ties count as ½ · list every (pos, neg) pair", C_AL, 28)
        self.end()
        # 2 Fβ
        self.begin("fbeta")
        self.play(FadeOut(VGroup(h, pos, neg, rows)), run_time=0.5)
        f = MathTex(r"F_\beta=\frac{(1+\beta^2)PR}{\beta^2 P+R}", font_size=52).shift(UP * 1.4)
        self.play(Write(f), run_time=1.4)
        why = VGroup(
            Text("weights on 1/P and 1/R:  1/(1+β²)  and  β²/(1+β²)", font_size=26, color=MUTED),
            Text("they sum to 1 → if P = R = x then Fβ = x", font_size=26, color=MUTED),
            Text("β² (not β): equal ∂F/∂P and ∂F/∂R when R/P = β", font_size=26, color=C_AL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 0.5)
        self.play(FadeIn(why), run_time=1.2)
        ex = Text("P=0.8, R=0.4 → F1≈0.53 · F2≈0.44 · F0.5≈0.67", font_size=28, color=C_GOOD).to_edge(DOWN, buff=1.1)
        self.play(FadeIn(ex), run_time=0.7)
        self.say("β>1 favors recall (escalation); β<1 favors precision", FG, 28)
        self.end()
        # 3 PR vs ROC
        self.begin("pr")
        self.play(FadeOut(VGroup(f, why, ex)), run_time=0.5)
        t = Text("Rare positives (1% fraud): AUROC can look great while precision is low", font_size=28).shift(UP * 2.0)
        self.play(FadeIn(t), run_time=0.7)
        bullets = VGroup(
            Text("ROC random baseline = 0.5  (any prevalence)", font_size=28, color=C_BV),
            Text("PR  random baseline = prevalence  (0.01 if 1% positive)", font_size=28, color=C_DO),
            Text("At TPR 0.9, FPR 0.05 on 10k (100 +): TP=90, FP≈495 → P≈0.15", font_size=26, color=C_BAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 0.2)
        self.play(FadeIn(bullets), run_time=1.2)
        self.say("use PR-AUC when positives are rare (fraud, escalations)", C_DO, 28)
        self.end()
        # 4 WER
        self.begin("wer")
        self.play(FadeOut(VGroup(t, bullets)), run_time=0.5)
        w = MathTex(r"\mathrm{WER}=\frac{S+D+I}{N_{\mathrm{ref}}}", font_size=56).shift(UP * 1.5)
        self.play(Write(w), run_time=1.2)
        e1 = Text("ref “yes” (N=1) · ASR “yes yes okay” → S=0 D=0 I=2 → WER=200%", font_size=26, color=C_BAD).shift(UP * 0.2)
        e2 = Text("WER = word-level Levenshtein / reference words · can exceed 100%", font_size=26, color=MUTED).shift(DOWN * 0.6)
        self.play(FadeIn(e1), FadeIn(e2), run_time=1.0)
        self.say("insertions count; denominator is only the reference length", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch2: Bias–variance
class BiasVariance(StepScene):
    def construct(self):
        title = caption("Train vs validation error · diagnose underfit / overfit", 32).to_edge(UP, buff=0.35)
        # 1 setup
        self.begin("setup")
        self.play(FadeIn(title), run_time=0.7)
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 1.1, 0.25], x_length=9, y_length=4.2,
                    axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(DOWN * 0.35)
        xl = Text("model complexity →", font_size=24, color=MUTED).next_to(axes, DOWN, buff=0.15)
        yl = Text("error", font_size=24, color=MUTED).next_to(axes, LEFT, buff=0.1)
        self.play(Create(axes), FadeIn(xl), FadeIn(yl), run_time=1.2)
        self.say("x-axis: richer models (more params / less regularization)", FG, 28)
        self.end()
        # 2 curves
        self.begin("curves")
        train = axes.plot(lambda x: 0.55 * np.exp(-0.35 * x) + 0.08, color=C_GOOD, stroke_width=5)
        val = axes.plot(lambda x: 0.25 + 0.35 * ((x - 4.5) / 4.5) ** 2 + 0.05 * np.exp(-0.4 * x), color=C_BAD, stroke_width=5)
        lt = Text("train error", font_size=24, color=C_GOOD).move_to(axes.c2p(8.2, 0.18))
        lv = Text("val error", font_size=24, color=C_BAD).move_to(axes.c2p(8.2, 0.72))
        self.play(Create(train), FadeIn(lt), run_time=1.2)
        self.play(Create(val), FadeIn(lv), run_time=1.2)
        self.say("train falls; val falls then rises — the rise is overfitting", FG, 28)
        self.end()
        # 3 underfit
        self.begin("under")
        u = DashedLine(axes.c2p(1.2, 0), axes.c2p(1.2, 1.05), color=C_UN, stroke_width=3)
        ul = Text("underfit", font_size=26, color=C_UN).next_to(u, UP, buff=0.1)
        self.play(Create(u), FadeIn(ul), run_time=0.9)
        self.say("both errors high · model too simple · add capacity or train longer", C_UN, 28)
        self.end()
        # 4 overfit
        self.begin("over")
        o = DashedLine(axes.c2p(8.0, 0), axes.c2p(8.0, 1.05), color=C_OV, stroke_width=3)
        ol = Text("overfit", font_size=26, color=C_OV).next_to(o, UP, buff=0.1)
        self.play(Create(o), FadeIn(ol), run_time=0.9)
        gap = Line(axes.c2p(8.25, 0.12), axes.c2p(8.25, 0.68), color=C_OV, stroke_width=4)
        gl = Text("gap", font_size=24, color=C_OV).next_to(gap, RIGHT, buff=0.12)
        self.play(Create(gap), FadeIn(gl), run_time=0.8)
        self.say("train low, val high · memorizing · regularize / more data / early stop", C_OV, 26)
        self.end()
        # 5 sweet
        self.begin("sweet")
        s = DashedLine(axes.c2p(4.5, 0), axes.c2p(4.5, 1.05), color=C_RG, stroke_width=3)
        sl = Text("sweet spot", font_size=26, color=C_RG).next_to(s, UP, buff=0.1)
        self.play(Create(s), FadeIn(sl), run_time=0.9)
        self.say("pick complexity / early-stop where validation error is lowest", C_RG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch3: Regularization
class Regularize(StepScene):
    def construct(self):
        title = caption("L2 · L1 · dropout — stop the model from memorizing", 32).to_edge(UP, buff=0.35)
        # 1 L2
        self.begin("l2")
        self.play(FadeIn(title), run_time=0.7)
        f = MathTex(r"L_{\mathrm{L2}}=L_{\mathrm{data}}+\lambda\sum_j w_j^2", font_size=48).shift(UP * 1.6)
        self.play(Write(f), run_time=1.3)
        pts = VGroup(
            Text("also called weight decay: each step multiplies w by (1 − 2ηλ)", font_size=26, color=MUTED),
            Text("shrinks weights toward 0 · prefers many small weights", font_size=26, color=C_L2),
            Text("prior view: Gaussian prior on w  →  −log prior ∝ ‖w‖²", font_size=26, color=C_L2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.4)
        self.play(FadeIn(pts), run_time=1.2)
        self.say("λ bigger → stronger shrink · less able to fit noise", C_L2, 28)
        self.end()
        # 2 L1
        self.begin("l1")
        self.play(FadeOut(VGroup(f, pts)), run_time=0.5)
        f2 = MathTex(r"L_{\mathrm{L1}}=L_{\mathrm{data}}+\lambda\sum_j |w_j|", font_size=48).shift(UP * 1.8)
        self.play(Write(f2), run_time=1.2)
        # diamond vs circle constraint sketch
        circ = Circle(1.1, color=C_L2, stroke_width=3).shift(LEFT * 3.2 + DOWN * 0.3)
        diam = Polygon(*[np.array(p) for p in [[0,1.2,0],[1.2,0,0],[0,-1.2,0],[-1.2,0,0]]], color=C_L1, stroke_width=3).shift(RIGHT * 3.0 + DOWN * 0.3)
        lc = Text("L2 ball", font_size=24, color=C_L2).next_to(circ, DOWN, buff=0.2)
        ld = Text("L1 diamond", font_size=24, color=C_L1).next_to(diam, DOWN, buff=0.2)
        tip = Text("loss contours hit a corner → some wi = 0 (sparse)", font_size=26, color=C_L1).shift(DOWN * 2.4)
        self.play(Create(circ), Create(diam), FadeIn(lc), FadeIn(ld), run_time=1.4)
        self.play(FadeIn(tip), run_time=0.7)
        self.say("L1 → Laplace prior · corners encourage exact zeros", C_L1, 28)
        self.end()
        # 3 dropout
        self.begin("drop")
        self.play(FadeOut(VGroup(f2, circ, diam, lc, ld, tip)), run_time=0.5)
        nodes = VGroup(*[Circle(0.28, color=C_DO, fill_opacity=0.25, stroke_width=3) for _ in range(6)]).arrange(RIGHT, buff=0.55).shift(UP * 0.6)
        labs = VGroup(*[Text(str(i + 1), font_size=22).move_to(nodes[i]) for i in range(6)])
        self.play(FadeIn(nodes), FadeIn(labs), run_time=0.9)
        self.play(*[nodes[i].animate.set_fill(C_BAD, 0.5).set_stroke(C_BAD) for i in (1, 4)],
                  *[labs[i].animate.set_color(C_BAD) for i in (1, 4)], run_time=1.0)
        note = VGroup(
            Text("each training step: randomly zero units (p ≈ 0.5 hidden)", font_size=26, color=MUTED),
            Text("forces redundancy · at test time scale by keep-prob (or inverted dropout)", font_size=26, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 1.3)
        self.play(FadeIn(note), run_time=1.0)
        self.say("dropout ≈ training a bag of thinned networks", C_DO, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch4: Splits & leakage
class SplitsLeak(StepScene):
    def construct(self):
        title = caption("Train / val / test · k-fold · data leakage (speech)", 32).to_edge(UP, buff=0.35)
        # 1 three-way
        self.begin("three")
        self.play(FadeIn(title), run_time=0.7)
        boxes = VGroup(
            VGroup(Rectangle(width=2.8, height=1.4, stroke_color=C_GOOD, fill_color=C_GOOD, fill_opacity=0.15, stroke_width=3), Text("train\nfit weights", font_size=26, color=C_GOOD, line_spacing=1.1)),
            VGroup(Rectangle(width=2.8, height=1.4, stroke_color=C_AL, fill_color=C_AL, fill_opacity=0.15, stroke_width=3), Text("validation\npick λ, epoch, arch", font_size=24, color=C_AL, line_spacing=1.1)),
            VGroup(Rectangle(width=2.8, height=1.4, stroke_color=C_BV, fill_color=C_BV, fill_opacity=0.15, stroke_width=3), Text("test\none final report", font_size=26, color=C_BV, line_spacing=1.1)),
        )
        for b in boxes:
            b[1].move_to(b[0])
        boxes.arrange(RIGHT, buff=0.35).shift(UP * 0.5)
        self.play(FadeIn(boxes), run_time=1.2)
        self.say("never tune on test · peeking turns test into a second val set", FG, 28)
        self.end()
        # 2 k-fold
        self.begin("kfold")
        self.play(FadeOut(boxes), run_time=0.5)
        folds = VGroup()
        for k in range(5):
            row = VGroup()
            for j in range(5):
                col = C_AL if j == k else C_GOOD
                lab = "val" if j == k else "tr"
                cell = VGroup(Rectangle(width=1.5, height=0.7, stroke_color=col, fill_color=col, fill_opacity=0.2, stroke_width=2),
                              Text(lab, font_size=22, color=col))
                cell[1].move_to(cell[0]); row.add(cell)
            row.arrange(RIGHT, buff=0.08)
            folds.add(row)
        folds.arrange(DOWN, buff=0.12).shift(UP * 0.2)
        self.play(FadeIn(folds), run_time=1.3)
        self.say("5-fold: each chunk is val once · average the 5 scores", FG, 28)
        self.end()
        # 3 leakage speakers
        self.begin("speak")
        self.play(FadeOut(folds), run_time=0.5)
        bad = VGroup(
            Text("LEAK (speech): same speaker in train and test", font_size=28, color=C_BAD),
            Text("model memorizes voice → WER looks great, fails on new callers", font_size=26, color=MUTED),
            Text("fix: split by speaker_id (or session), not by utterance", font_size=26, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(UP * 0.6)
        self.play(FadeIn(bad), run_time=1.2)
        self.say("group leakage: correlated rows must stay on one side of the split", C_BAD, 26)
        self.end()
        # 4 normalize leak
        self.begin("norm")
        self.play(FadeOut(bad), run_time=0.5)
        n = VGroup(
            Text("LEAK: normalize with mean/std computed on train+test", font_size=28, color=C_BAD),
            Text("test statistics sneak into training → optimistic metrics", font_size=26, color=MUTED),
            Text("fix: fit scaler on train only; apply those numbers to val/test", font_size=26, color=C_GOOD),
            Text("also leaky: picking features with all-data correlation tests", font_size=26, color=C_DO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).shift(UP * 0.3)
        self.play(FadeIn(n), run_time=1.2)
        self.say("rule: any choice that used test labels/stats is leakage", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch5: Loss as MLE
class LossMLE(StepScene):
    def construct(self):
        title = caption("Where losses come from: maximum likelihood", 32).to_edge(UP, buff=0.35)
        # 1 idea
        self.begin("idea")
        self.play(FadeIn(title), run_time=0.7)
        idea = VGroup(
            Text("Assume a noise model for y | x, θ", font_size=30),
            Text("Likelihood = P(data | θ)   →   maximize it", font_size=30, color=C_LOSS),
            Text("≡ minimize  −log likelihood   (the loss)", font_size=30, color=C_LOSS),
        ).arrange(DOWN, buff=0.4).shift(UP * 0.4)
        self.play(FadeIn(idea), run_time=1.2)
        self.say("loss is not arbitrary — it encodes an assumption about noise", FG, 28)
        self.end()
        # 2 MSE
        self.begin("mse")
        self.play(FadeOut(idea), run_time=0.5)
        h = Text("Regression: Gaussian noise  y = fθ(x) + ε,  ε ~ N(0, σ²)", font_size=28, color=C_BV).shift(UP * 2.0)
        self.play(FadeIn(h), run_time=0.7)
        der = MathTex(r"-\log p(y\mid x)=-\log\Big(\tfrac{1}{\sqrt{2\pi}\sigma}e^{-(y-f)^2/(2\sigma^2)}\Big)", font_size=36).shift(UP * 0.7)
        self.play(Write(der), run_time=1.6)
        res = MathTex(r"= \tfrac{1}{2\sigma^2}(y-f_\theta(x))^2 + \mathrm{const}", font_size=40, color=C_GOOD).shift(DOWN * 0.6)
        self.play(Write(res), run_time=1.2)
        self.say("MSE = NLL under Gaussian noise (σ fixed)", C_GOOD, 28)
        self.end()
        # 3 CE bernoulli
        self.begin("ce")
        self.play(FadeOut(VGroup(h, der, res)), run_time=0.5)
        h2 = Text("Binary classification: y ~ Bernoulli(p),  p = σ(z)", font_size=28, color=C_SIG).shift(UP * 2.0)
        self.play(FadeIn(h2), run_time=0.7)
        lik = MathTex(r"p(y\mid p)=p^y(1-p)^{1-y}", font_size=44).shift(UP * 0.8)
        self.play(Write(lik), run_time=1.2)
        nll = MathTex(r"-\log p= -[y\log p+(1-y)\log(1-p)]", font_size=42, color=C_GOOD).shift(DOWN * 0.4)
        self.play(Write(nll), run_time=1.3)
        self.say("binary cross-entropy = Bernoulli NLL", C_GOOD, 28)
        self.end()
        # 4 categorical
        self.begin("cat")
        self.play(FadeOut(VGroup(h2, lik, nll)), run_time=0.5)
        h3 = Text("K classes: y one-hot, p = softmax(z)", font_size=28, color=C_DO).shift(UP * 1.6)
        ce = MathTex(r"L=-\sum_{k=1}^{K} y_k\log p_k", font_size=48).shift(UP * 0.3)
        note = Text("when y is one-hot this is just −log p_correct", font_size=28, color=MUTED).shift(DOWN * 0.8)
        self.play(FadeIn(h3), Write(ce), FadeIn(note), run_time=1.6)
        self.say("categorical CE = multinoulli NLL · same idea, K-way", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch6: Gradient descent
class GradDesc(StepScene):
    def construct(self):
        title = caption("Gradient descent: follow −∇ to lower the loss", 32).to_edge(UP, buff=0.35)
        # 1 rule
        self.begin("rule")
        self.play(FadeIn(title), run_time=0.7)
        f = MathTex(r"w \leftarrow w - \eta\,\nabla_w L(w)", font_size=56).shift(UP * 1.2)
        self.play(Write(f), run_time=1.4)
        bits = VGroup(
            Text("∇L = direction of steepest ascent", font_size=28, color=MUTED),
            Text("minus sign → descend · η = learning rate (step size)", font_size=28, color=C_GD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(DOWN * 0.4)
        self.play(FadeIn(bits), run_time=1.0)
        self.say("one step: evaluate gradient at current w, then nudge", FG, 28)
        self.end()
        # 2 lr
        self.begin("lr")
        self.play(FadeOut(VGroup(f, bits)), run_time=0.5)
        axes = Axes(x_range=[-0.5, 4.5, 1], y_range=[0, 5, 1], x_length=8, y_length=3.6,
                    axis_config={"color": MUTED}, tips=False).shift(DOWN * 0.2)
        curve = axes.plot(lambda x: (x - 3) ** 2, color=C_BV, stroke_width=4)
        self.play(Create(axes), Create(curve), run_time=1.0)
        # too big
        dots_big = [axes.c2p(0, 9)]  # will clip - use visible points
        # show annotations instead
        a = VGroup(
            Text("η too small → crawl", font_size=26, color=C_UN),
            Text("η too big → overshoot / diverge", font_size=26, color=C_BAD),
            Text("η just right → steady progress to the min", font_size=26, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.4).shift(UP * 0.8)
        self.play(FadeIn(a), run_time=1.0)
        self.say("tune η on val loss; schedules (decay, warmup) help deep nets", FG, 26)
        self.end()
        # 3 by hand
        self.begin("hand")
        self.play(FadeOut(VGroup(axes, curve, a)), run_time=0.5)
        h = Text("By hand:  L(w)=(w−3)²   ·   ∇L=2(w−3)   ·   η=0.25   ·   w₀=0", font_size=26, color=C_AL).shift(UP * 2.1)
        self.play(FadeIn(h), run_time=0.7)
        steps = VGroup(
            Text("step 1:  g=2(0−3)=−6   →   w=0−0.25(−6)=1.5", font_size=28),
            Text("step 2:  g=2(1.5−3)=−3  →   w=1.5−0.25(−3)=2.25", font_size=28),
            Text("step 3:  g=2(2.25−3)=−1.5 →  w=2.25−0.25(−1.5)=2.625", font_size=28),
            Text("→ closing in on w*=3", font_size=28, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.2)
        self.play(FadeIn(steps), run_time=1.4)
        self.say("write g first, then the update — don't skip the gradient", FG, 28)
        self.end()
        # 4 sgd
        self.begin("sgd")
        self.play(FadeOut(VGroup(h, steps)), run_time=0.5)
        tab = VGroup(
            Text("batch GD:  ∇ over all N examples — exact but slow / memory-heavy", font_size=26, color=C_BV),
            Text("SGD:       ∇ on 1 random example — noisy, cheap, often generalizes", font_size=26, color=C_DO),
            Text("mini-batch: ∇ on B examples (32–256) — GPU-friendly middle ground", font_size=26, color=C_GOOD),
            Text("noise in SGD can help escape sharp minima; batch size tunes it", font_size=26, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(UP * 0.3)
        self.play(FadeIn(tab), run_time=1.3)
        self.say("in practice almost everyone uses mini-batch SGD / Adam", C_GOOD, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch7: Logistic regression
class LogReg(StepScene):
    def construct(self):
        title = caption("Logistic regression: sigmoid, log-loss, gradient (p−y)x", 30).to_edge(UP, buff=0.35)
        # 1 sigmoid
        self.begin("sig")
        self.play(FadeIn(title), run_time=0.7)
        s = MathTex(r"p=\sigma(z)=\frac{1}{1+e^{-z}}\in(0,1)", font_size=52).shift(UP * 1.3)
        self.play(Write(s), run_time=1.4)
        props = VGroup(
            Text("z = w·x  (plus bias) · score → probability", font_size=28, color=MUTED),
            Text("σ(0)=0.5 · σ→1 as z→+∞ · σ→0 as z→−∞", font_size=28, color=C_SIG),
            Text("σ'(z)=σ(z)(1−σ(z))  — handy for backprop", font_size=28, color=C_SIG),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.5)
        self.play(FadeIn(props), run_time=1.1)
        self.say("one linear score, squashed to a probability", FG, 28)
        self.end()
        # 2 loss
        self.begin("loss")
        self.play(FadeOut(VGroup(s, props)), run_time=0.5)
        loss = MathTex(r"L=-[y\log p+(1-y)\log(1-p)]", font_size=48).shift(UP * 1.4)
        self.play(Write(loss), run_time=1.3)
        note = VGroup(
            Text("y=1 → L=−log p   (punish low confidence on positives)", font_size=26, color=C_GOOD),
            Text("y=0 → L=−log(1−p)", font_size=26, color=C_BAD),
            Text("same Bernoulli NLL from the last chapter", font_size=26, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.3)
        self.play(FadeIn(note), run_time=1.0)
        self.say("perfect prediction → L→0 · confident & wrong → L→∞", FG, 28)
        self.end()
        # 3 grad derive
        self.begin("grad")
        self.play(FadeOut(VGroup(loss, note)), run_time=0.5)
        d = VGroup(
            MathTex(r"\frac{\partial L}{\partial z}=p-y", font_size=48),
            Text("chain: ∂L/∂p · ∂p/∂z simplifies to (p − y)", font_size=26, color=MUTED),
            MathTex(r"\frac{\partial L}{\partial w}=(p-y)\,x", font_size=48, color=C_AL),
            Text("because ∂z/∂w = x ·  (vector form: (p−y)x)", font_size=26, color=MUTED),
        ).arrange(DOWN, buff=0.35).shift(UP * 0.2)
        self.play(FadeIn(d), run_time=1.5)
        self.say("the famous (p−y)x gradient — remember this for interviews", C_AL, 28)
        self.end()
        # 4 by hand
        self.begin("hand")
        self.play(FadeOut(d), run_time=0.5)
        setup = Text("x=2,  w=0.3,  y=1,  η=0.5", font_size=28, color=C_AL).shift(UP * 2.1)
        self.play(FadeIn(setup), run_time=0.6)
        steps = VGroup(
            Text("z = w·x = 0.6", font_size=28),
            Text("p = σ(0.6) = 1/(1+e^{−0.6}) ≈ 0.646", font_size=28),
            Text("L = −log(0.646) ≈ 0.437", font_size=28),
            Text("∇w = (p−y)x = (0.646−1)·2 ≈ −0.709", font_size=28, color=C_AL),
            Text("w ← 0.3 − 0.5(−0.709) ≈ 0.654", font_size=28, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(DOWN * 0.15)
        self.play(FadeIn(steps), run_time=1.5)
        self.say("p < y → gradient negative → w increases → p rises next time", C_GOOD, 26)
        self.end(2.2)

# ---------------------------------------------------------------- Ch8: Backprop
class Backprop(StepScene):
    def construct(self):
        title = caption("Backprop by hand on a tiny net", 32).to_edge(UP, buff=0.35)
        # 1 net
        self.begin("net")
        self.play(FadeIn(title), run_time=0.7)
        net = MathTex(r"x \xrightarrow{w_1} z=w_1 x \xrightarrow{\sigma} h=\sigma(z) \xrightarrow{w_2} \hat y=w_2 h", font_size=36).shift(UP * 1.5)
        self.play(Write(net), run_time=1.6)
        loss = MathTex(r"L=\tfrac12(\hat y-y)^2", font_size=44, color=C_LOSS).shift(UP * 0.3)
        vals = Text("numbers: x=1, w₁=1, w₂=2, y=1   (no bias, for clarity)", font_size=26, color=MUTED).shift(DOWN * 0.6)
        self.play(Write(loss), FadeIn(vals), run_time=1.2)
        self.say("1 input → 1 sigmoid hidden → linear output · squared loss", FG, 28)
        self.end()
        # 2 forward
        self.begin("fwd")
        self.play(FadeOut(VGroup(net, loss, vals)), run_time=0.5)
        fwd = VGroup(
            Text("Forward pass", font_size=30, color=C_GOOD),
            Text("z = w₁·x = 1·1 = 1", font_size=28),
            Text("h = σ(1) = 1/(1+e^{−1}) ≈ 0.731", font_size=28),
            Text("ŷ = w₂·h = 2·0.731 ≈ 1.462", font_size=28),
            Text("L = ½(1.462−1)² ≈ 0.107", font_size=28, color=C_LOSS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(UP * 0.3)
        self.play(FadeIn(fwd), run_time=1.4)
        self.say("write every intermediate — you'll need them going backward", FG, 28)
        self.end()
        # 3 backward
        self.begin("bwd")
        self.play(FadeOut(fwd), run_time=0.5)
        bwd = VGroup(
            Text("Backward (chain rule)", font_size=30, color=C_BP),
            Text("∂L/∂ŷ = ŷ−y ≈ 0.462", font_size=28),
            Text("∂L/∂w₂ = (∂L/∂ŷ)·h ≈ 0.462·0.731 ≈ 0.338", font_size=28, color=C_AL),
            Text("∂L/∂h = (∂L/∂ŷ)·w₂ ≈ 0.462·2 ≈ 0.924", font_size=28),
            Text("∂h/∂z = h(1−h) ≈ 0.731·0.269 ≈ 0.197", font_size=28),
            Text("∂L/∂w₁ = (∂L/∂h)·(∂h/∂z)·x ≈ 0.924·0.197·1 ≈ 0.182", font_size=28, color=C_AL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).shift(UP * 0.15)
        self.play(FadeIn(bwd), run_time=1.6)
        self.say("each local derivative multiplies along the path to that weight", C_BP, 26)
        self.end()
        # 4 update + faded
        self.begin("upd")
        self.play(FadeOut(bwd), run_time=0.5)
        upd = VGroup(
            Text("One update with η=0.5", font_size=30, color=C_GOOD),
            Text("w₂ ← 2 − 0.5·0.338 ≈ 1.831", font_size=28),
            Text("w₁ ← 1 − 0.5·0.182 ≈ 0.909", font_size=28),
            Text("both fell: ŷ was too high, so we reduce the path weights", font_size=26, color=MUTED),
            Text("Faded: if y=0 instead, ∂L/∂ŷ=ŷ−0>0 still… wait ŷ≈1.46,", font_size=24, color=C_DO),
            Text("sign of (ŷ−y) flips the whole chain — check it on paper", font_size=24, color=C_DO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(UP * 0.2)
        self.play(FadeIn(upd), run_time=1.5)
        self.say("interview move: forward numbers → chain → one update", C_GOOD, 28)
        self.end(2.2)
