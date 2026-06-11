# Trust-Gated Forecasting

> When can you trust a zero-shot forecast enough to act on it automatically? A conformal trust-gating layer on AWS.

---

### White Paper: Turning Zero-Shot Time-Series Forecasting into Trustworthy Industrial Automation

#### — A Forecast-Confidence-Based Automation Gating Approach

*Audience: AWS Solutions Architects and Account Managers*
*(Note: "forecast-confidence gating" is a descriptive name for the methodology, not a product name. Citations appear inline and in Appendix A.)*

---

## Executive Summary

Time-series forecasting underpins industrial automation across demand/inventory, equipment failure, energy load, finance, and healthcare, and the AI market built on it is growing fast — for example, the AI-driven predictive maintenance market is projected to grow from $2.61B in 2026 to $19.27B in 2032, a 39.5% CAGR (*MarketsandMarkets, 2026*). The technology behind this growth is the **zero-shot Time-Series Foundation Model (TSFM)**. The old structure of training a separate model per series has shifted to a single model that forecasts all series with no training, and as of 2026 it has reached production-viable maturity, with real enterprises such as Deutsche Bahn adopting it (*AWS, 2025*).

Yet in the field, humans still review forecasts one by one. The remaining bottleneck is not accuracy but the trust judgment: "which forecasts can be acted on without a human?" A model's forecasts may be quite accurate on average, but for an individual series, under distribution shift, or in the face of asymmetric costs, the model cannot guarantee "can I trust this single case?" (the model-level gap). Existing harness engineering and AWS services provide the scaffolding for safety, auditing, and review, but leave the decision logic of "when to trust" empty (the tooling-level gap).

This white paper proposes a **thin judgment layer** (a trust-decision layer placed on top of the model) to fill these two gaps. It scores each forecast's reliability from observed outcomes (conformal prediction), reflects the fact that being wrong in one direction can cost more than the other (cost asymmetry), automatically executes the trustworthy cases while escalating the uncertain ones to humans, and improves itself over time. In a real-data validation (retail inventory) it **cut inventory cost by 18.6%**, and the larger the loss from being wrong in one direction, the greater the benefit.

> **Document structure**
> - **Ch.1** Target industrial solutions and how they are changing (zero-shot TSFM adoption)
> - **Ch.2** The development and current state of zero-shot forecasting models
> - **Ch.3** The blueprint and its remaining limits (model level)
> - **Ch.4** Reliability features and blind spots of existing harnesses and AWS services (tooling level)
> - **Ch.5** The proposed methodology to fill the gaps, and real results

---

## Chapter 1. Target Industrial Solutions and How They Are Changing

### 1.1 The industrial solutions this material covers

This white paper targets industrial solutions in which time-series forecasting triggers autonomous or semi-autonomous actions. The solutions below differ by domain but share the same structure — "estimate a future time series, then act on the result" — so they can be addressed with a single approach.

| Domain | Forecast target | Triggered action | Field pain point |
|---|---|---|---|
| Demand/Inventory (Retail/CPG/Logistics) | Future demand | Ordering, replenishment, safety stock | Daily store-level SKUs are intermittent and erratic, making forecasting hard (*ResearchGate*) |
| Predictive maintenance (Manufacturing/Industry) | Equipment state, failure | Maintenance timing, downtime avoidance | Missing a failure brings large unplanned-downtime and safety losses |
| Energy load (Power/Utilities) | Load, generation | Dispatch, peak shaving | Needs real-time response to variability; maintaining a model per series is burdensome |
| Finance / Healthcare / Traffic | Risk, vital signs, flow | Positioning, monitoring, routing | High-stakes decisions where being wrong is costly (*Adler et al., 2025*) |

> (Market-size figures for predictive maintenance etc. are in the Executive Summary and Appendix A — this table focuses on pain points.)

Because the "estimate a future series → act" structure is shared regardless of domain, the discussion here generalizes to any solution with this structure, not just one domain.

### 1.2 How they were originally designed (the traditional structure)

Demand/inventory forecasting was long designed by building and maintaining a separate model per SKU or series. Early on, statistical models such as ARIMA/SARIMA or Holt-Winters were fit per series (*Fattah et al., 2018, SAGE*), and later machine-learning methods like XGBoost/LSTM were added to capture nonlinear patterns (*Nature Sci. Reports, 2025*). Either way, the structure is the same: as the number of forecast targets grows, so does the number of models.

Predictive maintenance solutions were likewise built per equipment and per sensor, with condition-monitoring and anomaly-detection models configured individually (*A Survey of Predictive Maintenance; Anomaly Detection for PdM in Industry 4.0*). Even recent deep-learning approaches (e.g., hybrids combining an LSTM autoencoder with an MLP) or digital-twin-based methods sit within the same frame, since they still require training per asset (*MDPI Machines, 2026; arXiv:2509.24443*). As a result many sites remain reactive rather than predictive, failing to effectively prevent unplanned-downtime losses estimated at roughly $260,000 per hour (widely cited industry estimate).

AWS itself points out the limits of this traditional structure: conventional ML requires extensive per-dataset tuning and customization, making development long and resource-intensive (*AWS ML Blog, 2025.03*). In short, three bottlenecks remain: first, the scalability problem of building a model for every series/asset; second, cold start for new targets with little history; and third, the dependence on ML expertise for proper forecasting.

### 1.3 The change — what zero-shot TSFM adoption shifted

The biggest change is the compression of the workflow. Where each new task once required data preparation, model selection, and training/tuning/validation from scratch, a single inference call on a pretrained model now replaces that process (*Shuai Guo, "Five Questions About Chronos-2", TDS, 2026.05*). This is because the zero-shot approach produces forecasts immediately, with no training on the target data (*AWS ML Blog, 2025*). As a result, cold start for new targets is eased, dependence on ML expertise drops, and instead of a single point estimate you get a probabilistic output (quantiles) that tells you "with what probability the value falls in which range."

Notably, many organizations are already using zero-shot TSFMs without recognizing it as a special transition. The original Chronos became the most-downloaded model on Hugging Face in 2024, and Deutsche Bahn has used Chronos to improve its forecasting, offering it through Amazon Bedrock Marketplace (*AWS, 2025*). Applications are spreading quickly to power-grid imbalance (ML6), electricity prices (Chronos-Bolt), disease incidence (HFMD) forecasting, and more (*ml6.eu, 2026; emergentmind, 2025; Frontiers Public Health, 2025*).

In a domestic context the same trend is visible: after I proposed Amazon Chronos-2 to research groups at major heavy-industry and electronics companies and a government research institute, they expressed surprise at the zero-shot performance achieved without any training and are pursuing follow-up research. In short, zero-shot TSFM is no longer an experiment confined to a few leading firms but a trend being validated simultaneously across many industrial sites.

In summary, the industry is moving from "build a model per series" to "one model forecasts all series with no training." In areas such as energy, some even say that building a separate model per series is no longer the norm (*howtostoreelectricity, 2026*), and as of 2026 this approach is regarded as already production-viable (*Spheron, 2026*).

---

## Chapter 2. The Development and Current State of Zero-Shot Forecasting Models

### 2.1 The paradigm shift — why it works

The reason a time-series model works even on data it never trained on is that it does not memorize specific data; it learns "shapes" — cycles, trends, level shifts, and sudden spikes. These shapes recur across domains and are far fewer in kind than the possible numeric values, so they are recognized even in series the model has never seen (*TDS, 2026*). The insight from the Chronos team is that LLMs and time-series forecasting are the same kind of problem — "decode a sequential pattern to predict what comes next" — so a time series can be treated like a language (*AWS ML Blog, 2025; Ansari et al., TMLR 2024*). As a result, zero-shot TSFM performance is now the baseline to beat (*TDS, 2026; see the benchmarks in 2.4*).

### 2.2 Model landscape (development history)

| Model (developer) | Appeared | Significance (one line) |
|---|---|---|
| TimeGPT (Nixtla) | 2023 | First commercial time-series foundation model |
| TimesFM (Google) | 2024 (through 2.5) | General forecasting model trained on real + synthetic data |
| Moirai (Salesforce) | 2024~2025 | A family that broadly handles diverse distributions and multivariate inputs |
| Chronos / Chronos-2 (Amazon) | 2024 → 2025.10 | Treats time series like language; v2 extends to multivariate |
| Lag-Llama | 2023 | Early open model applying the LLaMA architecture to time series |
| Toto (Datadog) | 2025 | Specialized for monitoring (observability) data; top of recent benchmarks |
| TiRex (NX-AI) | 2025 | Strong at long-horizon forecasting without autoregression |
| YingLong | 2025 | Improves long-horizon accuracy by looking further ahead, then revisiting |
| IBM Granite TTM | 2024~ | Ultra-light model, well suited to lightweight deployment |

> (Papers and sources for each model are in Appendix A.)

### 2.3 A representative model of the current state — Chronos-2

Amazon Chronos-2, released in October 2025, is a good example of where TSFMs stand today. It is a relatively small, encoder-only model of 120M parameters (about 478MB), yet it runs on both CPU and GPU and is released under Apache-2.0, so it can be adopted without heavy infrastructure.

Architecturally, it embeds the time series into continuous patches and alternates attention along the time direction and across series (group attention). As a result a single model supports four modes — univariate, multivariate, covariate-informed, and cross-learning. In other words, you can handle different forecasting tasks by changing only the input configuration rather than using a different model per task. Its output head produces 21 quantiles at once, so it is probabilistic by default, and it supports a context of up to 8,192 steps and forecasts of up to 1,024 steps.

In terms of performance, in one case the univariate forecast error (WAPE) of 8.4% dropped to **4.2%** once future covariates were added, and a cold-start target with only three days of history improved from 22.2% to 16.7% through cross-learning with other series. Interestingly, much of the training data is synthetic. On the operations side, as of December 2025 it can be deployed on Amazon SageMaker for real-time (GPU/CPU), serverless, and batch inference (*arXiv:2510.15821; TDS, 2026; Hugging Face amazon/chronos-2, 2025.12*).

### 2.4 Capability frontier and benchmarks

Beyond univariate forecasting, recent models support multivariate (forecasting several series together), covariates (using auxiliary information known in advance), and cross-learning (borrowing patterns from related series). They also ease cold start for new targets and provide probabilistic quantiles by default, while keeping efficiencies such as small model size, CPU inference, and long context. In a large-scale transportation benchmark, they outperformed classical statistical methods and purpose-built deep learning, with the advantage especially clear over long horizons (*arXiv:2602.24238*).

On the leaderboard side, Datadog's Toto 2.0 tops BOOM, GIFT-Eval, and the new data-leakage-free TIME benchmark while improving speed and efficiency (*aihorizonforecast, 2026*), and Moirai, TimesFM, and IBM Granite TTM trade the top spot across benchmarks (*howtostoreelectricity, 2026*). That said, aggregate leaderboards should be read with care: the best model on average may not be best for your specific task, and few-shot rankings differ from full-data rankings (*HF papers 2606.04525*). Concerns about data leakage and evaluation bias in the benchmarks themselves have also been raised (*arXiv:2510.13654; Google, "Rethinking Context-Enriched..."*).

### 2.5 Spreading across domains

Adoption is not confined to one industry. Research is underway simultaneously to handle load and grid operations in energy, equipment anomaly detection in manufacturing/industry, disease incidence and vital signs in healthcare, and trading and risk in finance — all with the same zero-shot time-series approach. The fact that one approach works across data of such different character shows that TSFM is becoming a general-purpose technology rather than a niche technique. (Domain-by-domain research sources are in Appendix A.)

---

## Chapter 3. The Blueprint and Its Remaining Limits (Model Level)

### 3.1 The blueprint — fully autonomous time-series automation

The ideal picture is clear. The model forecasts all series, the system automatically performs actions such as ordering, maintenance, and dispatch, and humans handle only the exceptions. Since the model can already forecast the entire catalog in seconds, this blueprint is technically within reach.

### 3.2 The reality — the blueprint is stuck

Reality is far from the blueprint. The model forecasts all series, but humans review only a fraction of them, and on a weekly cycle at that. Worse, research shows that human manual adjustments actually degrade forecasts in 30~40% of cases (*Fildes et al., 2009; Franses & Legerstee, 2009*). Many sites, predictive maintenance included, remain reactive rather than predictive. In the end, a model capable of handling all series sits largely idle.

### 3.3 Why the model alone cannot reach the blueprint (the model-level limits)

The first issue, and the key one, is the subtlety of calibration. A recent systematic study reports that TSFMs are consistently better calibrated than baselines such as ARIMA and N-BEATS, with no systematic overconfidence in short-horizon forecasts (*Adler et al., "Are Time Series Foundation Models Well-Calibrated?", arXiv:2510.16060*). But the same study adds important caveats: the farther out you forecast, the worse both accuracy and calibration get; autoregressive long-horizon forecasting is consistently overconfident; and because the evaluation was limited to the univariate, aggregate level, the impact of distribution shift (non-stationarity, where the pattern itself changes over time) was left for future work.

The implication is clear. Even when average calibration is good, things can go badly off for an individual series, a specific catalog, or under distribution shift. The case where "an 80% prediction interval covered only 70.6% of outcomes across 15,772 SKUs" in retail inventory is exactly this kind of deviation. In other words, "the overall average is fine" and "can I trust this single series?" are entirely different questions.

The second issue is that the model does not optimize for asymmetric costs. When wrong, the loss often skews heavily to one side — stockout vs. overstock, missed failure vs. false alarm, blackout vs. over-generation. Even the Chronos-2 authors note that if you need behavior the zero-shot objective does not optimize for (e.g., when under-forecasting costs 10x as much as over-forecasting), you should fine-tune with your own loss function (*TDS, 2026*).

The third issue is domain dependence. Zero-shot performance is strongly tied to the composition of the pretraining data (*arXiv:2510.00742*), so in areas unlike the pretraining data — such as clinical vital signs — strong benchmark results do not translate into real accuracy (*Gu et al., ML4H 2025*).

In short, the model has solved speed, scale, and even average calibration, but it cannot answer on its own the question "can I act on this single case, without a human, weighing the cost risk?" And this gap does not close by making the model bigger.

---

## Chapter 4. Reliability Features and Blind Spots of Existing Harnesses and AWS Services (Tooling Level)

If Chapter 3 covered the model's limits, Chapter 4 looks at how far the **tooling and platform** that operate the model guarantee reliability.

### 4.1 Existing harness engineering

Here, a "harness" means the scaffolding placed around a model to make it actually work — prompts, tool connections, output validation, guardrails, and so on. On the reliability side, a harness provides devices such as output-schema validation, retries and fallbacks, and rule-based guardrails (block lists or fixed thresholds).

But the limits are clear. These devices are mostly hand-written fixed rules, so they stay put even as conditions change; they cannot learn from observed outcomes when the model is wrong; and the confidence they rely on may itself be biased. Above all, they cannot escape the "all-automatic or all-manual" dichotomy, so they fail to cross over into production automation.

### 4.2 AWS services that contribute to stability and reliability

| Service | Reliability features | Blind spot |
|---|---|---|
| Bedrock Guardrails | Content filters, PII masking, Contextual Grounding (confidence scores to block hallucinations), Automated Reasoning (formal-logic verification, up to 99% verification accuracy) | Checks whether text matches facts/rules, but not whether a forecast number is safe to act on. Thresholds are fixed and manual |
| Bedrock AgentCore | Identity (permissions, least privilege, token vault), Observability (tracing, audit), human-in-the-loop (HITL) hooks, multi-tenant isolation | Provides who/what/safe/audit, but the "when to trust" decision logic is absent. HITL provides only the hook |
| SageMaker Model Monitor | Detects and alerts on data/concept/bias drift; integrates with A2I | Model-level, after-the-fact alerts. Does not decide whether an individual case may be automated |
| Amazon A2I | "Below threshold → human review" workflow; integrates with Model Monitor and Clarify | The threshold is based on the model's (possibly biased) self-reported confidence. No cost asymmetry or self-calibration |

> (Sources for each service's features are the official AWS docs for Bedrock Guardrails, AgentCore, SageMaker Model Monitor, and A2I, plus AWS blogs (2025); see Appendix A.)

### 4.3 Why not just use an LLM as the decision layer?

A natural objection: why not hand the empty decision logic to an LLM? In fact, using LLMs as a judgment layer is an active trend — "LLM-as-judge" to assess output quality, agentic orchestration where an LLM itself decides whether to act or escalate, verification via reasoning models — and these have clear strengths: contextual understanding, natural-language explanations, integration of heterogeneous signals, and flexible workflow control.

However, for the purpose of judging the "action trust" of a numeric time-series forecast, there are structural limits. Above all, an LLM's self-reported confidence does not match its actual accuracy well. LLMs tend to be overconfident, systematically overstating their true correctness (*arXiv:2502.11028; 2405.02917*), and LLM-as-judge likewise shows an "overconfidence phenomenon" where predicted confidence far exceeds actual accuracy (*arXiv:2508.06225*). A judge's imperfections, noise, and bias can invalidate statistical guarantees themselves (*arXiv:2601.20913*). On top of this come the cost and latency of having an LLM judge tens of thousands to millions of series per day case by case, the failure to reflect asymmetric costs such as stockout vs. overstock, and the absence of self-calibration from observed feedback.

In the end, LLM-based decision-making is not a replacement for this gap but a complement. The "quantification" of trust is best handled by a statistically grounded method, while LLMs are better suited to generating human-readable explanations of that judgment and orchestrating the workflow. Chapter 5 makes this division of roles concrete.

### 4.4 The common blind spot — the empty "decision layer"

The trust signals these services provide share four common gaps: they look only at text or at the whole-model level rather than judging the action trust of an individual case; their thresholds are static, fixed values; they do not consider cost asymmetry; and they do not self-calibrate from observed outcomes.

This is less a flaw in AWS than a design choice. AWS standardizes the infrastructure and safety layers but leaves the decision logic — which drives customer differentiation — to the solution layer. In one CTO survey, about 67% of US enterprises said they build their own agent/decision layers rather than use commercial platforms (*Codiste CTO Survey, 2026*); that is precisely the spot Chapter 5 sets out to fill.

---

## Chapter 5. The Proposed Methodology and Real Results

We propose a thin judgment layer that simultaneously fills the two gaps exposed in Chapters 3 and 4 — the model's lack of per-case, cost-aware, self-calibrating trust (Ch.3), and the platform's missing decision logic (Ch.4).

### 5.1 The idea in one line

The core proposal is to place, on top of the model, a "thin layer that judges, for each forecast, whether it is trustworthy." The model itself is not retrained. High-trust forecasts are executed automatically and low-trust ones are handed to humans, but the boundary is set not by hand-written rules but by data, reflecting cost asymmetry, and improving itself over time.

### 5.2 The rationale — why conformal prediction

Conformal Prediction (CP) is a post-hoc framework that guarantees coverage (the fraction of cases in which the prediction interval contains the true value) with no distributional assumptions (*"A Gentle Introduction to Conformal Time Series Forecasting", arXiv:2511.13608*). Unlike the LLM-based judgment of 4.3, CP's key strength is that it sits on top of any model and provides a statistical guarantee. We chose CP for three reasons.

First, CP suits high-stakes decisions. It has become a standard uncertainty-quantification technique in medicine and finance (*arXiv:2503.11709, "Conformal Prediction and Human Decision Making"*), and decision-theoretically, prediction sets have been proven optimal for risk-averse decision-makers who want to manage their Value-at-Risk (*OpenReview, "Decision Theoretic Foundations for Conformal Prediction"*).

Second, CP is synergistic with TSFMs. TSFMs provide more reliable conformal intervals even with little data, and because more data can be used for calibration, the calibration process is more stable (*arXiv:2507.08858*).

Third, there is room to handle non-stationarity. Standard CP assumes exchangeability (that swapping the order of the data leaves the distribution unchanged) (*arXiv:2511.13608*), but extensions such as Temporal Conformal Prediction for non-stationary series are active (*arXiv:2507.05470*). This methodology's distribution-shift detection and feedback calibration address exactly this limit at the operational level.

### 5.3 Methodology components

This judgment layer works in four broad stages from receiving a forecast to taking an action. The key symbols are the per-series trust score τ (tau) and the cost-minimizing order quantile q\*.

**(1) Measuring trust — conformal calibration.** For each series, it computes how often the model's recent prediction intervals contained the true value (the empirical coverage). For example, if an interval the model labeled "80%" contained only 60% of recent observations, the model is overconfident for that series and should be treated more conservatively. Conformal prediction quantifies this coverage gap, answering "how far can the model's confidence be trusted for this series?" from data — without touching the model's internals.

**(2) Combining into a trust score τ.** Calibration alone is not enough, so three signals are combined into a single score: first, the calibration just described (how well coverage matches the stated value); second, the forecast's own uncertainty (how wide and variable the interval is); and third, distribution-shift detection. Shift is caught by tracking whether recent forecast errors pile up in one direction — a signal that the pattern itself is changing, such as a demand-regime shift or equipment degradation. Combining these three signals yields a per-series trust score τ between 0 and 1.

**(3) Cost-optimal decision and action rules.** How aggressively to act is set by the cost structure. Per newsvendor theory (a classic inventory theory that weighs shortage vs. overage cost to find the optimal order quantity), the cost-minimizing order quantile is q\* = Cu/(Cu+Co), where Cu is the shortage (stockout) cost and Co is the overage cost. If the trust score is at or above the threshold τ\*, the system executes automatically at the cost-optimal quantile q\*; if trust is low, it raises to a more conservative quantile (e.g., P90) to reduce stockout risk; and if trust is very low (e.g., τ<0.3), it halts automatic execution and hands the case to a human with an explanation of what is wrong. The key point is that the decision formula itself does not change — only the quantile fed into it changes, and the trust score modulates that. With a single knob τ\*, an operator sets "how much confidence to require before automatic execution," controlling the cost-vs-safety balance (the Pareto frontier) with one value.

**(4) Guard layer and feedback loop.** Independent of the trust score, the guard layer always validates physical and business constraints such as budget, warehouse capacity, and minimum order quantity, escalating with an explanation when an unresolvable conflict arises. Finally, when actual values arrive at the end of a forecast horizon, the conformal calibrator and shift detector are updated. As a result, series that have forecasted well gain trust and become increasingly automated, while series that often err lose trust and are handled more conservatively or escalated. Cold-start series with little history start on the default (q\*), and after about 10 days of accumulated observations the trust estimate stabilizes and per-series differentiation begins.

### 5.4 System and AWS reference architecture

The overall flow starts with data and ends with action. Time-series data enters the forecasting model and a forecast is produced; the proposed layer evaluates that forecast's reliability; high-trust forecasts are executed automatically and low-trust ones are handed to humans. By layer:

- Data: store time-series history in Amazon S3 or a time-series store.
- Forecast: Amazon Chronos / SageMaker generates prediction intervals zero-shot.
- Trust judgment (the proposed layer): decides automate-vs-escalate using a conformal-based trust score and cost asymmetry (q\*), tunes conservativeness with the τ\* knob, and self-calibrates from observed outcomes.
- Execute or review: high-trust cases run automatically via action APIs (ordering, maintenance, dispatch); low-trust cases go to human review via Amazon A2I or AgentCore HITL hooks.
- Governance foundation (the Ch.4 services): AgentCore (identity, observability, isolation), Bedrock Guardrails (safety), SageMaker Model Monitor (drift), and CloudTrail (audit) support the whole process.

In this way the proposed layer does not compete with AWS services but layers on top of them, acting as the "brain" that ties the Chapter 4 pieces into real production automation. (For presentation materials, visualizing this layered structure as a formal diagram is recommended.)

### 5.5 Real results (retail-inventory instance, M5 dataset)

Implementing this methodology for retail inventory and validating it on Walmart public data (M5), the layer **cut inventory cost by 18.6%** versus median-based ordering and consistently outperformed single-signal heuristics even under matched spend (the same budget). Detailed results:

- Data: 15,772 active SKUs out of Walmart's 30,490, over 1,941 days (2011~2016), using actual sale prices.
- Per-category improvement (τ\*=0.7): Foods +20.1%, Household +17.3%, Hobbies +12.8%.
- Cost-asymmetry sensitivity: electronics (Cu/Co=2) +2pp, groceries (3.6) +7pp, pharma (9) +10pp — the larger the loss from being wrong in one direction versus the other, the greater the benefit.
- Robustness to distribution shift: under a synthetic demand shock (2x demand on 15% of series), shift detection saved an additional 0.7%.
- Practitioner assessment (three reviewers with 8~15 years' experience, each managing 2,000~5,000 SKUs): "If it just flags the uncertain 10~15%, we can put it into production," "Reviews that took five minutes per SKU now take seconds."

That said, these figures come from a single-domain validation on public data, so real adoption requires a revalidation step on the customer's operational data.

### 5.6 Domain expansion and roadmap

This approach is not limited to retail inventory. It extends by the same principle to any time-series solution where the loss from being wrong skews to one side — predictive maintenance (missed failure vs. false alarm), energy load (blackout vs. over-generation), and beyond.

Adoption is safest done in stages:
1. Apply it to a single workflow first and measure ROI.
2. Accumulate about 10 days of observations to stabilize calibration.
3. Tune the τ\* knob to gradually widen the share of automation.
4. Roll the validated pattern out to adjacent domains.

From a business standpoint, this method converts automation that was stuck in pilots into real production consumption and lets humans focus only on exceptions. In a market where models are increasingly commoditized, the "trust layer on top of the model" becomes a point of differentiation that competitors cannot easily follow.

---

## Conclusion

Time-series forecasting models have solved speed, scale, and even average calibration, but two gaps — trust at the level of the individual series, distribution shift, and asymmetric cost (Ch.3), and the platform's decision logic (Ch.4) — block the blueprint of fully autonomous automation. The thin judgment layer proposed here (Ch.5) does not retrain the model; it calibrates trust with conformal prediction, sets the automate-vs-escalate boundary by reflecting cost asymmetry, and improves itself from observed outcomes. And because it layers on top of AWS services rather than competing with them, it ties scattered governance pieces into real production automation.

In practice, rather than a sweeping all-at-once rollout, we recommend starting with a PoC on a single workflow where cost asymmetry is clear. Put trust gating on top of the AWS stack, verify with ROI that "only the uncertain few go to humans, the rest run automatically" actually works, and then widen via τ\* tuning and adjacent-domain expansion — the safest and fastest path. In a market where models become commodities, the contest is ultimately decided by this trust layer placed on top of the model.

---

## Appendix A. References

### A.1 Academic / primary sources

**Traditional forecasting / predictive maintenance**
- Fattah, J. et al. (2018). Forecasting of demand using ARIMA model. *SAGE*.
- Fildes, R. et al. (2009); Franses, P. H. & Legerstee, R. (2009). Judgmental manual adjustments degrade forecast accuracy in a substantial share of cases (~30~40%).
- Comparison of statistical and ML methods for daily SKU demand forecasting (ResearchGate).
- Adaptive demand forecasting framework. *Nature Scientific Reports* (2025), s41598-025-23352-w.
- A Survey of Predictive Maintenance Systems (ResearchGate); Anomaly Detection for PdM in Industry 4.0 (ResearchGate).
- Hybrid Deep Learning for PdM. *MDPI Machines* (2026), 14(2):191; A Systematic Review of Digital Twin-Driven PdM (arXiv:2509.24443).

**Time-series foundation models (TSFM)**
- Ansari, A. F. et al. (2024). Chronos: Learning the Language of Time Series. *TMLR*.
- Chronos-2: From Univariate to Universal Forecasting (arXiv:2510.15821, 2025).
- Das, A. et al. (2024). TimesFM: A decoder-only foundation model for time-series forecasting. *ICML* (arXiv:2310.10688).
- Woo, G. et al. (2024). Moirai. *ICML*; Rasul, K. et al. Lag-Llama (arXiv:2310.08278).
- Cohen, B. et al. (2025). Toto (arXiv:2505.14766); Auer, A. et al. TiRex (arXiv:2505.23719); Wang, X. et al. YingLong (arXiv:2506.11029).
- Garza, A. et al. TimeGPT-1 (arXiv:2310.03589); Aksu, T. et al. GIFT-Eval (arXiv:2410.10393).
- TSFMs as Strong Baselines in Transportation Forecasting (arXiv:2602.24238).
- Benchmarking Challenges and Requirements (arXiv:2510.13654); Rethinking Context-Enriched Time-Series Forecasting Evaluation (Google Research); leaderboard caution: few-shot ≠ full-data (HF papers 2606.04525).
- TSFM for HFMD forecasting. *Frontiers in Public Health* (2025).

**Calibration · conformal prediction · limits**
- Adler, C. et al. (2025). Are Time Series Foundation Models Well-Calibrated? (arXiv:2510.16060).
- How Foundational are Foundation Models for TS Forecasting? (arXiv:2510.00742).
- A Gentle Introduction to Conformal Time Series Forecasting (arXiv:2511.13608).
- Conformal Prediction and Human Decision Making (arXiv:2503.11709).
- Decision Theoretic Foundations for Conformal Prediction (OpenReview, Ukjl86EsIk).
- Foundation models for time series forecasting — conformal synergy (arXiv:2507.08858).
- Temporal Conformal Prediction / adaptive risk forecasting (arXiv:2507.05470).
- Probabilistic TSFM with Uncertainty Decomposition (arXiv:2601.10591).
- Gu, X. et al. (2025). Vital sign forecasting in healthcare. *ML4H*.

**Limits of LLM decision-making / calibration**
- Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution (arXiv:2508.06225).
- Robust Statistical Evaluation of LLMs with Imperfect Judges (arXiv:2601.20913).
- Overconfidence, Calibration, and Distractor Effects in LLMs (arXiv:2502.11028).
- Verbalized Uncertainty Evaluation in LLMs/VLMs (arXiv:2405.02917).

**AWS official documentation**
- AWS ML Blog (2025.03). Time series forecasting with LLM-based foundation models and scalable AIOps on AWS.
- AWS. Amazon Bedrock AgentCore / Bedrock Guardrails / Automated Reasoning checks official docs.
- Hugging Face. amazon/chronos-2 (SageMaker deployment guide, 2025.12).

### A.2 Secondary sources (market research, technical blogs, industry cases)

*The following are not primary academic sources but market research, blogs, and industry cases, used for trend and context.*

- MarketsandMarkets (2026). Predictive Maintenance Market / AI-Driven Predictive Maintenance Market. (Single source for this paper's market figures.)
- Codiste CTO Survey (2026). ~67% of US enterprises build their own agent/decision layers.
- HKU/AWS (2025). How Deutsche Bahn redefines forecasting using Chronos (Bedrock Marketplace). *(Cited as "AWS, 2025" in the body.)*
- Shuai Guo (2026). Five Questions About Chronos-2. *Towards Data Science (TDS)*.
- aihorizonforecast (2026). Toto 2.0; ml6.eu (2026). Forecasting System Imbalance with Chronos-2; emergentmind (2025). Chronos-Bolt electricity-price forecasting; Spheron (2026). Deploy Time Series Foundation Models on GPU Cloud; howtostoreelectricity (2026). Time Series Transformer Foundation Models.
- Operational pain metric: unplanned downtime ~$260,000/hour — widely cited industry estimate (not a single authoritative source).
