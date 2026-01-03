# AI Presentation Generator Prompt

**Instructions**: Create a professional, modern football analytics presentation. Use a dark theme with accent colors (blue/gold or team colors). Include placeholder areas for charts/visualizations where noted.

---

## PTP Score: Predicting Pro Transition Potential from Shrine Bowl Performance

Subtitle: A Machine Learning Framework for Identifying NFL-Ready Talent

Team: [Your Name]

---

## The Problem: Finding Diamonds in the Rough

NFL teams invest millions evaluating draft prospects, yet late-round picks like Brock Purdy (Round 7) and Isiah Pacheco (Round 7) become stars while first-rounders bust.

Traditional scouting relies heavily on:
- Subjective film evaluation
- Combine testing in controlled environments
- College production that may not translate

What if Shrine Bowl practice data could reveal which players are truly NFL-ready before draft day?

---

## Our Solution: The PTP Score

**Pro Transition Potential (PTP)** is a machine learning model that predicts NFL rookie playing time using:

- **Tracking Data**: 43M+ Zebra RFID records from Shrine Bowl practices capturing real movement patterns
- **Kinematic Features**: Speed, burst, change-of-direction, acceleration derived from raw coordinates
- **Athletic Profiles**: Combine metrics normalized by position into composite indices
- **College Production**: Career stats transformed into efficiency and dominance metrics

The result: A 0-100 score predicting a player's likelihood to earn significant rookie-year snaps.

---

## How It Works: Feature Engineering

**Tracking Features** (extracted from practice drills):
- Max speed, 75th/90th percentile speed
- Direction changes per minute (agility indicator)
- Speed coefficient of variation (consistency)

**Athletic Indices**:
- Athleticism Index = weighted combination of 40-time, vertical, broad jump
- Explosion Index = vertical + broad jump composite
- Agility Index = 3-cone + shuttle composite

**Production Metrics**:
- Dominator Rating = (yards + TDs) / team total
- Per-season efficiency rates
- Career trajectory (improving vs. declining)

[Visual: Feature importance bar chart]

---

## Position-Specific Models Outperform

One-size-fits-all models miss position nuances. We trained separate models:

| Position Group | AUC Score | Key Features |
|---------------|-----------|--------------|
| SKILL (WR/RB/TE) | **0.65** | Direction changes, vertical, 75th% speed |
| DB | 0.55 | Tackles/season, speed CV, 3-cone |
| OL | 0.52 | Bench reps, athleticism index, explosion |

**Insight**: Skill position success is most predictable from practice tracking data. Lineman evaluation requires different signals.

[Visual: Position-specific leaderboard images]

---

## Validation: The Model Found the Sleepers

Our model identified late-round stars BEFORE their NFL success:

**Brock Purdy** (QB, Round 7 / Mr. Irrelevant)
- PTP Score: 72.8 (Rank 18 of 113)
- Model saw: Elite efficiency metrics, strong speed percentiles

**Isiah Pacheco** (RB, Round 7)
- PTP Score: 73.0 (Rank 16 of 113)
- Model saw: Exceptional burst, high direction change rate

**Tarheeb Still** (DB, Round 5)
- PTP Score: 93.4 (Rank 1 of 113)
- Model saw: Top-tier speed, outstanding combine metrics

[Visual: SHAP force plots showing feature contributions]

---

## Case Study: Beanie Bishop - 2025 Sleeper Alert

**Shannon "Beanie" Bishop** | CB | Undrafted

- PTP Score: **80.5** (Rank 4 of 113 players)
- Predicted Snaps: 689 (actual: 724)

**Why the model is bullish**:
- Exceptional practice speed metrics (p75 speed in top 10%)
- High direction change rate indicates elite footwork
- Strong combine profile despite being overlooked

**The story**: Despite going undrafted, Beanie's practice tracking data showed NFL-caliber movement patterns that scouts may have missed.

[Visual: Case study radar chart with feature breakdown]

---

## Key Takeaways for NFL Evaluators

1. **Practice data reveals hidden value**: Tracking metrics from Shrine Bowl drills differentiate players beyond traditional scouting

2. **Position context matters**: Skill positions show strongest signal; lineman models need different approaches (blocking efficiency, hand timing)

3. **Late-round value exists**: Players with high PTP scores but low draft capital (or undrafted) represent arbitrage opportunities

4. **Model limitations**: Sample size of 113 players; OL/DL models need blocking-specific features; need more years of data for stability

---

## Methodology and Reproducibility

**Data Sources**:
- Shrine Bowl tracking data (2022-2024 practices)
- NFL Combine results
- College career statistics
- NFL rookie snap counts

**Technical Stack**:
- XGBoost with GridSearchCV hyperparameter tuning
- Feature selection via Random Forest importance
- 5-fold cross-validation
- SHAP for model interpretability

**Code**: Full pipeline available at [GitHub link]

All analysis uses only provided Shrine Bowl datasets and publicly available statistics.

---

## Appendix: Top 20 PTP Leaderboard

| Rank | Player | Pos | PTP Score | Draft Rd | Actual Snaps |
|------|--------|-----|-----------|----------|--------------|
| 1 | Tarheeb Still | CB | 93.4 | 5 | 881 |
| 2 | Renardo Green | CB | 88.0 | 2 | 751 |
| 3 | Jarrian Jones | CB | 87.2 | 3 | 812 |
| 4 | Beanie Bishop | CB | 80.5 | UDFA | 724 |
| 5 | Tip Reiman | TE | 80.3 | 3 | 639 |
| 6 | Dadrion Taylor-Demerson | S | 80.3 | 4 | 544 |
| 7 | Chigoziem Okonkwo | TE | 79.0 | 4 | 601 |
| 8 | Tyrone Tracy | RB | 77.2 | 5 | 673 |
| 9 | Percy Butler | S | 75.3 | 4 | 407 |
| 10 | Jalen Coker | WR | 74.6 | UDFA | 461 |

[Include remaining top 20 in visual format]

