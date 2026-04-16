# What Makes a Hacker News Post Go Viral? We Analyzed 260,000 Posts to Find Out

We trained machine learning models on Hacker News posts to predict which ones reach the top 10% by score. Here's what the data actually says.

---

## The Models

| Model | Training data | Posts | Viral rate | AUC |
|---|---|---|---|---|
| `show_hn_3y` ⭐ | Show HN, Apr 2023–Apr 2026 | 46,000 | 5.3% | **0.762** |
| `recent_5y` | All HN posts, Apr 2021–Apr 2026 | 215,000 | 9.2% | 0.714 |

"Viral" = top 10% by score within the training window. For Show HN that's roughly 22+ points.

**Training approach:** LightGBM binary classifier with Platt calibration (3-way 70/15/15 split — calibration is fitted on a separate holdout, not the test set). Class imbalance handled with `scale_pos_weight` (17.9× for `show_hn_3y`, 9.9× for `recent_5y`). 46 features across timing, title style, narrative framing, tech stack, domain, and topic signals.

---

## Finding #1: AI/ML Is the Worst Category to Launch In

This one surprised us.

| Topic | Show HN posts | Viral rate |
|---|---|---|
| Open Source Tools | 7,728 | **13.9%** |
| Systems (Rust, DBs, compilers) | 11,878 | **12.3%** |
| Web Dev Tools | 11,726 | 11.2% |
| Hardware | 1,649 | 10.5% |
| Security | 7,050 | 10.1% |
| Business / Startup | 2,812 | 8.9% |
| Science | 1,521 | 7.8% |
| Policy | 2,625 | 6.9% |
| **AI / ML** | **25,839** | **4.6%** |

AI/ML has more Show HN submissions than every other category combined — and the lowest virality rate of all of them. The category is so flooded that even genuinely good AI projects get buried.

The trend is getting worse:

| Year | AI/ML viral rate |
|---|---|
| 2021 | 15.0% |
| 2022 | 10.9% |
| 2024 | 7.1% |
| 2025 | 4.5% |

Three years ago, an AI project had a 1-in-7 shot at going viral on Show HN. Today it's closer to 1-in-22. Meanwhile, systems tools have held steady at 12–19% virality since 2015.

**The takeaway:** If you're launching an AI wrapper or ChatGPT-adjacent product, expect low HN traction regardless of quality. If you're launching an open source infrastructure tool, your odds are roughly 3× higher.

---

## Finding #2: Caps Ratio Is the Strongest Predictive Signal

The single most important feature in both models is `title_caps_ratio` — the fraction of characters in the title that are uppercase.

This is more nuanced than "don't use all-caps." The pattern is that titles with unusual or elevated capitalisation correlate strongly with lower virality. HN readers associate heavy caps with hype and marketing language.

The sweet spot is natural sentence case or standard title case. Acronyms and proper nouns (SQLite, WebGL, WASM) are fine — they tend to be specific and technical. What tanks the score is promotional capitalisation: "Build Your Next AI App With Our Platform."

**The takeaway:** Write your title like you're explaining something to a colleague, not pitching to an investor.

---

## Finding #3: Technical Depth, Not Hype

The keywords most associated with viral Show HN posts tell a clear story:

| Keyword | Posts | Viral rate |
|---|---|---|
| lua | 143 | 25.2% |
| implemented | 155 | 24.5% |
| midi | 124 | 24.2% |
| lisp | 266 | 22.6% |
| compiler | 297 | 20.9% |
| repl | 127 | 21.3% |
| webrtc | 246 | 20.3% |
| sqlite | 322 | 18.9% |
| emulator | 187 | 18.7% |
| debugger | 141 | 18.4% |
| postgres | 403 | 18.6% |
| webgl | 403 | 19.4% |

Notice what's missing: "AI", "GPT", "productivity", "SaaS", "startup". The words that predict virality signal technical ambition — a compiler, a REPL, an emulator, a low-level networking tool.

The word "implemented" at 24.5% is particularly telling. Titles like "I implemented X from scratch" or "Show HN: A Lisp interpreter implemented in Zig" consistently outperform. HN readers reward the work, not the pitch.

Niche language communities (Lua, Lisp, Haskell, Zig) are disproportionately powerful — small but extremely loyal upvoters.

---

## Finding #4: Post at Noon UTC on a Sunday

**Best hours (UTC):**

| Hour | Viral rate |
|---|---|
| 12:00 | 11.0% |
| 11:00 | 10.3% |
| 18:00 | 10.3% |
| 17:00 | 10.2% |
| 15:00 | 10.1% |

**Best days:**

| Day | Viral rate |
|---|---|
| Sunday | 10.6% |
| Saturday | 9.9% |
| Monday | 9.5% |
| Friday | 9.0% |

For Show HN specifically, **noon UTC on Sunday** is the sweet spot. Weekend posts face less competition from news articles and have more dwell time on the front page. Avoid posting late at night — midnight UTC is the worst time for Show HN.

Note: `hour` and `day_of_week` are in the top 5 features by importance. The model has a meaningful step-change between peak and off-peak hours — timing alone can shift the score by 10–15 points.

---

## Finding #5: Open Source Is the Strongest Single Content Signal

An open source project has a 13.9% virality rate on Show HN — nearly 3× the rate of an AI project. This reflects a core HN community value, not just a topic correlation.

Projects linking to GitHub perform significantly better than those without. A live demo link (separate from the GitHub repo) is an additional positive signal — it lowers the activation energy for trying the project.

The pattern of top Show HN posts is consistent: open source, technically deep, with a working demo or clear GitHub repo.

---

## Finding #6: Title Style Matters More Than Topic

Across both models, title style features rank above topic features:

1. `title_caps_ratio` — fraction of uppercase characters
2. `title_len` — character count (sweet spot: 60–80)
3. `title_sweet_spot` — binary flag for 60–80 chars
4. `has_parens` — whether the title has a parenthetical
5. `word_count` — number of words

**Titles with a parenthetical perform consistently well** — e.g., `Show HN: FastDB – a zero-dependency SQLite alternative (written in Zig)`. The parenthetical lets you pack in a key differentiator or constraint without cluttering the main title.

Ideal structure based on what wins:
```
Show HN: [What it is] – [Key differentiator] ([tech/language/constraint])
```

Examples:
- `Show HN: A Lisp interpreter that compiles to WebAssembly (written in Zig)`
- `Show HN: SQLite-compatible database with sub-millisecond replication`
- `Show HN: I implemented a MIDI sequencer from scratch in the browser`

---

## Finding #7: "I Built" Beats "We Built"

Posts starting with **"I built"** (solo builder) consistently outperform **"We built"** (team launch) across topics and time periods. HN readers appear to reward the solo hacker narrative.

The **"built in X days/weeks"** framing gets a further boost:
- `Show HN: I built a full-text search engine in a weekend`
- `Show HN: I wrote a Postgres-compatible DB in 14 days`

The constraint makes the achievement concrete and the story compelling.

---

## Finding #8: Don't Submit from the Same Domain Twice in a Row

`days_since_domain_show_hn` — days since this domain last appeared in a Show HN post — ranks as a top-5 feature by importance in both models.

Repeat submitters get significantly less traction. A domain that ran a Show HN three months ago scores meaningfully lower on its next submission regardless of what the project is. The community has already seen you once; the novelty is gone.

This effect is strong enough that **waiting** is the highest-ROI thing a frequent builder can do before their next Show HN. Spacing submissions at least 6 months apart correlates with higher performance.

Launching from a fresh domain (dedicated project URL or GitHub repo vs. your personal domain with prior Show HN history) is a real tactical advantage.

---

## Practical Checklist for Show HN

Based on the data:

- [ ] **It's open source** — closed-source launches have ~3× lower virality
- [ ] **GitHub link in the URL** — not just a landing page
- [ ] **Title signals technical depth** — specific technical terms, not marketing language
- [ ] **Post on Sunday or Saturday, 11am–6pm UTC**
- [ ] **Avoid leading with the AI angle** — if AI is incidental, don't make it the headline
- [ ] **Title has a parenthetical** with a key constraint or implementation detail
- [ ] **60–80 character title** — long enough to be specific, short enough to scan
- [ ] **"I built" not "we built"** — solo narrative consistently outperforms team framing
- [ ] **Add a timeframe if honest** — "I built this in a weekend" is a strong signal
- [ ] **Wait at least 6 months since your last Show HN**
- [ ] **Fresh domain** — a dedicated project URL outperforms a personal domain with prior Show HN history

---

## Caveats

- "Viral" = top 10% by score — not necessarily front page. A score of 22 qualifies.
- The model predicts based on title, URL, and timing — it can't assess actual project quality, which matters enormously.
- HN's ranking algorithm (time decay, flagging) isn't captured.
- Raw calibrated probability tops out at ~15% for the best Show HN posts. Even a perfect score has roughly a 15–18% chance of reaching the front page — HN has a large random/community component no model can predict.
