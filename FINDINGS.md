# What Makes a Hacker News Launch Go Viral? We Analyzed 190,000 Show HN Posts to Find Out

We built machine learning models trained on every Show HN post ever submitted — 190,000 project launches spanning 15 years — to predict which ones go viral. Here's what the data actually says.

---

## The Dataset

We pulled the complete Hacker News dataset from Hugging Face (`open-index/hacker-news`), filtered to Show HN posts only, and defined "viral" as landing in the top 10% by score — roughly 22+ points. That's 17,794 viral launches out of 189,892 total.

We trained LightGBM classifiers on 46 features across timing, title style, narrative signals, tech stack, domain signals, and topic categories. Our best model — trained on the past 3 years of Show HN posts — reached **AUC 0.757**, meaningfully better than chance and good enough to surface real patterns.

---

## Finding #1: AI/ML Is the Worst Category to Launch In

This one surprised us.

| Topic | Show HN Posts | Viral Rate |
|---|---|---|
| Open Source Tools | 7,728 | **13.9%** |
| Systems (Rust, DBs, compilers) | 11,878 | **12.3%** |
| Web Dev Tools | 11,726 | **11.2%** |
| Hardware | 1,649 | 10.5% |
| Security | 7,050 | 10.1% |
| Business / Startup | 2,812 | 8.9% |
| Science | 1,521 | 7.8% |
| Policy | 2,625 | 6.9% |
| **AI / ML** | **25,839** | **4.6%** |

AI/ML has more Show HN submissions than every other category combined — and the lowest virality rate of all of them. The category is so flooded that even genuinely good AI projects get buried.

And the trend is getting worse:

| Year | AI/ML Viral Rate |
|---|---|
| 2021 | 15.0% |
| 2022 | 10.9% |
| 2024 | 7.1% |
| 2025 | 4.5% |

Three years ago, an AI project had a 1-in-7 shot at going viral on Show HN. Today it's closer to 1-in-22. Meanwhile, systems tools have held steady at 12–19% virality since 2015.

**The takeaway:** If you're launching an AI wrapper or ChatGPT-adjacent product, expect low HN traction regardless of quality. If you're launching an open source infrastructure tool, your odds are roughly 3x higher.

---

## Finding #2: HN Rewards Technical Depth, Not Hype

The keywords most associated with viral Show HN posts tell a clear story:

| Keyword | Posts | Viral Rate |
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

Notice what's not here: "AI", "GPT", "productivity", "SaaS", "startup". The words that predict virality are words that signal you built something technically ambitious — a compiler, a REPL, an emulator, a low-level networking tool.

The word "implemented" at 24.5% is particularly telling. Titles like "I implemented X from scratch" or "Show HN: A Lisp interpreter implemented in Zig" consistently outperform. HN readers reward the work, not the pitch.

**Niche language communities are disproportionately powerful.** Lua, Lisp, Haskell, Zig — these communities are small but extremely loyal upvoters. Releasing a tool in or for a niche language gives you a built-in engaged audience.

---

## Finding #3: Post at Noon UTC on a Sunday

**Best hours to post (UTC):**

| Hour (UTC) | Viral Rate |
|---|---|
| 12:00 | 11.0% |
| 11:00 | 10.3% |
| 18:00 | 10.3% |
| 17:00 | 10.2% |
| 15:00 | 10.1% |

**Best days:**

| Day | Viral Rate |
|---|---|
| Sunday | 10.6% |
| Saturday | 9.9% |
| Monday | 9.5% |
| Friday | 9.0% |

This is the opposite of what you might expect. For Show HN specifically, **noon UTC on a Sunday** is the sweet spot. The most likely explanation: during business hours, developers are actively browsing HN and engaging with interesting projects. Weekend posts face less competition from news articles and have more dwell time on the front page.

Avoid posting late at night — midnight UTC is the worst time for Show HN (though it's the best for general news links).

---

## Finding #4: Open Source Is the Single Strongest Signal

An open source project has a 13.9% virality rate on Show HN — nearly 3x the rate of an AI project. This isn't just a topic correlation; it reflects a core HN community value.

Projects that link to GitHub in their title or URL perform significantly better than those without. Having a demo link (separate from the GitHub repo) is an additional positive signal — it lowers the activation energy for trying the project.

The pattern of top Show HN posts is consistent: open source, technically deep, with a working demo or clear GitHub repo.

---

## Finding #5: Title Style Matters More Than Topic

Across all our models, **title style features outrank topic features** in importance:

1. Caps ratio (fraction of uppercase letters)
2. Title length
3. Word count
4. Posting hour
5. Whether it has a parenthetical

The caps ratio finding is nuanced — it's not that all-caps titles win. It's that titles with unusual capitalization patterns (like `MyTool` or `SQLite`, `WebGL`, `MIDI`) stand out and tend to be more specific and technical.

**Titles with a parenthetical perform well** — e.g., `Show HN: FastDB – a zero-dependency SQLite alternative (written in Zig)`. The parenthetical signals specificity and lets you pack in a key differentiator without cluttering the main title.

Ideal title structure based on what wins:
```
Show HN: [What it is] – [Key differentiator] ([tech/language/constraint])
```

Examples:
- `Show HN: A Lisp interpreter that compiles to WebAssembly (written in Zig)`
- `Show HN: SQLite-compatible database with sub-millisecond replication`
- `Show HN: I implemented a MIDI sequencer from scratch in the browser`

---

## Finding #6: "I Built" Beats "We Built" — and "In X Days" Beats Both

One of the most consistent patterns across our models is that narrative framing matters.

Posts that start with **"I built"** (solo builder) consistently outperform those starting with **"We built"** (team launch). This isn't just a sample size effect — it holds across topics and time periods. HN readers appear to reward the solo hacker narrative: one person solving a real problem from scratch.

The **"built in X days/weeks"** framing gets a further boost. Titles like:
- `Show HN: I built a full-text search engine in a weekend`
- `Show HN: I wrote a Postgres-compatible DB in 14 days`

...outperform equivalent projects without the timeframe signal. The constraint makes the achievement more concrete and the story more compelling.

**Why it matters:** If you built something yourself over a weekend or a few weeks, say so explicitly in the title. It's a signal HN has historically rewarded.

---

## Finding #7: Don't Submit from the Same Domain Twice in a Row

`days_since_domain_show_hn` — the number of days since your domain last appeared in a Show HN post — ranked **3rd most important feature** in both the all-time and 3-year models, behind only title style.

The pattern is clear: repeat submitters get significantly less traction. A domain that submitted a Show HN three months ago gets meaningfully lower virality on its next submission, regardless of what the new project is. The community has already seen you once, and the novelty is gone.

This effect is strong enough that for frequent builders, the single most impactful thing you can do before your next Show HN is **wait**. Spacing submissions at least 6 months apart correlates with higher performance.

It also suggests that launching a project on a fresh domain (GitHub repo or dedicated project URL vs. your personal domain) is a real tactical advantage.

---

## What We Built

The predictor is a LightGBM classifier trained on 46 features. We train five versions:

| Model | Training Data | AUC | Features |
|---|---|---|---|
| `full` | All 4.6M HN posts | 0.667 | 31 (legacy) |
| `show_hn` | 190K Show HN posts, all time | 0.696 | 46 |
| `show_hn_3y` | 74K Show HN posts, 2023–now | **0.757** | 46 |
| `recent_5y` | 1.5M posts, 2021–now | 0.735 | 31 (legacy) |
| `recent_1y` | 329K posts, 2025–now | 0.747 | 31 (legacy) |

The `show_hn_3y` model is the recommended choice for anyone launching a Show HN post — it's trained exclusively on the most recent patterns and has the highest AUC.

```bash
python3 predict.py --title "Show HN: A Postgres-compatible database written in Rust" \
                   --url "https://github.com/..." \
                   --model show_hn_3y
```

---

## Practical Checklist for Launching on Show HN

Based on the data:

- [ ] **It's open source** — closed-source launches have ~3x lower virality
- [ ] **It has a GitHub link** — not just a landing page
- [ ] **The title signals technical depth** — use specific technical terms, not marketing language
- [ ] **Post on Sunday or Saturday, 11am–6pm UTC**
- [ ] **Avoid launching as "an AI tool"** — if AI is incidental to your project, don't lead with it
- [ ] **Title has a parenthetical** with a key constraint or implementation detail
- [ ] **It does one thing well** — Swiss army knife projects underperform focused tools
- [ ] **Use "I built" not "we built"** — the solo narrative consistently outperforms team framing
- [ ] **Add a timeframe if it's honest** — "I built this in a weekend" is a strong signal
- [ ] **Wait at least 6 months since your last Show HN** — repeat submitters get significantly less traction
- [ ] **Launch from a fresh domain** — a dedicated project URL or GitHub repo outperforms a personal domain with prior Show HN history

---

## Caveats

- "Viral" here means top 10% by score — not necessarily front page domination. A score of 22 counts.
- This model predicts based on title and timing alone — it can't assess actual project quality, which matters enormously.
- HN's algorithm (including time decay and flagging) isn't captured here.
- The 2023 data has anomalies (near-zero virality rates) suggesting a data collection gap in the source dataset.

---

*Code and data pipeline: [github.com/mrimek/hn-virality-predictor](https://github.com/mrimek/hn-virality-predictor)*
