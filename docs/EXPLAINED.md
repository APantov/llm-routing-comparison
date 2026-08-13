# EXPLAINED — the whole project in plain language

The no-jargon version, covering the **ideas**. For how the code is laid out and
a trace of one real task through it, read [WALKTHROUGH.md](WALKTHROUGH.md). For
the method and the full numbers, [METHOD.md](METHOD.md) and
[RESULTS.md](RESULTS.md).

It assumes you know Python exists and nothing else.

---

## 1. The problem in one paragraph

Big language models are expensive. Small ones are cheap and get things wrong
more often. If you run a service answering thousands of questions, sending every
one to the big model wastes money, because most questions are easy. So you want
the easy ones to go to the cheap model and the hard ones to the expensive model.
Deciding which is which is called **routing**.

## 2. Two ways to route, and why they fail differently

**Predictive routing — guess first.** Look at the question, guess whether it is
hard, send it to one model. You pay for exactly one answer. But if you guess
wrong you get a bad answer and **you never find out**, because nothing is
checking the guess.

**Cascade routing — try cheap, then check.** Send it to the cheap model, look at
the answer, and escalate only if the answer seems bad. You never waste the
expensive model on an easy question. But when you do escalate you pay **twice**
— once for the cheap answer you threw away, once for the expensive one.

Predictive routing is cheaper but blind. Cascading costs more but actually looks
at the answer.

## 3. What this project measures

A cascade only works if you can tell a bad answer from a good one **without
already knowing the right answer**. That checker is the **verifier**, and it is
the whole point:

> **A cascade is exactly as good as its verifier.** A predictive router has no
> verifier at all. So "which should I use" is really "how good is my verifier".

To measure that you need a verifier whose quality you can turn up and down like
a dial. That is what this repository builds.

## 4. Why there are two kinds of task

The 417 questions come in two flavours, chosen because they sit at opposite ends
of the verifier problem.

**60 maths problems** (MATH-500, hardest level). To check a maths answer you
would need to know the answer, and at runtime you do not. So the verifier
cheats: ask the cheap model the same question **five times** with randomness on,
and see whether it agrees with itself.

- All five agree → it probably knows. Accept.
- The answers scatter → it is guessing. Escalate.

That works, sort of. It is a guess about a guess, and it costs five cheap calls
instead of one.

**357 programming problems** (MBPP+). Each ships with test code. To check the
answer you just **run the tests**. Free, and never wrong.

One half has a terrible verifier, the other a perfect one. That contrast is the
experiment.

## 5. The trap in that contrast, and the fix

Here is a mistake the project could easily have made, and the single most
important idea in it.

Suppose the code half does well and the maths half badly. Tempting conclusion:
"cascades need a good verifier, proved."

That comparison is worthless, because **five things changed at once**:

| | code half | maths half |
|---|---|---|
| verifier | perfect | a guess |
| subject | programming | mathematics |
| dataset | MBPP+ | MATH-500 |
| grading | run tests | compare text |
| verifier cost | free | 4 extra calls |

Maybe cascades need good verifiers. Maybe programming is just easier than
competition maths. Nothing in that table can tell you which, so it is not a
finding.

**The fix:** take the code half, where the verifier is perfect, and
**deliberately break the verifier** by a controlled amount. Run it again. Break
it more. Same questions, same models, same grading — only the verifier's
reliability moves.

The dial is a number `p` between 0 and 1:

- `p = 0` → the verifier reads the test results properly. Perfect.
- `p = 0.5` → half the time it ignores the tests and flips a coin.
- `p = 1` → always a coin flip. Useless.

Now you get a **curve** instead of two dots, and it answers a real engineering
question: *how bad can my verifier get before cascading stops being worth it?*

![Cascade quality against verifier quality](../figures/degradation.wide.svg)

The answer, on 357 code tasks: accuracy falls from 96.1% to 89.1% as the
verifier goes from perfect to worthless, and cost more than doubles. Cascading
survives a lot of verifier damage — but it degrades smoothly rather than
falling off a cliff, which means **there is no threshold below which you can
stop caring about your verifier.**

## 6. The comparison nobody remembers to make

Say a router gets 90% right and the cheap model alone gets 83%. Seven points
better. The router works, right?

**No.** The router also sent a third of the questions to the expensive model. So
compare it against a router that picks a third of the questions **at random**.
That coin flip also beats the cheap model, because a third of the traffic now
goes somewhere better. It has zero intelligence and still looks good.

The honest question is not "did the router beat the cheap model" but "did it
beat a coin flip that spends the same money". This repo calls that
`random_matched`, and the answer is the most useful negative result here:

> **Neither learned routing nor LLM-as-router beats the coin flip. Six
> comparisons, three ladders, not one of them significant.** The cascade beats
> both on every ladder.

The reason is structural rather than a criticism of those routers. A predictive
router commits before seeing an attempt. A cascade decides after verifying one.
Looking is worth more than guessing.

## 7. Where the numbers come from

Every accuracy figure in this repository is a **real model answering a real
question**. 5,075 responses were bought from Anthropic and DeepSeek for $8.51,
and every one is committed to `cache/`.

That is what makes the project reproducible by a stranger: the responses are in
the repository, so you can re-run the entire analysis — every policy, every
ladder, the whole degradation sweep — and get **exactly the same numbers**, with
no API key and no network, for **$0.00**.

There are three modes:

| mode | what it does | costs money? |
|---|---|---|
| **replay** | re-reads answers a real run already paid for | no |
| **real** | actually calls the API | **yes** |
| **mock** | makes up replies from a formula, for testing plumbing | no |

Mock mode exists so the pipeline can be developed and tested without spending —
it is what CI runs. Its numbers are fabricated and labelled as such everywhere
they appear: a banner at the top and bottom of every run, a tag above every
table, and a `"simulated": true` field on every output row. Nothing fabricated
is committed.

## 8. Why the same call is never made twice

Ten policies all need "the cheap model's answer to question 7". Without care
that is ten API calls returning **ten slightly different answers**, because
these models are not deterministic even at temperature 0.

That would quietly wreck the comparison. If two policies disagree you could not
tell whether their *strategies* differ or whether they got different rolls of
the dice. So every call goes through a cache: the first answer is saved and
everyone else gets that one. All policies are judged on identical model output,
which is the only way the comparison means anything.

One subtlety: **the cache does not make policies look cheaper.** Each is still
charged full price for every call, even a cached one, because in production you
run one policy and have nothing to share with. The cache saves *the experiment*
money, not the policy. Both numbers are reported.

## 9. What the `oracle` is, and the bug that was in it

The `oracle` cheats: it tries everything, sees what worked, and reports the
cheapest thing that got the right answer. You cannot deploy it — it needs the
answer in order to pick.

It exists as a **ceiling**. If the best real router scores 95.7% and the oracle
scores 96.2%, routing is nearly maxed out and further optimisation is not where
the wins are.

There was a real bug here, and it shows how this kind of project goes wrong
quietly. The oracle used to choose between two options: the cheap answer or the
expensive one. But the maths cascade has a *third* — the majority vote across
five cheap samples. That was outside the oracle's choices, so **the cascade
could score above the "ceiling"**. It did. A ceiling that is not a ceiling makes
every "fraction of available improvement captured" number meaningless. It is
fixed, and `run_eval` prints an explicit bound check every run so it cannot come
back unnoticed.

## 10. Does the difference survive a significance test?

The report shows `cascade` at 95.7% and `always_expensive` at 92.3%. Three and a
half points. Is that real, or luck?

The right test is a **paired** one: every policy answered the same questions, so
instead of comparing two averages you look only at the questions where the two
**disagreed**. On the held-out half of the `wide` ladder there are 209 tasks,
and the cascade wins that comparison at **p = 0.039**.

This is where sample size bites. An earlier version of this project ran on 100
tasks and found **zero of eight** comparisons significant, and honestly reported
that accuracy differences were undetectable. Growing the code half from 35 tasks
to 357 changed it to **four of eight** — because the number of tasks that can
tell two routers apart at all rose from 7 to 73.

The lesson is worth more than the result: *most of the tasks in a routing
benchmark are doing no work.* Both models get them right, or both get them
wrong. Only the disagreements carry information, and there are far fewer of them
than the headline task count suggests.

## 11. Policies are curves, not points

Every policy has a dial. The cascade has "how much must the samples agree before
I trust them". The learned router has "how confident must I be before I
upgrade". Turn any dial toward caution and you buy better answers with money;
turn it toward thrift and the reverse.

So comparing two policies at one dial setting each is close to meaningless —
whoever set the dials picks the winner. "Our router beat the cascade" usually
means "our router was set to spend more".

`frontier.py` sweeps every dial across its whole range and draws each policy as
a **line** on a cost-versus-accuracy chart:

![Cost-quality frontier](../figures/frontier.wide.svg)

Now you can ask the question that matters — *at my budget, which line is
highest?* — and the answer differs at different budgets, which is exactly what a
single table row cannot capture.

## 12. The false choice at the heart of the project

The repo is framed as "cascade versus predictive routing". One paper turns that
framing on its head.

Dekoninck, Baader and Vechev proved the two are not rival designs at all: both
are special cases of a single strategy, and doing both at once beats either
alone. The unified version is simple to describe — give every model a guess at
how likely it is to be right and a known price, pick the best quality per dollar
available now, then use the verifier to update the guess and decide whether to
stop or pay for another.

A cascade is that strategy forced to always start at the cheapest model. A
predictive router is that strategy forbidden from ever taking a second step.
Neither restriction is a good idea, and both are implemented here so the
unrestricted version (`cascade_routing`) can be measured against them.

Their main conclusion is also, independently, this project's thesis: **how good
your quality estimator is decides whether any of this works.** They tested it by
adding artificial noise to a quality signal. This repo tests it by degrading a
real verifier on tasks with real right answers.

## 13. Two things that turned up along the way

**The two models count tokens differently.** Text is chopped into "tokens" and
you are billed per token. Newer Claude models chop the same text into roughly
30% more tokens than older ones. So the expensive rung is not simply 5x the
price of the cheap one: it charges 5x per token *and* counts about 1.3x as many
tokens for the identical prompt. That works *against* cascading, because
escalation means paying the inflated input cost on top of the cheap call you
already made. It is modelled explicitly rather than ignored.

**Greedy decoding is not deterministic.** Temperature 0 is supposed to give the
same answer every time. Across 21 tasks with five or more draws each, more than
one distinct answer came back on **76%** of tasks for Opus 5 and **67%** for
DeepSeek v4-flash. This matters for the whole field, not just here: a routing
benchmark that takes one draw per model per task is partly measuring luck. When
this project redrew the tasks that decide its headline number, roughly a fifth
of the apparent routing opportunity turned out to be one model having a bad day.

## 14. Why half the questions are hidden from the tuning

If you pick the agreement threshold by trying several on all 417 questions and
then report the best score, the number partly measures your own choice. That is
the classic mistake and it inflates results silently.

So `splits.py` cuts the set in half, stratified so both halves hold the same mix
of subjects and difficulties. Thresholds and quality estimates are fitted on one
half; every reported number comes from the other, which the tuning never saw.
That is why runs report **n=209**, not n=417.

## 15. What to run, in order

```bash
python -m llm_routing.build_taskset     # 1. build the 417 questions from data/
python -m llm_routing.sanity_check      # 2. prove the graders work
python -m llm_routing.splits            # 3. show the calibration / evaluation split
python -m llm_routing.run_eval          # 4. every policy, reported on the hidden half
python -m llm_routing.frontier          # 5. draw each policy as a curve
python -m llm_routing.stats             # 6. ask whether any difference is real
python -m llm_routing.sweep_degraded    # 7. the verifier-quality curve
python -m llm_routing.plot              # 8. write figures/*.svg
```

Add `ROUTER_MODE=replay` to any of them to run against the real committed
responses instead of the mock.

Step 2 is the one to care about: if a grader cannot score a *known correct*
answer then every number after it is garbage, so it exits with an error rather
than printing something and carrying on.

Step 6 is the one to believe. It is the only script that will tell you a
difference you were excited about is not there.

None of it needs an API key or a network connection, and the only installed
package anywhere is numpy — which grades the code half, because every expanded
MBPP+ test program imports it. The charts need nothing: they are written as SVG
text by hand rather than with a plotting library.
