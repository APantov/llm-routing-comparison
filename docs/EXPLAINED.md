# EXPLAINED — the whole repo in plain language

This is the no-jargon version of [README.md](../README.md), covering the IDEAS.
For how the CODE is laid out and a trace of one real task through it, read
[WALKTHROUGH.md](WALKTHROUGH.md).

It assumes you know Python exists and nothing else. If you have been away from
this project for a month and cannot remember what any of it does, start here.

---

## 1. The problem in one paragraph

Big language models are expensive. Small ones are cheap and get things wrong more
often. If you are running a service that answers thousands of questions, sending
every question to the big model is a waste, because most questions are easy. So
you want to send the easy ones to the cheap model and the hard ones to the
expensive model. Deciding which is which is called **routing**.

## 2. Two ways to route, and why they are different

There are two ways to do this, and they fail in opposite ways.

**Predictive routing — guess first.**
Look at the question. Guess whether it is hard. Send it to one model. Done.

- You pay for exactly one answer. Cheap.
- If you guess wrong, you get a bad answer and **you never find out**. There is
  nothing checking your guess.

**Cascade routing — try cheap, then check.**
Send it to the cheap model. Look at the answer. If the answer seems bad, send it
to the expensive model instead.

- You never send an easy question to the expensive model, because the cheap model
  already handled it.
- When you do escalate, you pay **twice** — once for the cheap answer you threw
  away, once for the expensive one.

So predictive routing is cheaper but blind. Cascading is more expensive but it
actually looks at the answer.

## 3. The thing this project is actually measuring

A cascade only works if you can tell a bad answer from a good one **without
already knowing the right answer**. That checker is called a **verifier**, and
here is the whole point of the project:

> **A cascade is exactly as good as its verifier.** A predictive router does not
> have a verifier at all. So the question "which one should I use" is really the
> question "how good is my verifier".

To measure that, you need a verifier whose quality you can turn up and down like
a dial. That is what this repo builds.

## 4. Why there are two kinds of task

The 100 test questions come in two flavours, picked on purpose because they sit at
opposite ends of the verifier problem.

**60 maths problems** (from a set called MATH500). To check a maths answer you
would need to know the answer, and at runtime you do not. So the verifier has to
cheat: ask the cheap model the same question **five times** with some randomness
turned on, and see whether it gives the same answer each time.

- All five agree → it probably knows. Accept.
- The answers are all over the place → it is guessing. Escalate.

This works, sort of. It is a **guess about a guess**. Also it costs five cheap
calls instead of one.

**40 programming problems** (from a set called MBPP). Each one ships with test
code. To check the answer you just **run the tests**. If they pass, the answer is
right. This is free and it is never wrong.

So one half of the task set has a terrible verifier and the other half has a
perfect one. That contrast is the experiment.

## 5. The problem with that contrast, and the fix

Here is a mistake the project could easily have made, and the single most
important idea in the repo.

Suppose the code half does well and the maths half does badly. Tempting
conclusion: "cascades need a good verifier, and this proves it."

But that comparison is worthless, because **five things changed at once**:

| | code half | maths half |
|---|---|---|
| verifier | perfect | a guess |
| subject | programming | mathematics |
| dataset | MBPP | MATH500 |
| grading | run tests | compare text |
| verifier cost | free | 4 extra calls |

Maybe cascades need good verifiers. Or maybe programming is just easier than
competition maths. Nothing in that table can tell you which, so it is not a
finding.

**The fix:** take the code half, where the verifier is perfect, and **deliberately
break the verifier** by a controlled amount. Then run it again. Then break it a
bit more. Same questions, same models, same grading, same everything — only the
verifier's reliability changes.

The dial is a number `p` between 0 and 1:

- `p = 0` → the verifier reads the test results properly. Perfect.
- `p = 0.5` → half the time it ignores the tests and flips a coin instead.
- `p = 1` → it always flips a coin. Completely useless.

Now you get a **curve** instead of two dots, and the curve answers a real
engineering question: *how bad can my verifier get before cascading stops being
worth it?* That is `llm_routing/sweep_degraded.py`.

## 6. The comparison nobody remembers to make

Say the smart router gets 89% right and the cheap model alone gets 74%. Fifteen
points better. The router works, right?

**No.** The router also sent about a third of the questions to the expensive
model. So compare it against a router that picks a third of the questions **at
random** and sends those to the expensive model. That coin-flip router also
scores much better than the cheap model alone, because a third of the traffic is
now going to a better model. It has zero intelligence and it still looks good.

So the honest question is not "did the router beat the cheap model" — it is "did
the router beat a coin flip that spends the same money". This repo calls that
`random_matched`, and it reports the answer as a **routing skill** percentage:

> Of all the improvement that was theoretically available, what fraction did the
> router actually find?

- **0%** → no better than flipping a coin at the same cost. The router is
  decoration.
- **100%** → as good as it is possible to be.
- **Negative** → worse than a coin flip. This happens, and the report prints it
  rather than hiding it.

## 7. What "mock mode" means, and why no number here is real yet

**This is the part that matters most for reading anything this repo prints.**

Calling real models costs money and needs an API key. Both have now been used
once, briefly: a 5-task plumbing check on 30 July 2026 spent about 5 cents and
saved 47 real replies to disk. But every policy got every one of those five tasks
right, so that run settled nothing about accuracy — it only proved the wiring
works. Everything the project *reports* still runs in **mock mode**: instead of
asking a real model, it makes up an answer.

It is not making up answers randomly. It looks at how hard the question is, and
decides *by formula* how likely each model is to get it right — the cheap model
is set to fail more often, and more often on hard questions. Then it writes out
either the correct answer (which it already has, from the answer key) or a wrong
one.

This is genuinely useful. It proves every piece of plumbing works: the graders,
the escalation logic, the cost arithmetic, the caching, the reports. All of that
is real code doing real work.

But it means:

> **Every accuracy percentage this repo prints is currently a restatement of a
> constant somebody typed into `llm_routing/models.py`.** It is not a measurement of any
> model. If you edit `MOCK_SKILL` from 0.86 to 0.95, the "results" change, and
> nothing has been learned.

What **is** real in mock mode is the money side. The prices come from the actual
published price list, and the arithmetic on top of them is correct. So "this
policy costs 2.3x that one" is a real statement about the pricing structure, even
though "this policy is 89% accurate" is not a real statement about anything.

The repo tries hard to stop you forgetting this. Every run prints a warning block
at the top **and** the bottom. Every table has `SIMULATED, NOT MEASURED` above it.
Every output row carries a `"simulated": true` flag. And `runs/results.jsonl` is not
committed to the repo, so no made-up percentage ends up published.

## 8. The three modes

| mode | what it does | costs money? |
|---|---|---|
| **mock** | makes up model replies from a formula | no |
| **real** | actually calls the API | **yes** |
| **replay** | re-reads answers a real run already paid for | no |

Replay mode is the clever one. The first real run saves every response to disk.
After that, anyone can re-run the entire experiment — including the whole
degradation sweep, hundreds of times over — and get **exactly the same numbers**
for free, with no API key at all. That is why the response cache had to be built
before any money was spent.

## 9. Why the same call is never made twice

Ten policies all need "the cheap model's answer to question 7". Without care,
that is ten separate API calls, and in real mode they would come back with **ten
slightly different answers**, because these models are not deterministic.

That would quietly wreck the comparison. If policy A and policy B disagree, you
would not know whether it was because their *strategies* differ or because they
happened to get different rolls of the dice. So every call goes through a cache:
the first answer is saved, and everyone else gets that same one. All policies are
then judged on identical model output, which is the only way the comparison means
anything.

One subtlety worth knowing: **the cache does not make policies look cheaper.**
Each policy is still charged full price for every call it makes, even a cached
one — because in production you would only be running one policy, and there would
be nothing to share with. The cache saves *the experiment* money, not the policy.
The reports print both numbers separately.

## 10. What the `oracle` is, and the bug that was in it

The `oracle` cheats. It tries everything, sees which worked, and reports the
cheapest thing that got the right answer. You cannot deploy it, because it needs
to know the answer in order to pick.

It exists as a **ceiling**. If the best real router scores 89% and the oracle
scores 91%, then routing is nearly maxed out and there is no point optimising
further. If the oracle scores 99%, there is a lot left on the table.

There was a real bug here, and it is a good illustration of how this kind of
project goes wrong quietly. The oracle used to choose between just two options:
the cheap answer or the expensive answer. But the maths cascade has a *third*
option — the majority vote across those five cheap samples. That option was
outside the oracle's choices, so **the cascade could score higher than the
"ceiling"**. Which it did.

A ceiling that is not a ceiling makes every "fraction of available improvement
captured" number meaningless. It is fixed now, and `llm_routing/run_eval.py` prints an
explicit check every run so it cannot come back unnoticed.

## 10b. The comparison nobody's numbers survive

Here is the most useful thing in the whole repo, and it is a negative result.

The report table shows `cascade` at 90% and `llm_router` at 88%. Two points. On
the version of this table that existed a week earlier the gap looked like six.
Either way, it looks like a win.

`llm_routing/stats.py` checks whether that gap is real. The right test here is a **paired**
one: every policy answered the same tasks, so instead of comparing two averages
you look at the tasks where the two policies *disagreed*. On the held-out half
there were three such tasks — two where the cascade was right and the router
wrong, one the other way.

Three out of zero sounds convincing. It isn't. If you flip a coin three times and
get three heads, that happens one time in eight by chance. The formal version of
that gives p = 0.25, and the confidence interval on the gap runs from 0 to +14
points. Translated: *the cascade is somewhere between no better at all and much
better, and 51 tasks cannot tell you which.*

Run it and every single comparison comes out that way. **Zero of nine.**

What *does* come out significant is cost. Several of the cost differences have
intervals nowhere near zero. So the honest one-line summary of this project today
is:

> Cost differences between routing architectures are measurable at this sample
> size. Accuracy differences are not.

That is a genuinely useful thing to know, and it says clearly what to do next: more
tasks. Several hundred, not one hundred. The response cache makes that affordable —
pay for the model calls once, and every analysis afterwards is free.

## 10c. Policies are curves, not points

Related, and the reason `llm_routing/frontier.py` exists.

Every policy here has a dial. The cascade has "how much must the samples agree
before I trust them". The learned router has "how confident must I be that the
big model wins before I upgrade". Turn any dial toward caution and you get better
answers for more money. Turn it toward thrift and the reverse.

Which means comparing two policies at one dial setting each is close to
meaningless. Whoever set the dials picks the winner. "Our router beat the cascade"
usually just means "our router was set to spend more".

So `llm_routing/frontier.py` sweeps every dial across its whole range and draws each policy as
a **line** on a cost-versus-accuracy chart. Then you can ask the question that
actually matters: *at my budget, which line is highest?* The answer turns out to be
different at different budgets, which is exactly why a single table row cannot
capture it.

It also produces one number per policy — the average accuracy across the entire
budget range — so policies can still be ranked, just fairly. On that measure, on
real models, the cascade comes out well ahead of the random baseline (91.2%
against 86.3%) and RouteLLM's pretrained router comes out slightly *behind* it
(85.4%). Which is consistent with the published finding that routers trained on
human preference data transfer badly to tasks with objectively right answers.

There is a sharper version of that here now. RouteLLM's score on these 100 tasks
never once drops below 0.5 — it never judges the small model more likely to win.
On competition maths and MBPP+ it simply cannot tell, so it always says "the big
one". You can watch the router fail before measuring a single answer.

## 10d. The false choice at the heart of the project

The repo is framed as "cascade versus predictive routing". While researching this,
one paper turned that framing on its head.

Dekoninck, Baader and Vechev proved that routing and cascading aren't two rival
designs at all — they're both special cases of a single strategy, and doing both at
once beats either alone. The unified version is simple to describe:

- give every model a guess at how likely it is to get this question right, and a
  known price;
- pick whichever offers the best quality-per-dollar right now;
- after it answers, use the verifier to update the guess, and decide whether to
  stop or pay for another.

A cascade is what you get when that strategy is forced to always start at the
cheapest model. A predictive router is what you get when it is forbidden from ever
taking a second step. Neither restriction is a good idea.

That is implemented here as `cascade_routing`, and it can do two things `cascade`
cannot: skip the cheap model entirely on a question that obviously needs more, and
jump straight to the top instead of climbing one rung at a time.

The paper's main conclusion is also, independently, this project's whole thesis:
**how good your quality estimator is decides whether any of this works.** They
tested that by adding artificial noise to a quality signal. This repo tests it by
degrading a real verifier on tasks with real right answers. Same claim, harder
evidence.

## 11. Something real that turned up while cleaning this up

The two models use **different tokenizers**. Text gets chopped into "tokens", and
you are billed per token. The newer models (including the expensive one here) chop
text into roughly 30% more tokens than the older ones for the same words.

So the expensive model is not simply 5x the price of the cheap one. It charges 5x
per token **and counts about 1.3x as many tokens** for the identical prompt. The
real input ratio is closer to 6.5x.

This works *against* cascading, because escalation means paying that inflated
input cost on top of the cheap call you already made. It is now modelled
explicitly in `llm_routing/models.py` and documented next to the price table.

## 11b. Why half the questions are hidden from the tuning

If you pick the agreement threshold by trying several and keeping whichever scored
best on all 100 questions, then report that score, the number is partly measuring
your own choice rather than the method. This is the classic mistake and it inflates
results silently.

So `llm_routing/splits.py` cuts the set in half. Thresholds get tuned and quality estimates get
fitted on one half; the reported numbers come from the other half, which the tuning
never saw. That is why `llm_routing/run_eval.py` says `n=51` rather than `n=100`.

The split is deliberately not a plain coin flip. It is balanced so that both halves
contain the same mix of easy, medium and hard questions in both subjects — with only
100 items, an unlucky random split could put most of the hard maths in one half and
the two halves would not be comparable at all.

## 12. What to run, in order

```bash
python -m llm_routing.build_taskset     # 1. build the 100 questions from data/
python -m llm_routing.sanity_check      # 2. prove the graders work: must say 40/40 and 60/60
python -m llm_routing.splits            # 3. show the calibration / evaluation split
python -m llm_routing.run_eval          # 4. run every policy, report on the hidden half
python -m llm_routing.frontier          # 5. draw each policy as a curve, not a point
python -m llm_routing.stats             # 6. ask whether any difference is real
python -m llm_routing.sweep_degraded    # 7. the verifier-quality curve (the experiment)
python -m llm_routing.plot              # 8. write figures/*.svg
```

Step 2 is the one to care about. If a grader cannot score the *known correct*
answer, then every number after it is garbage — so `llm_routing/sanity_check.py` exits with
an error rather than just printing something and carrying on.

Step 6 is the one to believe. It is the only script that will tell you a difference
you were excited about isn't there.

Nothing above needs an API key, an internet connection, or any installed package —
including the charts, which are written as SVG text by hand rather than with a
plotting library, so the whole repo still runs on a bare Python install.

## 13. What is still missing

The honest list lives in [NOTES.md](NOTES.md). The short version: **the experiment
has only ever been run against real models once, on five tasks, and every policy
tied at 100%** — so there is no result yet, only a machine that has been proven to
work and is ready to produce one.
