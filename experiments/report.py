"""Turn experiments/results.json into experiments/RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def num(x: float) -> str:
    return f"{x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def small_data(r, first) -> str:
    conditions = [k for k in r if k != "facts_per_set"]
    rows = [[c, pct(r[c]["set A"]), pct(r[c]["set B"]), pct(r[c]["set A after set B"])]
            for c in conditions]
    return "\n\n".join([
        "## 1. Learning from small datasets",
        f"Two unrelated sets of {r['facts_per_set']:.0f} made-up facts (\"A\" then \"B\"). "
        "\"Told once\" means each fact is stated a single time. The comparison is the "
        "model's own network with episodic memory and System 2 switched off, trained "
        "at its full learning rate on every statement, which is how a conventional "
        "network would learn the same stream.",
        table(["", "recall of A", "recall of B", "recall of A after learning B"], rows),
    ])


def systems(r, first) -> str:
    tasks = list(r["by_mode"]["switching"])
    out = ["## 2. Switching between System 1 and System 2",
           "Facts are told once, then the model sleeps. \"Steps\" counts every look-up "
           "made to produce the answer; the cheapest possible answer costs 2 (the answer "
           "and the full stop). For \"cannot be known\" the right behaviour is to say "
           "it does not know."]
    for metric, label, fmt in (("accuracy", "Correct", pct), ("system2", "Used System 2", pct),
                               ("steps", "Steps per question", num)):
        rows = [[mode] + [fmt(r["by_mode"][mode][task][metric]) for task in tasks]
                for mode in r["by_mode"]]
        out += [f"**{label}**", table(["mode"] + tasks, rows)]
    habit = r["habit"]
    rows = [[when, pct(habit[when]["accuracy"]), pct(habit[when]["system2"]),
             num(habit[when]["steps"])] for when in habit]
    out += ["**What System 2 works out becomes a System 1 habit** (the 2-fact questions, asked twice)",
            table(["", "correct", "used System 2", "steps"], rows)]
    return "\n\n".join(out)


def context(r, first) -> str:
    depths = list(r["flexible"]["by_depth"])
    rows = []
    for name, result in r.items():
        rows.append([name] + [pct(result["by_depth"][d]["accuracy"]) for d in depths]
                    + [pct(result["overall"])])
    window = [["flexible"] + [num(r["flexible"]["by_depth"][d]["window"]) for d in depths]]
    return "\n\n".join([
        "## 3. A context window that changes per task",
        "Each task's answer depends on exactly the last *d* words; anything earlier is "
        "noise that differs every time. Tasks with different *d* are mixed in one "
        "stream. Each rule is shown 3 times. \"Fixed at n\" is the same memory forced "
        "to always read exactly n words.",
        "**Correct, by how many words the task needs**",
        table(["context"] + [f"d={d}" for d in depths] + ["all"], rows),
        "**Words of context the flexible model actually read**",
        table([""] + [f"d={d}" for d in depths], window),
        f"The flexible model starts with a longest context of 8 words. The d=12 tasks "
        f"cannot be told apart within 8, so it raised its own limit to "
        f"{r['flexible']['context_limit']:.0f}.",
    ])


def pain(r, first) -> str:
    a, b = r["with pain"], r["without pain"]
    change = [k for k in a if k.startswith("told the old fact")]
    rows = [[k, num(a[k]["tellings needed to change"]), num(b[k]["tellings needed to change"]),
             num(a[k]["times it still gave the old answer"]),
             num(b[k]["times it still gave the old answer"])] for k in change]
    keep = "keeps a belief told 10x after one false statement"
    cut = "recall after losing three quarters of memory"
    inv = "invents an answer about a stranger"
    fresh = "known facts, not yet consolidated"
    other = [
        ["Keeps a belief told 10 times after one false statement", pct(a[keep]), pct(b[keep])],
        ["Repeats an answer it was rebuked for", pct(a["repeats an answer it was rebuked for"]),
         pct(b["repeats an answer it was rebuked for"])],
        ["Memory cut to a quarter: recall of facts learned in pain",
         pct(a[cut]["facts learned in pain"]), pct(b[cut]["facts learned in pain"])],
        ["Memory cut to a quarter: recall of facts learned calmly",
         pct(a[cut]["facts learned calmly"]), pct(b[cut]["facts learned calmly"])],
        ["Memory cut to a quarter: recall of all facts",
         pct(a[cut]["all facts"]), pct(b[cut]["all facts"])],
        ["Invents an answer about someone it never heard of, before any rebuke",
         pct(a[inv]["before any rebuke"]), pct(b[inv]["before any rebuke"])],
        ["The same, after being rebuked for it and after the pain has gone",
         pct(a[inv]["after being rebuked for it, pain gone"]),
         pct(b[inv]["after being rebuked for it, pain gone"])],
        ["Rebukes it took", num(a[inv]["rebukes received"]), num(b[inv]["rebukes received"])],
        ["Accuracy on known facts afterwards",
         pct(a[inv]["accuracy on known facts afterwards"]),
         pct(b[inv]["accuracy on known facts afterwards"])],
        ["Fresh memories checked with System 2 when calm",
         pct(a[fresh]["checked with System 2 when calm"]),
         pct(b[fresh]["checked with System 2 when calm"])],
        ["Fresh memories checked with System 2 while in pain",
         pct(a[fresh]["checked with System 2 while in pain"]),
         pct(b[fresh]["checked with System 2 while in pain"])],
        ["Steps per answer when calm / while in pain",
         f"{num(a[fresh]['steps when calm'])} / {num(a[fresh]['steps while in pain'])}",
         f"{num(b[fresh]['steps when calm'])} / {num(b[fresh]['steps while in pain'])}"],
        ["Accuracy on fresh memories while in pain",
         pct(a[fresh]["accuracy while in pain"]), pct(b[fresh]["accuracy while in pain"])],
    ]
    return "\n\n".join([
        "## 4. What the pain points do",
        "The same experiences, once with the affect system on and once with it off. "
        "With it off, pain levels stay at zero and drives never build, so every "
        "mechanism that reads them is silent; nothing else is changed.",
        "**A fact changes** (\"sky colour blue\" becomes \"sky colour green\")",
        table(["", "tellings needed, with pain", "without pain",
               "old answer still given, with pain", "without pain"], rows),
        "**Everything else**",
        table(["", "with pain", "without pain"], other),
    ])


def parameters(r, first) -> str:
    life = first["life"]
    rows = [[e["event"], e["vocab"], e["hidden_units"], e["cortex_weights"], e["traces"],
             e["context_limit"]] for e in life]
    pressure = r["pressure"]
    rows2 = [[k, pct(v["recall"]), num(v["traces kept"]), num(v["hidden units"]), num(v["sleeps"])]
             for k, v in pressure.items()]
    return "\n\n".join([
        "## 5. Flexible parameters",
        "**Size of one model over its life** (first seed)",
        table(["event", "words", "hidden units", "cortex weights", "memory traces",
               "longest context"], rows),
        f"**Memory too small for what is lived**: {r['facts']:.0f} facts need "
        f"{r['traces needed to hold everything']:.0f} traces; memory is capped at 400. "
        "The model decides for itself when to sleep.",
        table(["", "recall of all facts", "traces kept", "hidden units at the end", "times it slept"],
              rows2),
    ])


def interactions(r, first) -> str:
    kinds = r["kinds"]
    curve = r["share of all facts known after n steps"]
    return "\n\n".join([
        "## 6. Every interaction becomes training data",
        "A 300-step conversation with no training phase. The teacher states facts, asks "
        "questions, praises right answers, rebukes wrong ones and then gives the right "
        "one, and answers when the model says it does not know. Now and then a fact "
        "changes without the model being told. A turn \"changed the model\" if the hash "
        "of everything it has learned (weights, memory traces, prohibitions, caution, "
        "pain sensitivities, open questions) differs after it.",
        table(["", ""], [
            ["Turns", num(r["turns"])],
            ["Turns that changed the model", num(r["turns that changed the model"])],
            ["Statements / questions / questions with answer",
             f"{num(kinds['statement'])} / {num(kinds['question'])} / {num(kinds['question with answer'])}"],
            ["Praise / rebuke", f"{num(kinds['praise'])} / {num(kinds['rebuke'])}"],
            ["Facts that changed behind its back", num(r["facts that changed behind its back"])],
            ["Share of all facts known after 100 / 200 / 300 steps",
             " / ".join(pct(curve[k]) for k in curve)],
        ]),
    ])


def desire(r, first) -> str:
    rows = []
    for name, result in r.items():
        curve = result["facts known after n questions"]
        rows.append([name] + [pct(curve[k]) for k in curve]
                    + [num(result["questions that taught something new (of 60)"])])
    steps = list(next(iter(r.values()))["facts known after n questions"])
    return "\n\n".join([
        "## 7. Desire: what curiosity buys",
        "Half the facts are told. Then the model asks 60 questions of its own and the "
        "teacher answers those that have an answer. With drives on, curiosity picks the "
        "question the model is least sure about. With drives off it picks at random.",
        table([""] + [f"known after {k} questions" for k in steps] + ["questions that taught something new"],
              rows),
    ])


SECTIONS = {
    "small_data": small_data, "systems": systems, "context": context, "pain": pain,
    "parameters": parameters, "interactions": interactions, "desire": desire,
}


def write_report(results: dict, path: Path) -> None:
    seeds = sorted({v["seeds"] for v in results.values()})
    parts = [
        "# Results",
        "Generated by `python experiments/run.py`. Every number is the mean over "
        f"{', '.join(str(s) for s in seeds)} random seeds unless marked otherwise. "
        "All data is synthetic: made-up names, so nothing can be known in advance. "
        "Raw numbers are in `results.json`.",
    ]
    for name, render in SECTIONS.items():
        if name in results:
            parts.append(render(results[name]["mean"], results[name]["first_seed"]))
    path.write_text("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    write_report(json.loads((here / "results.json").read_text()), here / "RESULTS.md")
