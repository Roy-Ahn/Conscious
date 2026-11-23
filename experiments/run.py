"""Measure each property the model is required to have.

    python experiments/run.py            # all experiments, 5 seeds
    python experiments/run.py --seeds 2 --only small_data context

Writes experiments/results.json and experiments/RESULTS.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conscious import Config, Mind  # noqa: E402
from conscious.worlds import World, depth_rules, with_noise  # noqa: E402

HERE = Path(__file__).resolve().parent


def tell(mind: Mind, statements, times: int = 1) -> Mind:
    for _ in range(times):
        for statement in statements:
            mind.interact(statement)
    return mind


def probe(mind: Mind, questions, **changes) -> dict:
    """Ask on a copy, so that measuring does not teach."""
    mind = copy.deepcopy(mind)
    for name, value in changes.items():
        setattr(mind.cfg, name, value)
    replies = [(mind.interact(q), a) for q, a in questions]
    n = len(replies)
    return {
        "n": n,
        "accuracy": sum(r.text == a for r, a in replies) / n,
        "wrong": sum(r.text != a and not r.abstained for r, a in replies) / n,
        "abstained": sum(r.abstained for r, _ in replies) / n,
        "system2": sum(r.system == 2 for r, _ in replies) / n,
        "steps": sum(r.steps for r, _ in replies) / n,
        "window": sum(r.window for r, _ in replies) / n,
    }


def mean(rows: list[dict]) -> dict:
    """Average a list of same-shaped nested dicts of numbers."""
    first = rows[0]
    if isinstance(first, dict):
        return {key: mean([row[key] for row in rows]) for key in first}
    return round(float(np.mean(rows)), 4)


# ------------------------------------------------------------- experiments

def small_data(seed: int) -> dict:
    """Facts told once. Then a second, unrelated set of facts."""
    a = World(chains=12, length=5, seed=seed)
    b = World(chains=12, length=5, seed=seed + 1000)
    out = {"facts_per_set": len(a.facts())}

    mind = tell(Mind(Config(seed=seed)), a.statements())
    out["full model, told once"] = {"set A": probe(mind, a.facts())["accuracy"]}
    tell(mind, b.statements())
    out["full model, told once"]["set A after set B"] = probe(mind, a.facts())["accuracy"]
    out["full model, told once"]["set B"] = probe(mind, b.facts())["accuracy"]

    for times in (1, 20):
        net = Mind(Config(seed=seed, use_memory=False, allow_system2=False, auto_sleep=False,
                          awake_rate=1.0))
        tell(net, a.statements(), times)
        row = {"set A": probe(net, a.facts())["accuracy"]}
        tell(net, b.statements(), times)
        row["set A after set B"] = probe(net, a.facts())["accuracy"]
        row["set B"] = probe(net, b.facts())["accuracy"]
        out[f"network alone, told {times}x"] = row
    return out


def systems(seed: int) -> dict:
    """Which system answers which kind of question, and at what cost."""
    world = World(chains=12, length=6, seed=seed)
    mind = tell(Mind(Config(seed=seed)), world.statements())
    mind.sleep()
    mind.pain.soothe(1.0)
    strangers = [(f"{name} parent", "<?>") for name in
                 World(chains=4, length=6, seed=seed + 500).people]
    tasks = {
        "told directly (1 fact)": world.facts(),
        "never told (2 facts chained)": world.hops(2),
        "never told (3 facts chained)": world.hops(3),
        "cannot be known": strangers,
    }
    modes = {
        "System 1 only": dict(allow_system2=False),
        "System 2 always": dict(force_system2=True),
        "switching": dict(),
    }
    out = {"by_mode": {}, "habit": {}}
    for mode, changes in modes.items():
        out["by_mode"][mode] = {
            task: {k: v for k, v in probe(mind, questions, **changes).items()
                   if k in ("accuracy", "system2", "steps")}
            for task, questions in tasks.items()
        }
    again = copy.deepcopy(mind)
    for label in ("first time asked", "second time asked"):
        replies = [again.interact(q) for q, _ in tasks["never told (2 facts chained)"]]
        out["habit"][label] = {
            "accuracy": float(np.mean([r.text == a for r, (_, a) in
                                       zip(replies, tasks["never told (2 facts chained)"])])),
            "system2": float(np.mean([r.system == 2 for r in replies])),
            "steps": float(np.mean([r.steps for r in replies])),
        }
    return out


DEPTHS = [1, 2, 3, 5, 8, 12]


def context(seed: int) -> dict:
    """Tasks whose answer depends on the last d words, for several d at once."""
    rules, noise = depth_rules(DEPTHS, per_depth=10, seed=seed)

    def run(window):
        rng = np.random.default_rng(seed)
        start = 8 if window is None else 16
        mind = Mind(Config(seed=seed, auto_sleep=False, fixed_window=window, max_context=start))
        for _ in range(3):
            for _, words, answer in rules:
                mind.interact(with_noise(words, noise, rng) + f" {answer} .")
        by_depth = {}
        for depth in DEPTHS:
            questions = [(with_noise(words, noise, rng), answer)
                         for d, words, answer in rules if d == depth]
            result = probe(mind, questions)
            by_depth[str(depth)] = {"accuracy": result["accuracy"], "window": result["window"]}
        overall = float(np.mean([row["accuracy"] for row in by_depth.values()]))
        return {"overall": overall, "by_depth": by_depth, "context_limit": mind.memory.L}

    out = {"flexible": run(None)}
    for window in DEPTHS:
        out[f"fixed at {window}"] = run(window)
    return out


def pain(seed: int) -> dict:
    """The same experiences with the affect system on and off."""
    out = {}
    for label, affect in (("with pain", True), ("without pain", False)):
        row = {}

        # (a) the world changes: how many times must it be told?
        for known in (3, 10, 30):
            mind = tell(Mind(Config(seed=seed, affect=affect, auto_sleep=False)),
                        ["sky colour blue ."], known)
            stale, needed = 0, None
            for n in range(1, 61):
                mind.interact("sky colour green .")
                said = copy.deepcopy(mind).interact("sky colour").text
                stale += said == "blue"
                if said == "green":
                    needed = n
                    break
            row[f"told the old fact {known}x"] = {
                "tellings needed to change": needed if needed is not None else 60,
                "times it still gave the old answer": stale,
            }

        # (b) one false statement against an established belief
        mind = tell(Mind(Config(seed=seed, affect=affect, auto_sleep=False)),
                    ["sky colour blue ."], 10)
        mind.interact("sky colour green .")
        for filler in ("sea colour blue .", "grass colour green .", "coal colour black .",
                       "snow colour white .", "sun colour yellow ."):
            mind.interact(filler)
        row["keeps a belief told 10x after one false statement"] = float(
            copy.deepcopy(mind).interact("sky colour").text == "blue")

        # (c) rebuke
        world = World(chains=6, length=5, seed=seed)
        mind = tell(Mind(Config(seed=seed, affect=affect)), world.statements())
        mind.sleep()
        repeated = 0
        for question, _ in world.facts():
            first = mind.interact(question).text
            mind.feedback(-1)
            repeated += mind.interact(question).text == first
        row["repeats an answer it was rebuked for"] = repeated / len(world.facts())

        # (d) what survives when memory is cut to a quarter
        world = World(chains=8, length=5, seed=seed)
        statements = world.statements()
        rng = np.random.default_rng(seed)
        marked = set(rng.choice(len(statements), size=len(statements) // 4, replace=False).tolist())
        mind = Mind(Config(seed=seed, affect=affect, auto_sleep=False))
        for i, statement in enumerate(statements):
            mind.pain.soothe(1.0)
            if i in marked:
                mind.pain.inflict("rebuke", 0.9)
            mind.interact(statement)
        mind.pain.soothe(1.0)
        mind.memory.evict(3 * mind.memory.count // 4)
        facts = world.facts()
        hurt = [facts[i] for i in range(len(facts)) if i in marked]
        calm = [facts[i] for i in range(len(facts)) if i not in marked]
        row["recall after losing three quarters of memory"] = {
            "facts learned in pain": probe(mind, hurt, allow_system2=False)["accuracy"],
            "facts learned calmly": probe(mind, calm, allow_system2=False)["accuracy"],
            "all facts": probe(mind, facts, allow_system2=False)["accuracy"],
        }

        # (e) questions about people it has never heard of
        world = World(chains=10, length=5, seed=seed)
        strangers = [(f"{name} home", "<?>") for name in
                     World(chains=6, length=5, seed=seed + 500).people]
        mind = tell(Mind(Config(seed=seed, affect=affect)), world.statements())
        mind.sleep()
        mind.pain.soothe(1.0)
        before = probe(mind, strangers[10:])["wrong"]
        rebukes = 0
        for question, _ in strangers[:10]:
            if not mind.interact(question).abstained:
                mind.feedback(-1)
                rebukes += 1
        mind.pain.soothe(1.0)
        row["invents an answer about a stranger"] = {
            "before any rebuke": before,
            "after being rebuked for it, pain gone": probe(mind, strangers[10:])["wrong"],
            "rebukes received": rebukes,
            "accuracy on known facts afterwards": probe(mind, world.facts())["accuracy"],
        }

        # (f) fresh memories, calm and hurt
        mind = tell(Mind(Config(seed=seed, affect=affect, auto_sleep=False)), world.statements())
        mind.pain.soothe(1.0)
        calm_known = probe(mind, world.facts())
        mind.pain.add_point("ache", decay=0.0)
        mind.pain.inflict("ache", 0.9)
        hurt_known = probe(mind, world.facts())
        row["known facts, not yet consolidated"] = {
            "checked with System 2 when calm": calm_known["system2"],
            "checked with System 2 while in pain": hurt_known["system2"],
            "steps when calm": calm_known["steps"],
            "steps while in pain": hurt_known["steps"],
            "accuracy when calm": calm_known["accuracy"],
            "accuracy while in pain": hurt_known["accuracy"],
        }
        out[label] = row
    return out


def parameters(seed: int) -> dict:
    """Sizes over a life, and what a growing cortex buys under memory pressure."""
    out = {"life": [], "pressure": {}}

    mind = Mind(Config(seed=seed, hidden=8))

    def note(event):
        out["life"].append({"event": event, **mind.size(), "sleeps": mind.sleeps})

    note("born")
    a = World(chains=10, length=5, seed=seed)
    tell(mind, a.statements())
    note(f"told {len(a.facts())} facts")
    mind.sleep()
    note("slept")
    b = World(chains=30, length=5, seed=seed + 1)
    tell(mind, b.statements())
    note(f"told {len(b.facts())} more facts")
    mind.sleep()
    note("slept")
    rules, noise = depth_rules([12], per_depth=4, seed=seed)
    rng = np.random.default_rng(seed)
    for _, words, answer in rules:
        mind.interact(with_noise(words, noise, rng) + f" {answer} .")
    note("met a task needing 12 words of context")
    for question, _ in a.hops(2):
        mind.interact(question)
    note("worked out 2-step answers")

    world = World(chains=40, length=5, seed=seed)
    variants = {
        "no cortex": dict(use_cortex=False),
        "cortex fixed at 4 units": dict(hidden=4, grow_cortex=False),
        "cortex that grows": dict(hidden=4),
        "cortex that grows, no pain": dict(hidden=4, affect=False),
    }
    for label, changes in variants.items():
        mind = Mind(Config(seed=seed, capacity=400, **changes))
        tell(mind, world.statements())
        out["pressure"][label] = {
            "recall": probe(mind, world.facts(), allow_system2=False)["accuracy"],
            "traces kept": mind.memory.count,
            "hidden units": mind.cortex.hidden,
            "sleeps": mind.sleeps,
        }
    unbounded = tell(Mind(Config(seed=seed, use_cortex=False, auto_sleep=False)), world.statements())
    out["traces needed to hold everything"] = unbounded.memory.count
    out["facts"] = len(world.facts())
    return out


def interactions(seed: int) -> dict:
    """A conversation with no training phase: does every turn leave a mark?

    The teacher states facts, asks questions, sometimes gives the answer
    with the question, and otherwise reacts to what the model says: praise
    if right; rebuke, then the right answer, if wrong; the answer if the
    model said it did not know. Now and then somebody moves home without
    the model being told, so some of what it knows goes out of date.
    """
    world = World(chains=8, length=5, seed=seed)
    rng = np.random.default_rng(seed)
    mind = Mind(Config(seed=seed))
    facts = world.facts()
    met: set[int] = set()
    kinds = {"statement": 0, "question": 0, "question with answer": 0,
             "praise": 0, "rebuke": 0, "sleep": 0}
    turns = changed = 0
    curve = {}

    def turn(kind, act):
        nonlocal turns, changed
        before = mind.fingerprint()
        result = act()
        kinds[kind] += 1
        turns += 1
        changed += mind.fingerprint() != before
        return result

    moves = 0
    for step in range(1, 301):
        if rng.random() < 0.1:
            j = int(rng.integers(len(facts)))
            if facts[j][0].endswith("home"):
                others = [place for place in world.places if place != facts[j][1]]
                facts[j] = (facts[j][0], str(rng.choice(others)))
                moves += 1
        i = int(rng.integers(len(facts)))
        question, answer = facts[i]
        kind = str(rng.choice(["statement", "question", "question with answer"], p=[0.25, 0.5, 0.25]))
        if kind == "statement":
            turn(kind, lambda: mind.interact(f"{question} {answer} ."))
            met.add(i)
        elif kind == "question with answer":
            turn(kind, lambda: mind.interact(question, expect=answer))
            met.add(i)
        else:
            reply = turn(kind, lambda: mind.interact(question))
            if reply.abstained:
                turn("statement", lambda: mind.interact(f"{question} {answer} ."))
                met.add(i)
            elif reply.text == answer:
                turn("praise", lambda: mind.feedback(+1))
            else:
                turn("rebuke", lambda: mind.feedback(-1))
                turn("statement", lambda: mind.interact(f"{question} {answer} ."))
        if step % 100 == 0:
            curve[str(step)] = probe(mind, facts)["accuracy"]
    kinds["sleep"] = mind.sleeps
    return {
        "turns": turns,
        "turns that changed the model": changed,
        "journal entries": len(mind.journal),
        "kinds": kinds,
        "share of all facts known after n steps": curve,
        "recall of facts met in conversation": probe(mind, [facts[i] for i in sorted(met)])["accuracy"],
        "facts met": len(met),
        "facts": len(facts),
        "facts that changed behind its back": moves,
    }


def desire(seed: int) -> dict:
    """Half the facts are told. Then the model asks; the teacher answers."""
    world = World(chains=12, length=5, seed=seed)
    facts = world.facts()
    answers = dict(facts)
    rng = np.random.default_rng(seed)
    told = rng.choice(len(facts), size=len(facts) // 2, replace=False)
    out = {}
    for label, affect in (("curious (drives on)", True), ("drives off (asks at random)", False)):
        mind = Mind(Config(seed=seed, affect=affect, auto_sleep=False))
        for i in told:
            mind.interact(f"{facts[i][0]} {facts[i][1]} .")
        curve = {"0": probe(mind, facts, allow_system2=False)["accuracy"]}
        useful = 0
        for turn in range(1, 61):
            question = mind.ask()
            if question is None:
                break
            if question in answers:
                was_known = copy.deepcopy(mind).interact(question).text == answers[question]
                useful += not was_known
                mind.interact(question, expect=answers[question])
            if turn % 20 == 0:
                curve[str(turn)] = probe(mind, facts, allow_system2=False)["accuracy"]
        out[label] = {"facts known after n questions": curve,
                      "questions that taught something new (of 60)": useful}
    return out


EXPERIMENTS = {
    "small_data": small_data,
    "systems": systems,
    "context": context,
    "pain": pain,
    "parameters": parameters,
    "interactions": interactions,
    "desire": desire,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    args = parser.parse_args()

    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in args.only:
        started = time.time()
        runs = [EXPERIMENTS[name](seed) for seed in range(args.seeds)]
        results[name] = {
            "seeds": args.seeds,
            "seconds": round(time.time() - started, 1),
            "mean": mean([_numbers(run) for run in runs]),
            "first_seed": runs[0],
        }
        print(f"{name}: {results[name]['seconds']}s")
    path.write_text(json.dumps(results, indent=2) + "\n")

    try:
        from report import write_report
    except ImportError:
        return
    write_report(results, HERE / "RESULTS.md")


def _numbers(tree):
    """Keep only what can be averaged across seeds."""
    if isinstance(tree, dict):
        kept = {k: _numbers(v) for k, v in tree.items()}
        return {k: v for k, v in kept.items() if v is not None}
    if isinstance(tree, bool):
        return float(tree)
    if isinstance(tree, (int, float)):
        return tree
    return None


if __name__ == "__main__":
    main()
