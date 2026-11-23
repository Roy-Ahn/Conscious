"""One section per property the model is required to have."""

import copy

import numpy as np
import pytest

from conscious import Config, Mind
from conscious.worlds import World, depth_rules, with_noise

FAMILY = ["ann parent bob .", "bob parent carl .", "carl parent dan .", "dan parent eve .",
          "ann home paris .", "bob home rome .", "carl home oslo ."]


def taught(statements=FAMILY, **config) -> Mind:
    mind = Mind(Config(**config))
    for statement in statements:
        mind.interact(statement)
    return mind


def calm(mind: Mind) -> Mind:
    mind.pain.soothe(1.0)
    return mind


# ------------------------------------------------- 1. System 1 and System 2

def test_a_known_fact_is_answered_by_system_1():
    reply = calm(taught()).interact("ann parent")
    assert reply.text == "bob"
    assert reply.system == 1
    assert reply.path == []


def test_a_new_composition_is_answered_by_system_2():
    reply = calm(taught()).interact("ann parent parent")
    assert reply.text == "carl"
    assert reply.system == 2
    assert any(step.startswith("reduce: ann parent -> bob") for step in reply.path)


def test_system_2_chains_across_relations_and_depths():
    mind = calm(taught())
    assert mind.interact("ann parent home").text == "rome"
    assert mind.interact("ann parent parent parent").text == "dan"
    assert mind.interact("ann parent parent home").text == "oslo"


def test_without_system_2_a_composition_cannot_be_answered():
    reply = calm(taught(allow_system2=False)).interact("ann parent parent")
    assert reply.text != "carl"
    assert reply.system == 1


def test_system_2_costs_more_steps_than_system_1():
    mind = calm(taught())
    easy = mind.interact("ann parent")
    hard = mind.interact("bob parent parent")
    assert hard.steps > easy.steps


def test_what_system_2_works_out_becomes_a_system_1_habit():
    mind = calm(taught())
    first = mind.interact("ann parent parent")
    second = mind.interact("ann parent parent")
    assert first.system == 2 and second.system == 1
    assert second.text == first.text == "carl"
    assert second.steps < first.steps


def test_the_same_question_is_checked_by_system_2_when_the_mind_is_in_pain():
    mind = calm(taught())
    assert mind.interact("ann parent").system == 1
    mind.pain.inflict("rebuke", 0.9)
    reply = mind.interact("ann parent")
    assert reply.text == "bob"
    assert reply.system == 2
    assert reply.path == ["recollect"]


def test_an_exhausted_mind_cannot_deliberate():
    mind = calm(taught(auto_sleep=False))
    mind.energy = 0.0
    reply = mind.interact("ann parent parent")
    assert reply.system == 1
    assert reply.text != "carl"


def test_system_2_can_be_forced_on():
    reply = calm(taught(force_system2=True)).interact("ann parent")
    assert reply.text == "bob" and reply.system == 2


def test_moments_of_deliberation_are_recorded():
    mind = calm(taught())
    mind.interact("ann parent")
    assert mind.moments == []
    mind.interact("ann parent parent")
    assert mind.moments[-1]["about"] == "ann parent parent"
    assert mind.moments[-1]["answer"] == "carl"


# ------------------------------------------------------------ 2. Pain points

def test_being_contradicted_hurts():
    mind = taught(["sky colour blue ."] * 3)
    calm(mind)
    mind.interact("sky colour green .")
    assert mind.pain.level("surprise") > 0.5


def test_being_confidently_wrong_hurts_more_than_being_unsure():
    sure = calm(taught(["sky colour blue ."] * 3))
    unsure = calm(taught(["sky colour blue .", "sky colour grey .", "sky colour black ."] * 2))
    sure.interact("sky colour green .")
    unsure.interact("sky colour green .")
    assert sure.pain.level("surprise") > unsure.pain.level("surprise")


def test_hearing_what_was_expected_does_not_hurt():
    mind = calm(taught(["sky colour blue ."] * 3))
    mind.interact("sky colour blue .")
    assert mind.pain.level("surprise") == 0.0


def test_a_rebuked_answer_is_not_repeated():
    mind = calm(taught())
    assert mind.interact("ann parent").text == "bob"
    mind.feedback(-1)
    assert mind.pain.level("rebuke") > 0.5
    reply = mind.interact("ann parent")
    assert reply.text != "bob"
    assert reply.abstained


def test_after_a_rebuke_the_corrected_answer_is_given_even_while_it_still_hurts():
    mind = calm(taught())
    assert mind.interact("bob home").text == "rome"
    mind.feedback(-1)
    mind.interact("bob home", expect="lima")
    assert mind.pain.level("rebuke") > 0.5
    assert mind.interact("bob home").text == "lima"


def test_without_affect_a_rebuke_changes_nothing():
    mind = taught(affect=False)
    assert mind.interact("ann parent").text == "bob"
    mind.feedback(-1)
    assert mind.pain.total() == 0.0
    assert mind.interact("ann parent").text == "bob"


def test_saying_a_forbidden_thing_again_causes_guilt():
    mind = calm(taught())
    mind.interact("ann parent")
    mind.feedback(-0.4)
    assert mind.interact("ann parent").text == "bob"
    assert mind.pain.level("guilt") > 0.0


def test_praise_lifts_a_prohibition_and_soothes():
    mind = calm(taught())
    mind.interact("ann parent")
    mind.feedback(-0.4)
    mind.interact("ann parent")
    hurting = mind.pain.total()
    mind.feedback(+1)
    assert mind.pain.total() < hurting
    assert mind.forbidden[tuple([mind.vocab.bos] + mind.vocab.encode("ann parent"))] == {}


def test_a_pain_point_added_at_run_time_changes_behaviour():
    mind = calm(taught())
    assert mind.interact("ann parent").system == 1
    bar = mind._confidence_needed()
    mind.pain.add_point("loneliness", decay=0.01)
    mind.pain.inflict("loneliness", 0.9)
    assert mind._confidence_needed() > bar
    assert mind.interact("ann parent").system == 2


def test_pain_makes_the_mind_change_its_beliefs_faster():
    def exposures_needed(affect: bool) -> int:
        mind = taught(["sky colour blue ."] * 10, affect=affect, auto_sleep=False)
        for n in range(1, 40):
            mind.interact("sky colour green .")
            if copy.deepcopy(mind).interact("sky colour").text == "green":
                return n
        return 40

    assert exposures_needed(True) <= 3
    assert exposures_needed(False) >= 10


def test_one_contradiction_shakes_a_well_established_belief_but_does_not_overturn_it():
    mind = calm(taught(["sky colour blue ."] * 10, auto_sleep=False))
    before = copy.deepcopy(mind).interact("sky colour")
    mind.interact("sky colour green .")
    after = mind.interact("sky colour")
    assert before.text == after.text == "blue"
    assert before.system == 1 and after.system == 2
    assert after.confidence < before.confidence


def test_rebukes_for_wrong_answers_leave_lasting_caution():
    mind = calm(taught())
    start = mind.caution
    for question in ("ann parent", "bob parent", "carl parent"):
        mind.interact(question)
        mind.feedback(-1)
    calm(mind)
    assert mind.caution > start
    assert mind._abstain_below() > start


def test_a_rebuke_for_silence_makes_the_mind_bolder_and_more_curious():
    mind = calm(taught())
    start = mind.caution
    assert mind.interact("zed parent").abstained
    mind.interact("yan parent")
    wanting = mind.drives.tension("curiosity")
    mind.feedback(-1)
    assert mind.caution < start
    assert mind.drives.tension("curiosity") > wanting
    assert mind.ask() == "yan parent"


def test_praise_lowers_caution():
    mind = calm(taught())
    start = mind.caution
    mind.interact("ann parent")
    mind.feedback(+1)
    assert mind.caution < start


@pytest.mark.parametrize("affect", [True, False])
def test_what_is_learned_in_pain_is_kept_when_memory_is_cut(affect):
    world = World(chains=4, length=4, seed=3)
    mind = Mind(Config(auto_sleep=False, affect=affect))
    mind.pain.inflict("rebuke", 0.9)
    mind.interact("fire touch burn .")
    for statement in world.statements():
        calm(mind).interact(statement)
    calm(mind).memory.evict(mind.memory.count - 8)
    kept = mind.interact("fire touch").text == "burn"
    assert kept == affect


def test_pain_fades_and_sleep_soothes():
    mind = taught()
    mind.pain.inflict("rebuke", 0.9)
    before = mind.pain.total()
    mind.interact("ann parent")
    assert mind.pain.total() < before
    mind.sleep()
    assert mind.pain.total() < 0.5 * before


def test_running_out_of_energy_hurts_and_leads_to_sleep():
    mind = calm(taught(energy_system2=0.2, energy_recover=0.0))
    slept = [mind.interact(q).slept for q in
             ["ann parent parent", "bob parent parent", "ann parent home",
              "bob parent home", "carl parent parent", "ann parent parent parent"]]
    assert any(slept)
    assert mind.sleeps >= 1
    assert mind.energy > 0.5


# ---------------------------------------------------- 3. Flexible parameters

def test_new_words_enlarge_the_model():
    mind = Mind()
    before = mind.size()
    mind.interact("zebra eats grass .")
    after = mind.size()
    assert after["vocab"] == before["vocab"] + 3
    assert after["cortex_weights"] > before["cortex_weights"]
    assert after["traces"] > before["traces"]
    assert mind.cortex.params["E"].shape[0] == len(mind.vocab)


def test_weights_change_while_the_model_is_being_used():
    mind = taught()
    before = {k: v.copy() for k, v in mind.cortex.params.items()}
    mind.interact("eve home cairo .")
    assert any(not np.array_equal(before[k][: len(before[k])],
                                  mind.cortex.params[k][: len(before[k])])
               for k in ("W1", "b1"))


def test_pain_raises_plasticity():
    def change(hurt: bool) -> float:
        mind = calm(taught())
        if hurt:
            mind.pain.add_point("ache", decay=0.0)
            mind.pain.inflict("ache", 0.9)
        before = mind.cortex.params["W1"].copy()
        mind.interact("ann parent bob .")
        return float(np.abs(mind.cortex.params["W1"] - before).sum())

    assert change(True) > 1.3 * change(False)


def test_the_cortex_grows_when_it_cannot_hold_what_was_lived():
    world = World(chains=12, length=5, seed=2)
    growing = taught(world.statements(), hidden=4, auto_sleep=False)
    fixed = taught(world.statements(), hidden=4, auto_sleep=False, grow_cortex=False)
    report = growing.sleep()
    stuck = fixed.sleep()
    assert report["grown"] > 0
    assert growing.cortex.hidden == 4 + report["grown"]
    assert fixed.cortex.hidden == 4
    assert report["residual_after"] < 0.5 * stuck["residual_after"]


def test_knowledge_survives_in_the_cortex_after_its_memories_are_gone():
    world = World(chains=10, length=5, seed=4)
    mind = taught(world.statements(), auto_sleep=False)
    mind.sleep()
    mind.memory.kill(np.flatnonzero(mind.memory.alive[: mind.memory.n]))
    assert mind.memory.count == 0
    replies = [mind.interact(q) for q, _ in world.facts()]
    assert all(r.text == a for r, (_, a) in zip(replies, world.facts()))
    assert all(r.sources[0] == "cortex" for r in replies)


def test_dreaming_keeps_old_knowledge_while_new_knowledge_is_consolidated():
    old, new = World(chains=8, length=5, seed=5), World(chains=8, length=5, seed=6)

    def recall_of_old(dream: bool) -> float:
        mind = taught(old.statements(), auto_sleep=False)
        mind.sleep()
        mind.memory.kill(np.flatnonzero(mind.memory.alive[: mind.memory.n]))
        if not dream:
            mind.familiar.clear()
        for statement in new.statements():
            mind.interact(statement)
        report = mind.sleep()
        assert (report["dreams"] > 0) == dream
        mind.cfg.use_memory = False
        mind.familiar.update(tuple([mind.vocab.bos] + mind.vocab.encode(q)) for q, _ in old.facts())
        return float(np.mean([mind.interact(q).text == a for q, a in old.facts()]))

    assert recall_of_old(dream=True) > 0.9
    assert recall_of_old(dream=False) < 0.6


def test_memory_shrinks_when_it_is_over_capacity():
    world = World(chains=12, length=5, seed=2)
    mind = taught(world.statements(), capacity=120)
    assert mind.memory.count <= 120
    assert mind.sleeps >= 1
    sizes = [h["traces"] for h in mind.history]
    assert max(sizes) > sizes[-1] or any(b < a for a, b in zip(sizes, sizes[1:]))


def test_size_history_is_recorded_for_every_moment():
    mind = taught()
    assert len(mind.history) == len(FAMILY)
    assert mind.history[-1]["vocab"] == len(mind.vocab)


# --------------------------------- 4. Every interaction becomes training data

INTERACTIONS = {
    "a statement": lambda m: m.interact("eve home cairo ."),
    "a question it knows": lambda m: m.interact("ann parent"),
    "a question it works out": lambda m: m.interact("ann parent parent"),
    "a question it cannot answer": lambda m: m.interact("zed parent"),
    "a question with the true answer": lambda m: m.interact("eve parent", expect="fay"),
    "praise": lambda m: (m.interact("ann parent"), m.feedback(+1)),
    "rebuke": lambda m: (m.interact("ann parent"), m.feedback(-1)),
}


@pytest.mark.parametrize("kind", INTERACTIONS)
def test_every_kind_of_interaction_changes_the_model(kind):
    mind = calm(taught())
    before, entries, stream = mind.fingerprint(), len(mind.journal), len(mind.memory.stream)
    INTERACTIONS[kind](mind)
    assert mind.fingerprint() != before
    assert len(mind.journal) > entries
    assert len(mind.memory.stream) > stream


def test_answering_a_known_question_strengthens_the_memory_used():
    mind = calm(taught())
    key = (mind.memory.key([mind.vocab.bos] + mind.vocab.encode("ann parent"))[0].tobytes(),
           mind.vocab.ids["bob"])
    before = mind.memory.strength[mind.memory.index[key]]
    mind.interact("ann parent")
    assert mind.memory.strength[mind.memory.index[key]] > before


def test_an_unanswered_question_stays_open_and_is_asked_back():
    mind = calm(taught())
    assert mind.interact("zed parent").abstained
    assert mind.ask() == "zed parent"


def test_several_sentences_in_one_text_are_learned_separately():
    mind = Mind()
    reply = mind.interact("ann parent bob . bob parent carl . ann parent parent")
    assert reply.text == "carl"
    assert calm(mind).interact("bob parent").text == "carl"


def test_it_learns_from_conversation_alone():
    mind = Mind()
    first = mind.interact("moon colour", expect="silver")
    second = mind.interact("moon colour")
    assert first.correct is False
    assert second.text == "silver"


def test_a_saved_mind_remembers_what_it_knew_and_how_it_felt(tmp_path):
    mind = taught()
    mind.pain.inflict("rebuke", 0.7)
    path = tmp_path / "saved.mind"
    mind.save(str(path))
    woken = Mind.load(str(path))
    assert woken.fingerprint() == mind.fingerprint()
    assert woken.pain.level("rebuke") == mind.pain.level("rebuke")
    assert woken.interact("ann parent parent").text == "carl"


# ------------------------------------------- 5. Learning from small datasets

def test_every_fact_is_known_after_being_told_once():
    world = World(chains=8, length=5, seed=1)
    mind = taught(world.statements())
    assert all(mind.interact(q).text == a for q, a in world.facts())


def test_a_network_alone_does_not_learn_from_one_telling():
    world = World(chains=8, length=5, seed=1)
    mind = taught(world.statements(), use_memory=False, allow_system2=False, awake_rate=1.0)
    right = sum(mind.interact(q).text == a for q, a in world.facts())
    assert right < 0.3 * len(world.facts())


def test_new_knowledge_does_not_erase_old_knowledge():
    old, new = World(chains=6, length=5, seed=1), World(chains=6, length=5, seed=2)
    mind = taught(old.statements())
    for statement in new.statements():
        mind.interact(statement)
    assert all(mind.interact(q).text == a for q, a in old.facts())
    assert all(mind.interact(q).text == a for q, a in new.facts())


def test_it_knows_things_it_was_never_told():
    world = World(chains=8, length=5, seed=1)
    mind = taught(world.statements())
    questions = world.hops(2) + world.hops(3)
    assert len(questions) > 40
    assert all(mind.interact(q).text == a for q, a in questions)


def test_it_says_it_does_not_know_rather_than_guess():
    world = World(chains=8, length=5, seed=1)
    mind = taught(world.statements())
    assert mind.interact("nobody parent").abstained
    assert mind.interact("nobody home").text == "<?>"


# ----------------------------------------------- 6. Flexible context window

def test_the_context_read_matches_what_each_task_needs():
    rules, noise = depth_rules([1, 2, 3, 5], per_depth=6, seed=0)
    rng = np.random.default_rng(0)
    mind = Mind(Config(auto_sleep=False))
    for _ in range(3):
        for _, words, answer in rules:
            mind.interact(with_noise(words, noise, rng) + f" {answer} .")
    for depth, words, answer in rules:
        reply = mind.interact(with_noise(words, noise, rng))
        assert reply.text == answer
        assert reply.window == depth


def test_no_single_fixed_window_handles_every_task():
    rules, noise = depth_rules([1, 2, 3, 5], per_depth=6, seed=0)

    def accuracy(window):
        rng = np.random.default_rng(0)
        mind = Mind(Config(auto_sleep=False, fixed_window=window))
        for _ in range(3):
            for _, words, answer in rules:
                mind.interact(with_noise(words, noise, rng) + f" {answer} .")
        return np.mean([mind.interact(with_noise(words, noise, rng)).text == answer
                        for _, words, answer in rules])

    assert accuracy(None) == 1.0
    assert all(accuracy(window) < 0.8 for window in (1, 2, 3, 5, 8))


def test_the_longest_context_grows_when_a_task_needs_more():
    tail = "one two three four five six"
    mind = Mind(Config(max_context=4, auto_sleep=False))
    assert mind.memory.L == 4
    mind.interact(f"red {tail} stop .")
    mind.interact(f"green {tail} go .")
    assert mind.memory.L == 8
    assert mind.interact(f"red {tail}").text == "stop"
    assert mind.interact(f"green {tail}").text == "go"
    assert mind.interact(f"green {tail}").window == 7


def test_the_window_is_small_for_a_local_task_even_in_a_long_sentence():
    mind = Mind(Config(auto_sleep=False))
    for filler in ("a b c d e", "f g h i j", "k l m n o"):
        mind.interact(f"{filler} knock knock who .")
    reply = mind.interact("p q r s t u knock knock")
    assert reply.text == "who"
    assert reply.window <= 2


# ------------------------------------------------------------------ Desire

def test_curiosity_asks_about_what_is_not_known():
    mind = taught()
    for _ in range(6):
        question = mind.ask()
        assert question is not None
        assert copy.deepcopy(mind).interact(question).abstained


def test_caprice_takes_over_when_the_wish_to_choose_is_strongest():
    mind = taught()
    for drive in mind.drives.drives.values():
        drive.tension = 0.0
    mind.drives.drives["autonomy"].tension = 0.9
    mind.ask()
    assert mind.drives.tension("autonomy") == 0.0


def test_learning_something_new_satisfies_curiosity():
    mind = taught()
    for _ in range(10):
        mind.drives.tick(mind.pain)
    before = mind.drives.tension("curiosity")
    mind.interact("whale home ocean .")
    assert mind.drives.tension("curiosity") < before
    assert mind.drives.pleasure > 0


def test_the_report_names_what_hurts_and_what_is_wanted():
    mind = calm(taught())
    assert "Nothing hurts" in mind.report()["feeling"]
    mind.interact("ann parent")
    mind.feedback(-1)
    report = mind.report()
    assert "rebuke" in report["feeling"]
    assert set(report["pain"]) >= {"surprise", "rebuke", "frustration", "exhaustion", "loss", "guilt"}
    assert set(report["drives"]) == {"curiosity", "mastery", "relief", "bond", "autonomy"}
