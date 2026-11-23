import numpy as np

from conscious.memory import HEARD, EpisodicMemory

BOS = 0


def learn(memory, context, outcome, strength=1.0):
    base = memory.append_stream(context + [outcome])
    rev, truncated = memory.key(context)
    lesson = memory.learn_spans(rev, truncated, outcome, memory.match(rev))
    if lesson.needs_longer_context and memory.grow_context():
        rev, truncated = memory.key(context)
        lesson = memory.learn_spans(rev, truncated, outcome, memory.match(rev))
    slot, _ = memory.observe(rev, truncated, outcome, base + len(context), strength,
                             1.0, HEARD, 0, lesson.new_span)
    return slot


def recall(memory, context, vocab=40):
    rev, _ = memory.key(context)
    votes, support, window, _ = memory.vote(memory.match(rev), vocab, len(context))
    return (int(votes.argmax()) if support else None), window


def test_one_exposure_is_enough_to_recall():
    memory = EpisodicMemory()
    learn(memory, [BOS, 5, 6], 7)
    assert recall(memory, [BOS, 5, 6]) == (7, 1)


def test_repeating_a_moment_strengthens_one_trace():
    memory = EpisodicMemory()
    first = learn(memory, [BOS, 5, 6], 7)
    second = learn(memory, [BOS, 5, 6], 7)
    assert first == second
    assert memory.count == 1
    assert memory.strength[first] == 2.0


def test_a_trace_lengthens_its_span_only_when_it_is_confused():
    memory = EpisodicMemory()
    a = learn(memory, [BOS, 5, 6], 7)
    assert memory.span[a] == 1
    b = learn(memory, [BOS, 8, 6], 9)
    assert memory.span[a] == 2 and memory.span[b] == 2
    assert recall(memory, [BOS, 5, 6]) == (7, 2)
    assert recall(memory, [BOS, 8, 6]) == (9, 2)
    assert recall(memory, [BOS, 3, 6])[0] is None


def test_each_situation_gets_the_context_it_needs():
    memory = EpisodicMemory()
    learn(memory, [BOS, 1, 2, 3, 4], 20)
    learn(memory, [BOS, 9, 2, 3, 4], 21)
    learn(memory, [BOS, 5, 6], 22)
    learn(memory, [BOS, 7, 6], 23)
    learn(memory, [BOS, 8], 24)
    assert recall(memory, [BOS, 1, 2, 3, 4]) == (20, 4)
    assert recall(memory, [BOS, 7, 6]) == (23, 2)
    assert recall(memory, [BOS, 11, 12, 8]) == (24, 1)


def test_the_context_limit_grows_when_two_outcomes_cannot_be_told_apart():
    memory = EpisodicMemory(max_context=4)
    tail = [11, 12, 13, 14]
    learn(memory, [BOS, 1] + tail, 20)
    assert memory.L == 4
    learn(memory, [BOS, 2] + tail, 21)
    assert memory.L == 8
    assert recall(memory, [BOS, 1] + tail) == (20, 5)
    assert recall(memory, [BOS, 2] + tail) == (21, 5)


def test_the_context_limit_stops_at_its_ceiling():
    memory = EpisodicMemory(max_context=4, context_limit=4)
    tail = [11, 12, 13, 14]
    learn(memory, [BOS, 1] + tail, 20)
    learn(memory, [BOS, 2] + tail, 21)
    assert memory.L == 4


def test_destabilised_traces_weaken_and_then_die():
    memory = EpisodicMemory()
    slot = learn(memory, [BOS, 5, 6], 7)
    memory.destabilise(np.array([slot]), 0.5)
    assert memory.strength[slot] == 0.5
    memory.destabilise(np.array([slot]), 0.95)
    assert memory.count == 0
    assert recall(memory, [BOS, 5, 6])[0] is None


def test_eviction_takes_consolidated_traces_first():
    memory = EpisodicMemory()
    kept = learn(memory, [BOS, 1, 2], 3)
    gone = learn(memory, [BOS, 4, 5], 6)
    memory.mastery[gone] = 1.0
    count, lost = memory.evict(1)
    assert count == 1 and lost == 0.0
    assert memory.alive[kept] and not memory.alive[gone]


def test_compaction_keeps_the_survivors_reachable():
    memory = EpisodicMemory()
    slots = [learn(memory, [BOS, i, 30], i + 10) for i in range(1, 6)]
    memory.kill(np.array(slots[:2]))
    memory.compact()
    assert memory.n == 3
    assert recall(memory, [BOS, 4, 30])[0] == 14
    assert learn(memory, [BOS, 4, 30], 14) < 3
