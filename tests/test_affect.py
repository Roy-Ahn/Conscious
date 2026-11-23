import pytest

from conscious.affect import DriveSystem, PainPoint, PainSystem


def test_pain_rises_when_inflicted_and_fades_with_time():
    point = PainPoint("surprise", decay=0.2)
    felt = point.inflict(0.6)
    assert felt == pytest.approx(0.6)
    assert point.level == pytest.approx(0.6)
    point.tick()
    assert point.level == pytest.approx(0.48)


def test_pain_below_threshold_is_not_felt():
    point = PainPoint("surprise", threshold=0.1)
    assert point.inflict(0.05) == 0.0
    assert point.level == 0.0


def test_pain_level_never_exceeds_one():
    point = PainPoint("rebuke")
    for _ in range(50):
        point.inflict(1.0)
    assert 0.99 < point.level <= 1.0


def test_strong_hits_sensitise_and_mild_hits_habituate():
    hurt, bored = PainPoint("a"), PainPoint("b")
    for _ in range(5):
        hurt.inflict(0.9)
        bored.inflict(0.1)
    assert hurt.gain > 1.0
    assert bored.gain < 1.0
    assert hurt.inflict(0.3) > bored.inflict(0.3)


def test_sensitivity_drifts_back_to_normal():
    point = PainPoint("a")
    for _ in range(5):
        point.inflict(0.9)
    raised = point.gain
    for _ in range(200):
        point.tick()
    assert 1.0 <= point.gain < raised
    assert point.gain == pytest.approx(1.0, abs=0.01)


def test_total_pain_is_dominated_by_the_worst_point():
    pain = PainSystem()
    pain.inflict("surprise", 0.5)
    pain.inflict("rebuke", 0.5)
    assert pain.total() == pytest.approx(0.75)
    assert pain.total() >= max(p.level for p in pain.points.values())


def test_pain_points_can_be_added_and_removed_while_running():
    pain = PainSystem()
    before = set(pain.points)
    pain.add_point("loneliness", decay=0.05, gain=2.0)
    assert set(pain.points) == before | {"loneliness"}
    assert pain.inflict("loneliness", 0.2) == pytest.approx(0.4)
    assert pain.total() == pytest.approx(0.4)
    pain.remove_point("loneliness")
    assert pain.total() == 0.0


def test_disabled_pain_feels_nothing():
    pain = PainSystem(enabled=False)
    assert pain.inflict("rebuke", 1.0) == 0.0
    assert pain.total() == 0.0


def test_desire_builds_until_satisfied_and_discharge_is_pleasure():
    pain, drives = PainSystem(), DriveSystem()
    for _ in range(10):
        drives.tick(pain)
    tension = drives.tension("curiosity")
    assert tension == pytest.approx(0.4)
    pleasure = drives.satisfy("curiosity", 0.25)
    assert pleasure == pytest.approx(0.25)
    assert drives.tension("curiosity") == pytest.approx(0.15)
    assert drives.pleasure > 0.0


def test_a_drive_cannot_give_more_pleasure_than_its_tension():
    drives = DriveSystem()
    drives.drives["bond"].tension = 0.1
    assert drives.satisfy("bond", 1.0) == pytest.approx(0.1)


def test_the_strongest_drive_dominates():
    drives = DriveSystem()
    drives.drives["mastery"].tension = 0.7
    drives.drives["curiosity"].tension = 0.3
    assert drives.dominant() == "mastery"


def test_relief_drive_tracks_pain_and_its_easing_is_pleasant():
    pain, drives = PainSystem(), DriveSystem()
    pain.inflict("rebuke", 0.8)
    drives.tick(pain)
    assert drives.tension("relief") == pytest.approx(pain.total())
    pain.soothe(1.0)
    drives.tick(pain)
    assert drives.tension("relief") == 0.0
    assert drives.pleasure > 0.3


def test_desire_left_unmet_becomes_pain():
    pain, drives = PainSystem(), DriveSystem()
    for _ in range(60):
        drives.tick(pain)
    assert drives.tension("curiosity") == 1.0
    assert pain.level("frustration") > 0.0
