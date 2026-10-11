import threading
import time
from pathlib import Path

from src.domain.models import EquipmentType
from src.pipeline import batch
from src.rendering import runner
from src.resolution.variants import ExerciseVariant
from tests.conftest import make_exercise


def variant(equipment=EquipmentType.BARBELL) -> ExerciseVariant:
    return ExerciseVariant(name="Romanian Deadlift", equipment=equipment,
                           label_es="Peso muerto rumano", note_es="-")


class FakeApp:
    def __init__(self, fail_for=()):
        self.calls, self.fail_for = [], fail_for
        self.lock = threading.Lock()

    def invoke(self, state):
        with self.lock:
            self.calls.append(state)
        variant = state["variant"]
        if variant.slug in self.fail_for:
            raise RuntimeError("boom")
        exercise = make_exercise(equipment=variant.equipment.value,
                                 animation_video="https://x/video.mp4", images=["https://x/a.png"])
        return {"structured_exercise": exercise, "validation_errors": []}


def test_every_variant_is_invoked_with_its_own_state():
    # Arrange
    app = FakeApp()
    variants = [variant(EquipmentType.BARBELL), variant(EquipmentType.DUMBBELL)]

    # Act
    results = batch.run_variants(variants, existing=set(), app=app)

    # Assert
    assert sorted(call["variant"].slug for call in app.calls) == ["romanian_deadlift_barbell", "romanian_deadlift_dumbbell"]
    assert [r.status for r in results] == ["created", "created"]


def test_existing_variant_is_reported_as_updated():
    # Arrange / Act
    results = batch.run_variants([variant()], existing={"romanian_deadlift_barbell"}, app=FakeApp())

    # Assert
    assert results[0].status == "updated"


def test_one_crash_does_not_stop_the_others():
    # Arrange
    app = FakeApp(fail_for={"romanian_deadlift_barbell"})

    # Act
    results = batch.run_variants([variant(EquipmentType.BARBELL), variant(EquipmentType.DUMBBELL)],
                                 existing=set(), app=app)

    # Assert
    assert [r.status for r in results] == ["failed", "created"]
    assert "boom" in results[0].warnings[0]


def test_missing_video_is_reported_as_no_media():
    # Arrange
    exercise = make_exercise()

    # Act
    result = batch.summarize(variant(), {"structured_exercise": exercise, "validation_errors": ["render failed"]}, False)

    # Assert
    assert result.status == "no_media"
    assert result.warnings == ["render failed"]


def test_database_failure_is_reported_as_failed():
    # Arrange / Act
    result = batch.summarize(variant(), {"structured_exercise": make_exercise(),
                                         "validation_errors": ["Database insertion failed: x"]}, False)

    # Assert
    assert result.status == "failed"


def test_render_slots_cap_parallel_manim_processes(monkeypatch, tmp_path):
    # Arrange: a fake Manim that records how many run at once
    active, peak, lock = [0], [0], threading.Lock()

    def fake_run(command, **kwargs):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        media_dir = Path(command[command.index("--media_dir") + 1])
        (media_dir / "videos").mkdir(parents=True, exist_ok=True)
        (media_dir / "videos" / "render.mp4").write_bytes(b"mp4")
        with lock:
            active[0] -= 1
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    spec = __import__("tests.test_render_node", fromlist=["golden_spec"]).golden_spec()

    # Act
    threads = [threading.Thread(target=runner.render_animation,
                                args=(spec, "t", "barbell", [], tmp_path / f"v{i}.mp4")) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    assert peak[0] <= runner.MAX_PARALLEL_RENDERS
    assert all((tmp_path / f"v{i}.mp4").exists() for i in range(5))
