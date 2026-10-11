import math

from src.rendering.layout import Box, Capsule, capsule_box_penetration, overlap_score, place_label


def test_capsule_touching_box_has_penetration():
    # Arrange
    capsule = Capsule((0.0, 0.0), (2.0, 0.0), 0.3)
    box = Box(1.0, 0.3, 0.4, 0.2)

    # Act / Assert
    assert capsule_box_penetration(capsule, box) > 0


def test_distant_capsule_has_no_penetration():
    # Arrange
    capsule = Capsule((0.0, 0.0), (2.0, 0.0), 0.3)
    box = Box(1.0, 2.0, 0.4, 0.2)

    # Act / Assert
    assert capsule_box_penetration(capsule, box) == 0


def test_label_avoids_the_body():
    # Arrange: a vertical limb right where the preferred direction points
    limb = Capsule((3.6, -2.0), (3.6, 2.0), 0.35)

    # Act
    _, box = place_label((3.0, 0.0), 0.8, 0.3, [limb], [], prefer_angle=0.0)

    # Assert
    assert overlap_score(box, [limb], []) == 0


def test_label_avoids_labels_already_placed():
    # Arrange
    first = Box(3.6, 0.0, 0.8, 0.3)

    # Act
    _, box = place_label((3.0, 0.0), 0.8, 0.3, [], [first], prefer_angle=0.0)

    # Assert
    assert not box.intersects(first)


def test_previous_position_is_kept_while_still_free():
    # Arrange
    key, _ = place_label((3.0, 0.0), 0.8, 0.3, [], [], prefer_angle=math.pi / 2)

    # Act
    same_key, _ = place_label((3.05, 0.0), 0.8, 0.3, [], [], prefer_angle=0.0, previous=key)

    # Assert
    assert same_key == key


def test_boxed_in_label_still_gets_a_position():
    # Arrange: obstacles all around the anchor
    ring = [Capsule((3.0 + 1.5 * math.cos(a), 1.5 * math.sin(a)), (3.0, 0.0), 1.6)
            for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]

    # Act
    key, box = place_label((3.0, 0.0), 0.8, 0.3, ring, [])

    # Assert
    assert key is not None and box is not None
