from __future__ import annotations

import numpy as np
import pytest

from restoration.data import apply_joint_transform, build_pair_index


def test_pair_matching_uses_relative_stems_and_checks_2x(tmp_path):
    noisy = tmp_path / "NoisyLR"
    gt = tmp_path / "GT"
    (noisy / "source_a").mkdir(parents=True)
    (gt / "source_a").mkdir(parents=True)
    np.save(noisy / "source_a" / "same.npy", np.zeros((8, 9), dtype=np.float32))
    np.save(gt / "source_a" / "same.npy", np.zeros((16, 18), dtype=np.float32))
    np.save(noisy / "orphan.npy", np.zeros((8, 8), dtype=np.float32))
    pairs, missing_gt, missing_noisy = build_pair_index(noisy, gt, verify_shapes=True)
    assert [pair.key for pair in pairs] == ["source_a/same"]
    assert missing_gt == ["orphan.npy"]
    assert missing_noisy == []


def test_pair_shape_mismatch_is_rejected(tmp_path):
    noisy, gt = tmp_path / "NoisyLR", tmp_path / "GT"
    noisy.mkdir()
    gt.mkdir()
    np.save(noisy / "a.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(gt / "a.npy", np.zeros((15, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="violates 2x geometry"):
        build_pair_index(noisy, gt, verify_shapes=True)


@pytest.mark.parametrize("rotation_k", [0, 1, 2, 3])
@pytest.mark.parametrize("hflip,vflip", [(False, False), (True, False), (False, True), (True, True)])
def test_rotation_and_flip_preserve_pair_alignment(rotation_k, hflip, vflip):
    noisy = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    gt = np.repeat(np.repeat(noisy, 2, axis=0), 2, axis=1)
    noisy_aug, gt_aug = apply_joint_transform(noisy, gt, rotation_k, hflip, vflip)
    expected_gt = np.repeat(np.repeat(noisy_aug, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(gt_aug, expected_gt)

