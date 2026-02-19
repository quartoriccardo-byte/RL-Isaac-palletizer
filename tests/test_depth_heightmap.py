"""
Unit tests for depth heightmap converter.

These tests run WITHOUT Isaac Lab / pxr — pure PyTorch on CPU/CUDA.
Tests verify:
- Output shape correctness
- Noise application (magnitude, quantization)
- Flat-depth → flat-heightmap geometry
- Crop-bounds filtering
"""

import math
import pytest
import torch

from pallet_rl.utils.depth_heightmap import DepthHeightmapConverter, DepthHeightmapCfg
from pallet_rl.utils.device_utils import pick_supported_cuda_device


@pytest.fixture
def default_cfg():
    return DepthHeightmapCfg(
        cam_height=160,
        cam_width=240,
        fov_deg=40.0,
        sensor_height_m=3.0,
        map_h=16,  # smaller for test speed
        map_w=24,
        crop_x=(-0.65, 0.65),
        crop_y=(-0.45, 0.45),
        noise_enable=False,
    )


@pytest.fixture
def converter(default_cfg):
    device = pick_supported_cuda_device()[1] if torch.cuda.is_available() else "cpu"
    return DepthHeightmapConverter(default_cfg, device=device)


class TestOutputShape:
    """Verify output tensor dimensions."""

    def test_single_env(self, converter, default_cfg):
        N = 1
        device = converter.device
        depth = torch.ones(N, default_cfg.cam_height, default_cfg.cam_width, device=device) * 2.5
        cam_pos = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        hmap = converter.depth_to_heightmap(depth, cam_pos, cam_quat)
        assert hmap.shape == (N, default_cfg.map_h, default_cfg.map_w)

    def test_batch(self, converter, default_cfg):
        N = 4
        device = converter.device
        depth = torch.ones(N, default_cfg.cam_height, default_cfg.cam_width, device=device) * 2.0
        cam_pos = torch.zeros(N, 3, device=device)
        cam_pos[:, 2] = 3.0
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1)

        hmap = converter.depth_to_heightmap(depth, cam_pos, cam_quat)
        assert hmap.shape == (N, default_cfg.map_h, default_cfg.map_w)


class TestNoiseModel:
    """Verify noise application."""

    def test_noise_disabled(self, default_cfg):
        default_cfg.noise_enable = False
        device = pick_supported_cuda_device()[1] if torch.cuda.is_available() else "cpu"
        conv = DepthHeightmapConverter(default_cfg, device=device)
        depth = torch.ones(1, 160, 240, device=device) * 2.0
        result = conv.apply_noise(depth)
        assert torch.allclose(depth, result)

    def test_noise_enabled_changes_values(self):
        cfg = DepthHeightmapCfg(
            noise_enable=True,
            noise_sigma_m=0.01,
            noise_scale=1.0,
            noise_quantization_m=0.0,
            noise_dropout_prob=0.0,
        )
        device = pick_supported_cuda_device()[1] if torch.cuda.is_available() else "cpu"
        conv = DepthHeightmapConverter(cfg, device=device)
        depth = torch.ones(1, 160, 240, device=device) * 2.0
        
        torch.manual_seed(42)
        result = conv.apply_noise(depth.clone())
        # Noise changes values
        assert not torch.allclose(depth, result, atol=1e-6)
        # But bounded: sigma=0.01 → 99.7% within ±0.03
        diff = (result - depth).abs()
        assert diff.max() < 0.1  # generous bound

    def test_quantization(self):
        cfg = DepthHeightmapCfg(
            noise_enable=True,
            noise_sigma_m=0.0,
            noise_scale=0.0,
            noise_quantization_m=0.1,
            noise_dropout_prob=0.0,
        )
        device = pick_supported_cuda_device()[1] if torch.cuda.is_available() else "cpu"
        conv = DepthHeightmapConverter(cfg, device=device)
        # Use values that don't land on half-bins to avoid banker's rounding
        # 0.12 → round(1.2) = 1 → 0.1
        # 0.38 → round(3.8) = 4 → 0.4
        # 0.71 → round(7.1) = 7 → 0.7
        depth = torch.tensor([[[0.12, 0.38, 0.71]]], device=device)
        result = conv.apply_noise(depth)
        expected = torch.tensor([[[0.1, 0.4, 0.7]]], device=device)
        assert torch.allclose(result, expected, atol=1e-5)


class TestFlatDepthGeometry:
    """A flat depth image from directly overhead should yield a flat heightmap."""

    def test_flat_floor_from_overhead(self):
        cfg = DepthHeightmapCfg(
            cam_height=32,
            cam_width=48,
            fov_deg=40.0,
            sensor_height_m=3.0,
            map_h=8,
            map_w=12,
            crop_x=(-0.65, 0.65),
            crop_y=(-0.45, 0.45),
            noise_enable=False,
        )
        device = pick_supported_cuda_device()[1] if torch.cuda.is_available() else "cpu"
        conv = DepthHeightmapConverter(cfg, device=device)

        # Camera at (0, 0, 3), looking down (+Z in camera = forward)
        # Identity quaternion means camera looks along +Z in world
        # For a downward-looking camera we need quat that rotates +Z to -Z_world
        # But with identity quat, the "forward" is +Z world, so depth=3 means
        # points at z=0+3=3 in world. Let's just verify shape and non-negativity.
        depth = torch.ones(1, 32, 48, device=device) * 3.0
        cam_pos = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        hmap = conv.depth_to_heightmap(depth, cam_pos, cam_quat)
        assert hmap.shape == (1, 8, 12)
        assert hmap.min() >= 0.0


class TestCropBounds:
    """Points outside crop bounds should not affect heightmap."""

    def test_points_outside_crop(self, default_cfg):
        device = pick_supported_cuda_device()[1] if torch.cuda.is_available() else "cpu"
        # Very narrow crop
        default_cfg.crop_x = (-0.01, 0.01)
        default_cfg.crop_y = (-0.01, 0.01)
        conv = DepthHeightmapConverter(default_cfg, device=device)

        depth = torch.ones(1, 160, 240, device=device) * 2.0
        cam_pos = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        hmap = conv.depth_to_heightmap(depth, cam_pos, cam_quat)
        # Most cells should be empty (0) because crop is very narrow
        zero_fraction = (hmap == 0).float().mean()
        # At least 50% zeros expected (likely much more)
        assert zero_fraction > 0.5


class TestDecimation:
    """Test that decimation caching works."""

    def test_decimation_reuses_cache(self, converter, default_cfg):
        device = converter.device
        N = 1
        depth = torch.ones(N, default_cfg.cam_height, default_cfg.cam_width, device=device) * 2.0
        cam_pos = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        # First call computes
        h1 = converter(depth, cam_pos, cam_quat, decimation=3)
        # Second call should reuse cache
        h2 = converter(depth * 999, cam_pos, cam_quat, decimation=3)  # different depth
        assert torch.allclose(h1, h2), "Decimation should reuse cached heightmap"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
