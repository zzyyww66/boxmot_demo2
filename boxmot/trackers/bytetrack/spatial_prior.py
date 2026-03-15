from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialPriorStats:
    """Summary statistics for the learned spatial prior."""

    support_total: float
    birth_total: float


class SpatialPriorField:
    """Learn low-resolution spatial priors from confirmed tracking events."""

    _DECAY_FLUSH_THRESHOLD = np.float32(1e-3)

    def __init__(
        self,
        grid_w: int = 48,
        grid_h: int = 27,
        sigma: float = 1.5,
        decay: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 9.0,
    ) -> None:
        self.grid_w = max(4, int(grid_w))
        self.grid_h = max(4, int(grid_h))
        self.sigma = max(0.5, float(sigma))
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.prior_alpha = float(max(0.0, prior_alpha))
        self.prior_beta = float(max(1e-6, prior_beta))
        self.img_w = 0
        self.img_h = 0
        self._initialized = False
        self._decay_scale = np.float32(1.0)
        self._grid_x = np.arange(self.grid_w, dtype=np.float32)
        self._grid_y = np.arange(self.grid_h, dtype=np.float32)
        self._gaussian_scale = np.float32(-0.5 / (self.sigma ** 2))
        self._radius = max(1, int(np.ceil(self.sigma * 3.0)))
        self._offsets = np.arange(-self._radius, self._radius + 1, dtype=np.int32)
        self._grid_x_scale = np.float32(0.0)
        self._grid_y_scale = np.float32(0.0)
        self._allocate_maps()

    def _allocate_maps(self) -> None:
        shape = (self.grid_h, self.grid_w)
        self.support_count = np.zeros(shape, dtype=np.float32)
        self.birth_count = np.zeros(shape, dtype=np.float32)

    def configure_image(self, img_w: int, img_h: int) -> None:
        """Set the image shape used to project points into the field grid."""
        img_w = int(img_w)
        img_h = int(img_h)
        if img_w <= 0 or img_h <= 0:
            return
        if self._initialized and self.img_w == img_w and self.img_h == img_h:
            return
        self.img_w = img_w
        self.img_h = img_h
        self._grid_x_scale = np.float32((self.grid_w - 1) / float(self.img_w - 1))
        self._grid_y_scale = np.float32((self.grid_h - 1) / float(self.img_h - 1))
        self._initialized = True

    def step(self) -> None:
        """Apply a slow temporal decay to all maps once per frame."""
        if not self._initialized or self.decay >= 1.0:
            return
        self._decay_scale *= np.float32(self.decay)
        if self._decay_scale < self._DECAY_FLUSH_THRESHOLD:
            self._materialize_decay()

    def add_support(self, point: np.ndarray | tuple[float, float], weight: float = 1.0) -> None:
        self._splat(self.support_count, point, weight)

    def add_birth(self, point: np.ndarray | tuple[float, float], weight: float = 1.0) -> None:
        self._splat(self.birth_count, point, weight)

    def add_support_batch(self, points: list[np.ndarray], weight: float = 1.0) -> None:
        self._splat_many(self.support_count, points, weight)

    def add_birth_batch(self, points: list[np.ndarray], weight: float = 1.0) -> None:
        self._splat_many(self.birth_count, points, weight)

    def stats(self) -> SpatialPriorStats:
        """Return aggregate counts for debugging and tests."""
        return SpatialPriorStats(
            support_total=float(self.support_count.sum(dtype=np.float64) * float(self._decay_scale)),
            birth_total=float(self.birth_count.sum(dtype=np.float64) * float(self._decay_scale)),
        )

    def get_probability_maps(self) -> dict[str, np.ndarray]:
        """Return normalized walkability and birth probability maps."""
        if not self._initialized:
            shape = (self.grid_h, self.grid_w)
            zeros = np.zeros(shape, dtype=np.float32)
            return {
                "walkable": zeros.copy(),
                "birth": zeros.copy(),
                "confidence": zeros.copy(),
            }

        support_count = self._scaled_field(self.support_count)
        birth_count = self._scaled_field(self.birth_count)
        denom = support_count + self.prior_alpha + self.prior_beta
        walk_scale = max(float(support_count.max()), 1.0)
        confidence = support_count + birth_count
        return {
            "walkable": np.clip(support_count / walk_scale, 0.0, 1.0),
            "birth": np.clip((birth_count + self.prior_alpha) / denom, 0.0, 1.0),
            "confidence": confidence.astype(np.float32),
        }

    def get_metric_maps(self) -> dict[str, np.ndarray]:
        """Return local density and conditional-ratio maps used for region masks."""
        shape = (self.grid_h, self.grid_w)
        if not self._initialized:
            zeros = np.zeros(shape, dtype=np.float32)
            return {
                "support_density": zeros.copy(),
                "birth_density": zeros.copy(),
                "birth_ratio": zeros.copy(),
            }

        support_density = self._local_sum(self._scaled_field(self.support_count))
        birth_density = self._local_sum(self._scaled_field(self.birth_count))
        denom = np.maximum(support_density, 0.25).astype(np.float32)
        return {
            "support_density": support_density,
            "birth_density": birth_density,
            "birth_ratio": np.clip(birth_density / denom, 0.0, 1.0),
        }

    def point_to_index(self, point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        """Project an image point to the nearest grid-cell index."""
        gx, gy = self._point_to_grid(point)
        ix = int(np.clip(round(gx), 0, self.grid_w - 1))
        iy = int(np.clip(round(gy), 0, self.grid_h - 1))
        return ix, iy

    def _local_sum(self, field: np.ndarray) -> np.ndarray:
        """Compute a 3x3 local sum without requiring scipy."""
        padded = np.pad(field, 1, mode="constant")
        h, w = field.shape
        return (
            padded[0:h, 0:w]
            + padded[0:h, 1 : w + 1]
            + padded[0:h, 2 : w + 2]
            + padded[1 : h + 1, 0:w]
            + padded[1 : h + 1, 1 : w + 1]
            + padded[1 : h + 1, 2 : w + 2]
            + padded[2 : h + 2, 0:w]
            + padded[2 : h + 2, 1 : w + 1]
            + padded[2 : h + 2, 2 : w + 2]
        ).astype(np.float32, copy=False)

    def _point_to_grid(self, point: np.ndarray | tuple[float, float]) -> tuple[float, float]:
        px = float(point[0])
        py = float(point[1])
        if not self._initialized or self.img_w <= 1 or self.img_h <= 1:
            return 0.0, 0.0
        gx = px * float(self._grid_x_scale)
        gy = py * float(self._grid_y_scale)
        return gx, gy

    def _materialize_decay(self) -> None:
        """Flush the lazy global decay into the backing arrays when scale gets too small."""
        if self._decay_scale == 1.0:
            return
        for field in (self.support_count, self.birth_count):
            field *= self._decay_scale
        self._decay_scale = np.float32(1.0)

    def _scaled_field(self, field: np.ndarray) -> np.ndarray:
        """Return the effective decayed field values."""
        if self._decay_scale == 1.0:
            return field
        return field * self._decay_scale

    def _splat(
        self,
        field: np.ndarray,
        point: np.ndarray | tuple[float, float],
        weight: float,
    ) -> None:
        self._splat_many(field, [point], weight)

    def _splat_many(
        self,
        field: np.ndarray,
        points: list[np.ndarray] | tuple[np.ndarray, ...],
        weight: float,
    ) -> None:
        if not self._initialized:
            return
        if not points:
            return
        points_arr = np.asarray(points, dtype=np.float32)
        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(1, 2)

        gx = points_arr[:, 0] * self._grid_x_scale
        gy = points_arr[:, 1] * self._grid_y_scale
        cx = np.clip(np.rint(gx).astype(np.int32), 0, self.grid_w - 1)
        cy = np.clip(np.rint(gy).astype(np.int32), 0, self.grid_h - 1)

        x_idx = cx[:, None] + self._offsets[None, :]
        y_idx = cy[:, None] + self._offsets[None, :]
        valid_x = (x_idx >= 0) & (x_idx < self.grid_w)
        valid_y = (y_idx >= 0) & (y_idx < self.grid_h)

        x_delta = x_idx.astype(np.float32) - gx[:, None]
        y_delta = y_idx.astype(np.float32) - gy[:, None]
        kernel_x = np.exp((x_delta * x_delta) * self._gaussian_scale).astype(np.float32)
        kernel_y = np.exp((y_delta * y_delta) * self._gaussian_scale).astype(np.float32)
        kernel_x *= valid_x
        kernel_y *= valid_y

        kernel_sum = kernel_x.sum(axis=1, dtype=np.float64) * kernel_y.sum(axis=1, dtype=np.float64)
        valid_points = kernel_sum > 0.0
        if not np.any(valid_points):
            return

        x_idx = np.clip(x_idx[valid_points], 0, self.grid_w - 1)
        y_idx = np.clip(y_idx[valid_points], 0, self.grid_h - 1)
        valid_x = valid_x[valid_points]
        valid_y = valid_y[valid_points]
        kernel_x = kernel_x[valid_points]
        kernel_y = kernel_y[valid_points]
        norm = (
            np.float32(weight / float(self._decay_scale))
            / kernel_sum[valid_points].astype(np.float32)
        )

        updates = norm[:, None, None] * kernel_y[:, :, None] * kernel_x[:, None, :]
        valid_mask = valid_y[:, :, None] & valid_x[:, None, :]
        update_y = np.broadcast_to(y_idx[:, :, None], updates.shape)[valid_mask]
        update_x = np.broadcast_to(x_idx[:, None, :], updates.shape)[valid_mask]
        np.add.at(field, (update_y, update_x), updates[valid_mask])
