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
        self._initialized = True

    def step(self) -> None:
        """Apply a slow temporal decay to all maps once per frame."""
        if not self._initialized or self.decay >= 1.0:
            return
        for field in (self.support_count, self.birth_count):
            field *= self.decay

    def add_support(self, point: np.ndarray | tuple[float, float], weight: float = 1.0) -> None:
        self._splat(self.support_count, point, weight)

    def add_birth(self, point: np.ndarray | tuple[float, float], weight: float = 1.0) -> None:
        self._splat(self.birth_count, point, weight)

    def stats(self) -> SpatialPriorStats:
        """Return aggregate counts for debugging and tests."""
        return SpatialPriorStats(
            support_total=float(self.support_count.sum()),
            birth_total=float(self.birth_count.sum()),
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

        denom = self.support_count + self.prior_alpha + self.prior_beta
        walk_scale = max(float(self.support_count.max()), 1.0)
        confidence = self.support_count + self.birth_count
        return {
            "walkable": np.clip(self.support_count / walk_scale, 0.0, 1.0),
            "birth": np.clip((self.birth_count + self.prior_alpha) / denom, 0.0, 1.0),
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

        support_density = self._local_sum(self.support_count)
        birth_density = self._local_sum(self.birth_count)
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
        shape = field.shape
        out = np.zeros(shape, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                out += padded[dy : dy + shape[0], dx : dx + shape[1]]
        return out

    def _point_to_grid(self, point: np.ndarray | tuple[float, float]) -> tuple[float, float]:
        px = float(point[0])
        py = float(point[1])
        if not self._initialized or self.img_w <= 1 or self.img_h <= 1:
            return 0.0, 0.0
        gx = px / float(self.img_w - 1) * float(self.grid_w - 1)
        gy = py / float(self.img_h - 1) * float(self.grid_h - 1)
        return gx, gy

    def _splat(
        self,
        field: np.ndarray,
        point: np.ndarray | tuple[float, float],
        weight: float,
    ) -> None:
        if not self._initialized:
            return
        gx, gy = self._point_to_grid(point)
        radius = max(1, int(np.ceil(self.sigma * 3.0)))
        cx = int(np.clip(round(gx), 0, self.grid_w - 1))
        cy = int(np.clip(round(gy), 0, self.grid_h - 1))
        x0 = max(0, cx - radius)
        x1 = min(self.grid_w - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(self.grid_h - 1, cy + radius)
        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        kernel = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * self.sigma ** 2))
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 0:
            return
        field[y0 : y1 + 1, x0 : x1 + 1] += (weight / kernel_sum) * kernel.astype(np.float32)
