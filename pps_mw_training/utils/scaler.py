from dataclasses import dataclass
from typing import cast, Dict, List, Optional, Tuple, Union
import math

import numpy as np  # type: ignore


MIN_VALUE = 1e-6


@dataclass
class MinMaxScaler:
    """Scaler class for Min Max Scaling"""

    xoffset: np.ndarray
    gain: np.ndarray
    ymin: np.ndarray
    apply_log_scale: Optional[np.ndarray] = None

    def get_xoffset(
        self,
        idx: int,
    ) -> float:
        """Get xoffset."""
        return self.xoffset if self.xoffset.size == 1 else self.xoffset[idx]

    def get_gain(
        self,
        idx: int,
    ) -> float:
        """Get gain."""
        return self.gain if self.gain.size == 1 else self.gain[idx]

    def get_ymin(
        self,
        idx: int,
    ) -> float:
        """Get ymin."""
        return self.ymin if self.ymin.size == 1 else self.ymin[idx]

    def _apply_log_scale(self, idx) -> bool:
        """Check if log scaling should be applied."""
        if self.apply_log_scale is not None:
            return self.apply_log_scale[idx]
        return False

    def apply(
        self,
        x: np.ndarray,
        idx: Optional[int] = None,
    ) -> np.ndarray:
        """Apply forward scaling."""
        if idx is not None:
            if self._apply_log_scale(idx):
                x[x <= 0.0] = MIN_VALUE
                x = np.log(x)
            xoffset = self.get_xoffset(idx)
            ymin = self.get_ymin(idx)
            gain = self.get_gain(idx)
            return ymin + gain * (x - xoffset)

        return np.column_stack(
            [self.apply(x[:, idx], idx) for idx in range(x.shape[1])]
        )

    def reverse(
        self,
        y: np.ndarray,
        idx: Optional[int] = None,
    ) -> np.ndarray:
        """Apply reversed scaling."""
        if idx is not None:
            xoffset = self.get_xoffset(idx)
            ymin = self.get_ymin(idx)
            gain = self.get_gain(idx)

            data = xoffset + (y - ymin) / gain
            if self._apply_log_scale(idx):
                return np.exp(data)
            return data
        return np.column_stack(
            [self.reverse(y[:, idx], idx) for idx in range(y.shape[1])]
        )

    @staticmethod
    def get_min_value(param: Dict[str, Union[str, float]]) -> float:
        """Get min value from dict."""
        min_value = cast(float, param["min"])
        return (
            math.log(min_value + MIN_VALUE)
            if param["scale"] == "log"
            else min_value
        )

    @staticmethod
    def get_max_value(param: Dict[str, Union[str, float]]) -> float:
        max_value = cast(float, param["max"])
        return math.log(max_value) if param["scale"] == "log" else max_value

    @classmethod
    def from_dict(
        cls,
        params: List[Dict[str, Union[str, float]]],
        feature_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> "MinMaxScaler":
        """ "Get scaler object from dict."""
        y_min, y_max = feature_range
        try:
            return cls(
                xoffset=np.array([cls.get_min_value(p) for p in params]),
                gain=np.array(
                    [
                        (y_max - y_min)
                        / (cls.get_max_value(p) - cls.get_min_value(p))
                        for p in params
                    ]
                ),
                ymin=np.full(len(params), y_min),
                apply_log_scale=np.array([p["scale"] == "log" for p in params]),
            )
        except KeyError:
            raise ValueError(
                "Min and max should be set for all parameters in the inputdict"
            )


@dataclass
class StandardScaler:
    """Scaler class for Z score scaling"""

    mean: np.ndarray
    std: np.ndarray
    apply_log_scale: Optional[np.ndarray] = None

    def _apply_log_scale(self, idx) -> bool:
        """Check if log scaling should be applied."""
        if self.apply_log_scale is not None:
            return self.apply_log_scale[idx]
        return False

    def apply(
        self,
        x: np.ndarray,
        idx: Optional[int] = None,
    ) -> np.ndarray:
        """Apply forward scaling."""
        if idx is not None:
            if self._apply_log_scale(idx):
                x[x <= 0.0] = MIN_VALUE
                x = np.log(x)
            return (x - self.mean[idx]) / self.std[idx]

        return np.column_stack(
            [self.apply(x[:, idx], idx) for idx in range(x.shape[1])]
        )

    def reverse(
        self,
        y: np.ndarray,
        idx: Optional[int] = None,
    ) -> np.ndarray:
        """Apply reversed scaling."""
        if idx is not None:
            data = y * self.std + self.mean
            if self._apply_log_scale(idx):
                return np.exp(data)
            return data
        return np.column_stack(
            [self.reverse(y[:, idx], idx) for idx in range(y.shape[1])]
        )

    @staticmethod
    def get_mean(
        x: np.ndarray,
    ) -> float:
        """Get mean"""
        return np.float64(np.nanmean(x))

    @staticmethod
    def get_std(
        x: np.ndarray,
    ) -> float:
        """Get std"""
        return np.float64(np.nanstd(x))

    @classmethod
    def from_dict(
        cls,
        params: List[Dict[str, Union[str, float]]],
    ) -> "StandardScaler":
        """ "Get scaler object from dict."""
        try:
            return cls(
                mean=np.array([p["mean"] for p in params]),
                std=np.array([p["std"] for p in params]),
                apply_log_scale=np.array([p["scale"] == "log" for p in params]),
            )
        except KeyError:
            raise ValueError(
                "Mean and std should be set for all parameters in the inputdict"
            )


def get_scaler(
    params: list[dict[str, str | float]]
) -> Union[StandardScaler, MinMaxScaler]:
    """apropriate scaler class according to input params"""
    if "min" in list(params)[0] and "max" in list(params)[0]:
        return MinMaxScaler.from_dict(params)
    elif "std" in list(params)[0] and "mean" in list(params)[0]:
        return StandardScaler.from_dict(params)
    else:
        raise ValueError(
            "Either provide min, max or mean,std for scaling of data"
        )
