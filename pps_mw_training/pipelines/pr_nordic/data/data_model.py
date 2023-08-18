from enum import Enum


class ChannelAtms(Enum):
    """ATMS channel enum."""

    ATMS_01 = "23.8"
    ATMS_02 = "31.4"
    ATMS_03 = "50.3"
    ATMS_04 = "51.76"
    ATMS_05 = "52.8"
    ATMS_06 = "53.596±0.115"
    ATMS_07 = "54.4"
    ATMS_08 = "54.94"
    ATMS_09 = "55.5"
    ATMS_10 = "57.290344"
    ATMS_11 = "57.290344±0.217"
    ATMS_12 = "57.290344±0.3222±0.048"
    ATMS_13 = "57.290344±0.3222±0.022"
    ATMS_14 = "57.290344±0.3222±0.010"
    ATMS_15 = "57.290344±0.3222±0.0045"
    ATMS_16 = "88.2"
    ATMS_17 = "165.5"
    ATMS_18 = "183.31±7"
    ATMS_19 = "183.31±4.5"
    ATMS_20 = "183.31±3"
    ATMS_21 = "183.31±1.8"
    ATMS_22 = "183.31±1"


class Platform(Enum):
    """Platform enum. """
    JPSS1 = 0
    SNPP = 1

    @classmethod
    def from_string(cls, string: str) -> "Platform":
        if "JPSS-1" in string:
            return cls.JPSS1
        elif "SUOMI-NPP" in string:
            return cls.SNPP
        raise ValueError(f"Could not determine platform from string: {string}")
