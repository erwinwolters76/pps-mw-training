from enum import Enum


class ChannelAtms(Enum):
    """ATMS channel enum."""

    ATMS_01 = 0  # 23.8
    ATMS_02 = 1  # 31.4
    ATMS_03 = 2  # 50.3
    ATMS_04 = 3  # 51.76
    ATMS_05 = 4  # 52.8
    ATMS_06 = 5  # 53.596±0.115
    ATMS_07 = 6  # 54.4
    ATMS_08 = 7  # 54.94
    ATMS_09 = 8  # 55.5
    ATMS_10 = 9  # 57.290344
    ATMS_11 = 10  # 57.290344±0.217
    ATMS_12 = 11  # 57.290344±0.3222±0.048
    ATMS_13 = 12  # 57.290344±0.3222±0.022
    ATMS_14 = 13  # 57.290344±0.3222±0.010
    ATMS_15 = 14  # 57.290344±0.3222±0.0045
    ATMS_16 = 15  # 88.2
    ATMS_17 = 16  # 165.5
    ATMS_18 = 17  # 183.31±7
    ATMS_19 = 18  # 183.31±4.5
    ATMS_20 = 19  # 183.31±3
    ATMS_21 = 20  # 183.31±1.8
    ATMS_22 = 21  # 183.31±1


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


BANDS = {
    "mw_low": {
        0: ["ATMS_01", "AMSUA_01"],  # 23.8 GHz
        1: ["ATMS_02", "AMSUA_02"],  # 31.4 GHz
    },
    "mw_50": {
        0: ["ATMS_03", "AMSUA_03"],  # 50.3 GHz
        1: ["ATMS_04"],  # 51.76 GHz
        2: ["ATMS_05", "AMSUA_04"],  # 52.8 GHz
        3: ["ATMS_06", "AMSUA_05"],  # 53.596 +/- 0.115
        4: ["ATMS_07", "AMSUA_06"],  # 54.4 GHz
        5: ["ATMS_08", "AMSUA_07"],  # 54.94 GHz
        6: ["ATMS_09", "AMSUA_08"],  # 55.5 GHz
        7: ["ATMS_10", "AMSUA_09"],  # 57.290344 GHz
        8: ["ATMS_11", "AMSUA_10"],  # 57.290344 +/- 0.217 GHz
        9: ["ATMS_12", "AMSUA_11"],  # 57.290344 +/- 0.3222 +/- 0.048 GHz
        10: ["ATMS_13", "AMSUA_12"],  # 57.290344 +/- 0.3222 +/- 0.022 GHz
        11: ["ATMS_14", "AMSUA_13"],  # 57.290344 +/- 0.3222 +/- 0.010 GHz
        12: ["ATMS_15", "AMSUA_14"],  # 57.290344 +/- 0.3222 +/- 0.0045 GHz
    },
    "mw_90": {
        0: ["ATMS_16"],  # 88.2 GHz
        1: ["MHS_01", "AMSUA_15"],  # 89 GHz
    },
    "mw_160": {
        0: ["ATMS_17"],  # 166 GHz
        1: ["MHS_02"],  # 157 GHz
    },
    "mw_183": {
        0: ["ATMS_18", "MHS_05"],  # 183 +/- 1 GHz
        1: ["ATMS_19"],  # 183 +/- 1.8 GHz
        2: ["ATMS_20", "MHS_04"],  # 183 +/- 3, GHz
        3: ["ATMS_21"],  # 183 +/- 4.5 GHz
        4: ["ATMS_22", "MHS_03"],  # 183 +/- 7 GHz
    },
}
