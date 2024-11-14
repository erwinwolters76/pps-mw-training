from typing import Dict, List, Union

# all available label paramters
ALL_LABEL_PARAMS: List[Dict[str, Union[str, float]]] = [
    {
        "name": "cloud_base",
        "scale": "linear",
        "min": 0.0,
        "max": 12000.0,
    },
    {
        "name": "cloud_fraction",
        "scale": "linear",
        "min": 0.0,
        "max": 100.0,
    },
]

# all available input parameters
ALL_INPUT_PARAMS: List[Dict[str, Union[str, float]]] = [
    {
        "name": "M01",
        "scale": "linear",
        "mean": 45.722798145709,
        "std": 21.95480387639247,
    },
    {
        "name": "M02",
        "scale": "linear",
        "mean": 43.472239755819054,
        "std": 22.867776801356122,
    },
    {
        "name": "M03",
        "scale": "linear",
        "mean": 40.923600920967864,
        "std": 23.409937236096514,
    },
    {
        "name": "M04",
        "scale": "linear",
        "mean": 36.45118479368475,
        "std": 22.51684208364647,
    },
    {
        "name": "M05",
        "scale": "linear",
        "mean": 36.142117032150246,
        "std": 24.38639123384754,
    },
    {
        "name": "M06",
        "scale": "linear",
        "mean": 21.87554449134835,
        "std": 15.935278292929638,
    },
    {
        "name": "M07",
        "scale": "linear",
        "mean": 38.15007132764771,
        "std": 24.576855357564384,
    },
    {
        "name": "M08",
        "scale": "linear",
        "mean": 38.15007132764771,
        "std": 24.576855357564384,
    },
    {
        "name": "M09",
        "scale": "linear",
        "mean": 4.793895239582563,
        "std": 8.85750747797539,
    },
    {
        "name": "M10",
        "scale": "linear",
        "mean": 21.83183990110206,
        "std": 13.875611051817904,
    },
    {
        "name": "M11",
        "scale": "linear",
        "mean": 19.034346107150558,
        "std": 12.283460305970328,
    },
    {
        "name": "M12",
        "scale": "linear",
        "mean": 284.0697882722524,
        "std": 17.423538257835077,
    },
    {
        "name": "M13",
        "scale": "linear",
        "mean": 272.0085796736719,
        "std": 17.56035725381957,
    },
    {
        "name": "M14",
        "scale": "linear",
        "mean": 266.56287274085213,
        "std": 18.307663499201603,
    },
    {
        "name": "M15",
        "scale": "linear",
        "mean": 267.87728514528317,
        "std": 19.500754334747885,
    },
    {
        "name": "M16",
        "scale": "linear",
        "mean": 266.6215824795838,
        "std": 19.38803496330258,
    },
    {
        "name": "h_2meter",
        "scale": "linear",
        "mean": 76.53193867314339,
        "std": 16.630991795963556,
    },
    {
        "name": "t_2meter",
        "scale": "linear",
        "mean": 285.5732175305176,
        "std": 13.72634376284516,
    },
    {
        "name": "p_surface",
        "scale": "linear",
        "mean": 98391.67473109375,
        "std": 6456.561415968002,
    },
    {
        "name": "z_surface",
        "scale": "linear",
        "mean": 2433.1688326830313,
        "std": 5949.827141663237,
    },
    {
        "name": "ciwv",
        "scale": "linear",
        "mean": 23.143573866659402,
        "std": 16.47412636069295,
    },
    {
        "name": "t250",
        "scale": "linear",
        "mean": 224.69264364868164,
        "std": 6.968061496997991,
    },
    {
        "name": "t400",
        "scale": "linear",
        "mean": 245.3769554107666,
        "std": 10.1230745807611,
    },
    {
        "name": "t500",
        "scale": "linear",
        "mean": 256.3739975970459,
        "std": 10.107934260577437,
    },
    {
        "name": "t700",
        "scale": "linear",
        "mean": 271.5975372644043,
        "std": 10.673382563556011,
    },
    {
        "name": "t850",
        "scale": "linear",
        "mean": 279.3380216827393,
        "std": 11.950486262270868,
    },
    {
        "name": "t900",
        "scale": "linear",
        "mean": 281.54549735229494,
        "std": 12.608327215777589,
    },
    {
        "name": "t950",
        "scale": "linear",
        "mean": 283.7890126626587,
        "std": 13.363535983585342,
    },
    {
        "name": "t1000",
        "scale": "linear",
        "mean": 286.7742821728768,
        "std": 14.474404062499847,
    },
    {
        "name": "t_sea",
        "scale": "linear",
        "mean": 286.60209551483155,
        "std": 15.19607293352855,
    },
    {
        "name": "t_land",
        "scale": "linear",
        "mean": 286.60209551483155,
        "std": 15.19607293352855,
    },
    {
        "name": "rh250",
        "scale": "linear",
        "mean": 0.005291610806524404,
        "std": 0.003723000052433534,
    },
    {
        "name": "rh400",
        "scale": "linear",
        "mean": 0.006046445369025605,
        "std": 0.0033096767106421854,
    },
    {
        "name": "rh500",
        "scale": "linear",
        "mean": 0.005735704751800622,
        "std": 0.003239517962875649,
    },
    {
        "name": "rh700",
        "scale": "linear",
        "mean": 0.006034013814882565,
        "std": 0.0029108583874336603,
    },
    {
        "name": "rh850",
        "scale": "linear",
        "mean": 0.007134334129411873,
        "std": 0.0024109007921006044,
    },
    {
        "name": "rh900",
        "scale": "linear",
        "mean": 0.007627256122758554,
        "std": 0.0022071383987135465,
    },
    {
        "name": "rh950",
        "scale": "linear",
        "mean": 0.007926371323379571,
        "std": 0.0021499258670604083,
    },
    {
        "name": "rh1000",
        "scale": "linear",
        "mean": 0.00738849284664931,
        "std": 0.002174385255368654,
    },
    {
        "name": "q250",
        "scale": "linear",
        "mean": 6.507605248566506e-05,
        "std": 7.889838169028413e-05,
    },
    {
        "name": "q400",
        "scale": "linear",
        "mean": 0.00046727919108313247,
        "std": 0.0005378467546237773,
    },
    {
        "name": "q500",
        "scale": "linear",
        "mean": 0.001033781902913779,
        "std": 0.0011316728144658357,
    },
    {
        "name": "q700",
        "scale": "linear",
        "mean": 0.0028668664924348104,
        "std": 0.002547806436435812,
    },
    {
        "name": "q850",
        "scale": "linear",
        "mean": 0.005026202192237133,
        "std": 0.003941367242371887,
    },
    {
        "name": "q900",
        "scale": "linear",
        "mean": 0.005886857095257228,
        "std": 0.004439638445114495,
    },
    {
        "name": "q950",
        "scale": "linear",
        "mean": 0.006777255145888057,
        "std": 0.005117977373985545,
    },
    {
        "name": "q1000",
        "scale": "linear",
        "mean": 0.007224722144002037,
        "std": 0.0054276352054161625,
    },
]


def get_selected_params(
    names: List[str], input_params: List[Dict[str, Union[str, float]]]
) -> List[Dict[str, Union[str, float]]]:
    return [param for param in input_params if param["name"] in names]
