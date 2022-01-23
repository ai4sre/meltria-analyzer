import numpy as np
import pytest

from tsdr import tsdr

cases_for_stationality = [
    {
        'name': 's-front-end_latency',
        'desc': 'clear anomaly range',
        'is_unstationality': True,
        'datapoints': [
            0.0139, 0.0134, 0.0131, 0.0129, 0.0127, 0.0128, 0.012 , 0.0119,
            0.0115, 0.0127, 0.0134, 0.013 , 0.012 , 0.0122, 0.0123, 0.0123,
            0.0115, 0.0118, 0.0125, 0.0172, 0.0228, 0.0214, 0.0168, 0.0127,
            0.0129, 0.0134, 0.0144, 0.0139, 0.013 , 0.012 , 0.0134, 0.0149,
            0.0144, 0.015 , 0.0163, 0.0168, 0.0162, 0.0155, 0.0155, 0.0142,
            0.0136, 0.0123, 0.0122, 0.0116, 0.0116, 0.0115, 0.0116, 0.012 ,
            0.013 , 0.0132, 0.0144, 0.0144, 0.0168, 0.0159, 0.0153, 0.013 ,
            0.0125, 0.0126, 0.0135, 0.0144, 0.0139, 0.0127, 0.0123, 0.0118,
            0.0115, 0.0114, 0.0118, 0.0126, 0.0133, 0.014 , 0.014 , 0.0132,
            0.0131, 0.0138, 0.0142, 0.0139, 0.0133, 0.0129, 0.0139, 0.0138,
            0.0141, 0.0133, 0.0126, 0.0119, 0.0113, 0.0126, 0.0133, 0.0138,
            0.0123, 0.0121, 0.012 , 0.0121, 0.0136, 0.0137, 0.0138, 0.0129,
            0.0126, 0.0123, 0.0133, 0.0144, 0.0172, 0.0157, 0.0154, 0.0186,
            0.0271, 0.0334, 0.038 , 0.0374, 0.0362, 0.0366, 0.036 , 0.0355,
            0.0365, 0.0405, 0.0433, 0.0418, 0.0395, 0.039 , 0.0389, 0.0384
        ],
    }, {
        'name': 'c-catalogue_last_seen',
        'desc': 'straight line with upper trend',
        'is_unstationality': False,
        'datapoints': [
            1.63903003e+09, 1.63903004e+09, 1.63903006e+09, 1.63903008e+09,
            1.63903009e+09, 1.63903010e+09, 1.63903012e+09, 1.63903014e+09,
            1.63903015e+09, 1.63903016e+09, 1.63903018e+09, 1.63903020e+09,
            1.63903021e+09, 1.63903022e+09, 1.63903024e+09, 1.63903026e+09,
            1.63903027e+09, 1.63903028e+09, 1.63903030e+09, 1.63903032e+09,
            1.63903033e+09, 1.63903034e+09, 1.63903036e+09, 1.63903038e+09,
            1.63903039e+09, 1.63903040e+09, 1.63903042e+09, 1.63903044e+09,
            1.63903045e+09, 1.63903046e+09, 1.63903048e+09, 1.63903050e+09,
            1.63903051e+09, 1.63903052e+09, 1.63903054e+09, 1.63903056e+09,
            1.63903057e+09, 1.63903058e+09, 1.63903060e+09, 1.63903062e+09,
            1.63903063e+09, 1.63903064e+09, 1.63903066e+09, 1.63903068e+09,
            1.63903069e+09, 1.63903070e+09, 1.63903072e+09, 1.63903074e+09,
            1.63903075e+09, 1.63903076e+09, 1.63903078e+09, 1.63903080e+09,
            1.63903081e+09, 1.63903082e+09, 1.63903084e+09, 1.63903086e+09,
            1.63903087e+09, 1.63903088e+09, 1.63903090e+09, 1.63903092e+09,
            1.63903093e+09, 1.63903094e+09, 1.63903096e+09, 1.63903098e+09,
            1.63903099e+09, 1.63903100e+09, 1.63903102e+09, 1.63903104e+09,
            1.63903105e+09, 1.63903106e+09, 1.63903108e+09, 1.63903110e+09,
            1.63903111e+09, 1.63903112e+09, 1.63903114e+09, 1.63903116e+09,
            1.63903117e+09, 1.63903118e+09, 1.63903120e+09, 1.63903122e+09,
            1.63903123e+09, 1.63903124e+09, 1.63903126e+09, 1.63903128e+09,
            1.63903129e+09, 1.63903130e+09, 1.63903132e+09, 1.63903134e+09,
            1.63903135e+09, 1.63903136e+09, 1.63903138e+09, 1.63903140e+09,
            1.63903141e+09, 1.63903142e+09, 1.63903144e+09, 1.63903146e+09,
            1.63903147e+09, 1.63903148e+09, 1.63903150e+09, 1.63903152e+09,
            1.63903153e+09, 1.63903154e+09, 1.63903156e+09, 1.63903158e+09,
            1.63903159e+09, 1.63903160e+09, 1.63903162e+09, 1.63903164e+09,
            1.63903165e+09, 1.63903166e+09, 1.63903168e+09, 1.63903170e+09,
            1.63903171e+09, 1.63903172e+09, 1.63903174e+09, 1.63903176e+09,
            1.63903177e+09, 1.63903178e+09, 1.63903180e+09, 1.63903182e+09,
        ]
    }, {
        'name': 's-user_latency',
        'desc': 'A spike in short time',
        'is_unstationality': True,
        'datapoints': [
            0.0022, 0.002 , 0.0022, 0.0021, 0.0022, 0.002 , 0.0019, 0.0019,
            0.0021, 0.0021, 0.0021, 0.0023, 0.0023, 0.0023, 0.0022, 0.0022,
            0.0022, 0.0021, 0.0022, 0.0021, 0.002 , 0.0021, 0.0021, 0.0021,
            0.0023, 0.0024, 0.0025, 0.0023, 0.0021, 0.0021, 0.0021, 0.0022,
            0.0021, 0.002 , 0.0019, 0.002 , 0.0022, 0.0022, 0.0022, 0.002 ,
            0.002 , 0.0019, 0.002 , 0.002 , 0.0022, 0.0026, 0.0025, 0.0025,
            0.0021, 0.002 , 0.002 , 0.002 , 0.002 , 0.002 , 0.002 , 0.0019,
            0.0019, 0.0021, 0.0025, 0.0025, 0.0024, 0.0022, 0.0023, 0.0023,
            0.0021, 0.0022, 0.0022, 0.0022, 0.0022, 0.0021, 0.0022, 0.002 ,
            0.0021, 0.0021, 0.0022, 0.0021, 0.002 , 0.002 , 0.0022, 0.0023,
            0.0024, 0.0023, 0.0023, 0.0021, 0.0021, 0.0021, 0.0021, 0.0025,
            0.0025, 0.0024, 0.0021, 0.0021, 0.0023, 0.0022, 0.0022, 0.0021,
            0.0021, 0.0022, 0.0022, 0.0024, 0.0023, 0.0021, 0.002 , 0.0033,
            0.0173, 0.0158, 0.0273, 0.0027, 0.0038, 0.0035, 0.0025, 0.0038,
            0.0044, 0.0046, 0.0033, 0.0029, 0.0024, 0.0023, 0.0023, 0.0022
        ],
    }, {
        'name': 's-front-end_latency',
        'desc': 'Previously misdetected series in ADF test autolag',
        'is_unstationality': True,
        'datapoints': [
            0.0123, 0.0127, 0.013 , 0.0125, 0.0117, 0.0119, 0.0119, 0.0122,
            0.0122, 0.0123, 0.0121, 0.0117, 0.0119, 0.0123, 0.0126, 0.0128,
            0.0137, 0.0137, 0.0132, 0.0124, 0.0124, 0.0125, 0.015 , 0.0149,
            0.0145, 0.0115, 0.0115, 0.0117, 0.0118, 0.0124, 0.0142, 0.0148,
            0.0145, 0.013 , 0.0129, 0.0125, 0.0122, 0.0121, 0.0124, 0.0121,
            0.0116, 0.0114, 0.0119, 0.0119, 0.0126, 0.0128, 0.0128, 0.0118,
            0.0115, 0.0117, 0.0126, 0.0124, 0.0124, 0.0117, 0.012 , 0.0119,
            0.0119, 0.0117, 0.0118, 0.0124, 0.0125, 0.0123, 0.0116, 0.0114,
            0.0114, 0.0115, 0.0135, 0.0136, 0.0135, 0.0113, 0.0111, 0.0114,
            0.0116, 0.0127, 0.0131, 0.0128, 0.0116, 0.0109, 0.0111, 0.0111,
            0.0117, 0.0117, 0.0121, 0.0119, 0.012 , 0.0118, 0.0118, 0.0115,
            0.0118, 0.0123, 0.0128, 0.0125, 0.0122, 0.0119, 0.0122, 0.0119,
            0.0119, 0.0117, 0.012 , 0.0124, 0.0129, 0.0127, 0.0133, 0.0558,
            0.1504, 0.5013, 0.4801, 0.4302, 0.4447, 0.309 , 0.3522, 0.2519,
            0.3595, 0.3541, 0.4339, 0.1808, 0.0604, 0.0293, 0.0129, 0.0121
        ]
    }, {
        'name': 'c-user_memory_working_set_bytes',
        'desc': 'Previously misdetected series in ADF test autolag',
        'is_unstationality': True,
        'datapoints': [
            11206656., 11218944., 11243520., 11243520., 11259904., 11268096.,
            11300864., 11448320., 11448320., 11612160., 11517952., 11517952.,
            11616256., 11624448., 11644928., 11673600., 11673600., 11689984.,
            11698176., 11706368., 11718656., 11730944., 11743232., 11755520.,
            11821056., 11821056., 11837440., 11837440., 11878400., 11898880.,
            11898880., 11911168., 11911168., 11915264., 11931648., 11931648.,
            11935744., 12046336., 12046336., 12046336., 12054528., 12066816.,
            12066816., 12070912., 12201984., 12083200., 12083200., 12083200.,
            12087296., 12087296., 12111872., 12111872., 12115968., 12120064.,
            12124160., 12132352., 12132352., 12140544., 12144640., 12201984.,
            12201984., 12214272., 12214272., 12222464., 12230656., 12230656.,
            12238848., 12242944., 12247040., 12247040., 12251136., 12251136.,
            12251136., 12255232., 12255232., 12263424., 12263424., 12263424.,
            12263424., 12263424., 12267520., 12267520., 12267520., 12267520.,
            12271616., 12271616., 12271616., 12271616., 12279808., 12288000.,
            12288000., 12292096., 12296192., 12296192., 12296192., 12296192.,
            12357632., 12378112., 12378112., 12378112., 12382208., 12386304.,
            77279232., 76705792., 79749120., 81559552., 81559552., 81559552.,
            82788352., 83648512., 83652608., 83652608., 83726336., 83726336.,
            83726336., 84996096., 85188608., 85274624., 85303296., 85385216.
        ],
    }, {
        'name': 's-orders_latency',
        'desc': 'Previously misdetected series in ADF test maxlag=1',
        'is_unstationality': True,
        'datapoints': [
            2.2000e-02, 2.5900e-02, 2.6700e-02, 2.2000e-02, 2.1900e-02,
            2.2300e-02, 2.2800e-02, 2.3000e-02, 2.1900e-02, 2.2000e-02,
            2.0700e-02, 2.2800e-02, 2.1200e-02, 1.8600e-02, 1.6100e-02,
            1.6800e-02, 1.6700e-02, 1.4900e-02, 1.5300e-02, 1.9500e-02,
            2.1100e-02, 1.9600e-02, 1.7000e-02, 1.5600e-02, 1.7000e-02,
            1.5900e-02, 1.5800e-02, 1.6800e-02, 1.7600e-02, 1.9300e-02,
            2.0400e-02, 2.0000e-02, 2.0500e-02, 1.6400e-02, 1.6300e-02,
            1.3800e-02, 1.5300e-02, 1.5000e-02, 1.5900e-02, 1.5000e-02,
            1.7500e-02, 1.7000e-02, 1.7200e-02, 1.4000e-02, 1.5300e-02,
            1.8100e-02, 1.9200e-02, 1.8500e-02, 1.5400e-02, 1.9300e-02,
            2.0100e-02, 2.1900e-02, 1.7700e-02, 1.6300e-02, 1.5600e-02,
            1.6700e-02, 1.6500e-02, 1.5500e-02, 1.9400e-02, 1.9500e-02,
            2.4400e-02, 1.9800e-02, 2.2200e-02, 1.7300e-02, 1.7400e-02,
            1.6700e-02, 1.6600e-02, 1.7000e-02, 1.7000e-02, 1.9300e-02,
            2.1200e-02, 2.0600e-02, 2.2700e-02, 1.9600e-02, 2.0100e-02,
            1.6700e-02, 1.8400e-02, 1.6100e-02, 1.8100e-02, 1.8900e-02,
            1.9500e-02, 1.6800e-02, 1.6300e-02, 1.6300e-02, 1.6100e-02,
            1.5400e-02, 1.6500e-02, 2.0000e-02, 2.0800e-02, 1.7800e-02,
            1.6900e-02, 1.6300e-02, 1.7600e-02, 1.6500e-02, 1.6800e-02,
            1.6400e-02, 1.5500e-02, 1.6800e-02, 1.6000e-02, 1.4500e-02,
            1.4800e-02, 1.6000e-02, 1.7500e-02, 4.0280e-01, 1.2177e+00,
            1.8020e+00, 1.4841e+00, 1.2880e-01, 1.3000e-03, 1.2000e-03,
            1.2000e-03, 1.2000e-03, 1.2000e-03, 1.1000e-03, 1.2000e-03,
            1.2000e-03, 1.2000e-03, 1.1000e-03, 3.4710e-01, 2.0560e-01
        ]
    }, {
        'name': 's-orders_latency02',
        'desc': 'Previously misdetected series in ADF test maxlag=1',
        'is_unstationality': True,
        'datapoints': [
            2.3600e-02, 2.1300e-02, 1.7000e-02, 1.7300e-02, 1.9300e-02,
            1.9800e-02, 1.9800e-02, 1.7600e-02, 1.9900e-02, 1.9800e-02,
            1.9700e-02, 2.3700e-02, 2.6400e-02, 3.2500e-02, 2.9500e-02,
            2.8900e-02, 2.4000e-02, 2.0600e-02, 1.8700e-02, 1.9300e-02,
            2.4400e-02, 2.3200e-02, 2.2200e-02, 3.6400e-02, 3.6800e-02,
            4.3200e-02, 2.7600e-02, 2.8600e-02, 2.5800e-02, 2.3600e-02,
            1.9300e-02, 2.0300e-02, 1.6400e-02, 2.1100e-02, 2.2100e-02,
            2.3300e-02, 2.0000e-02, 1.7800e-02, 1.7700e-02, 2.1400e-02,
            2.4700e-02, 2.5300e-02, 2.1000e-02, 1.7500e-02, 1.6900e-02,
            1.6300e-02, 1.7600e-02, 1.7000e-02, 1.7300e-02, 1.7300e-02,
            1.8400e-02, 2.0400e-02, 2.1100e-02, 2.2000e-02, 1.9600e-02,
            2.6200e-02, 2.6600e-02, 2.5300e-02, 1.9700e-02, 2.2000e-02,
            2.4400e-02, 2.0700e-02, 2.1800e-02, 2.1700e-02, 2.2500e-02,
            2.0400e-02, 2.3300e-02, 2.4400e-02, 2.7200e-02, 2.5700e-02,
            2.5500e-02, 1.9100e-02, 1.7200e-02, 1.6800e-02, 1.6300e-02,
            1.6400e-02, 1.7600e-02, 1.8000e-02, 1.8100e-02, 1.6300e-02,
            1.6200e-02, 1.6900e-02, 1.7100e-02, 2.0300e-02, 1.9100e-02,
            2.1000e-02, 3.2500e-02, 3.4200e-02, 3.2100e-02, 1.7500e-02,
            1.7200e-02, 1.5600e-02, 1.7000e-02, 2.1000e-02, 2.1400e-02,
            2.5600e-02, 2.2200e-02, 2.2600e-02, 2.0300e-02, 2.2300e-02,
            2.3700e-02, 2.0300e-02, 1.6500e-02, 1.5870e-01, 5.3720e-01,
            1.0559e+00, 1.2465e+00, 1.0660e+00, 9.3380e-01, 8.8320e-01,
            1.0406e+00, 1.1566e+00, 1.0037e+00, 5.2010e-01, 1.1000e-03,
            1.1000e-03, 1.1000e-03, 1.1000e-03, 1.1000e-03, 1.1000e-03
        ],
    },
]


@pytest.mark.parametrize("case", cases_for_stationality)
@pytest.mark.parametrize("alpha", [0.01])
@pytest.mark.parametrize("cv_threshold", [0.1, 0.5])
@pytest.mark.parametrize("knn_threshold", [0.01, 0.02])
@pytest.mark.parametrize("regression", ['c', 'ct'])
def test_unit_root_based_model(case, alpha, cv_threshold, knn_threshold, regression):
    got: bool = tsdr.unit_root_based_model(
        series=np.array(case['datapoints']),
        tsifter_step1_unit_root_model='pp',
        tsifter_step1_unit_root_alpla=alpha,
        tsifter_step1_unit_root_regression=regression,
        tsifter_step1_post_cv=True,
        tsifter_step1_cv_threshold=cv_threshold,
        tsifter_step1_post_knn=True,
        tsifter_step1_knn_threshold=knn_threshold,
    )
    assert got == case['is_unstationality'], f"{case['name']}: {case['desc']}"