import requests
import csv
from io import StringIO
from typing import List
from datetime import datetime
import sys
import os
#from pipeline_functions import *
from typing import List
import pandas as pd

metadata = {
       "ALLINGE": {
        "t1s_cosphitrf1": 515615,
        "t2s_cosphitrf2": 515623,
        "t1s_belastning": 515613,
        "t2s_belastning": 515621,
        "t1s_sptrinvisning": 515618,
        "t2s_sptrinvisning": 515626,
        "lok_10kvskinnespend": 516443,
    },
    "OLSKER": {
        "t1s_aktiveffekt": 516063,
        "t1s_reaktiveffekt": 516069,
        "t2s_aktiveffekt": 516081,
        "t2s_reaktiveeffekt": 516087,
        "t1p_belastningill": 516059,
        "t1p_belastningil2": 516060,
        "t1p_belastningil3": 516061,
        "t1s_belastning": 516064,
        "t2p_belastningill": 516077,
        "t2p_belastningil2": 516078,
        "t2p_belastningil3": 516079,
        "t2s_belastning": 516082,
        "t1s_spendingvl1": 516071,
        "t1s_spendingv12": 516072,
        "t1s_spendingv13": 516073,
        "t2s_spendingvl1": 516089,
        "t2s_spendingv12": 516090,
        "t2s_spendingv13": 516091,
        "t1s_sptrinvisning": 516075,
        "t2s_sptrinvisning": 516093,
        "t1s_cosphitrf1": 516066,
        "t2s_cosphitrf2": 516084,
        "ost_aktiveffektny": 516043,
        "ost_reaktiveeffektny": 516050,
        "has_aktiveffektny": 516022,
        "has_reaktiveffektny": 516029,
        "all_belastning": 516018,
        "all_60kvliniespend": 516017,
        "lok_10kvskinnespend": 516040
    },
    "OSTERLARS": {
        "t1s_aktiveffekt": 516134,
        "t1s_reaktiveffekt": 516139,
        "tip_belastningill": 516130,
        "t1p_belastningil2": 516131,
        "tip_belastningil3": 516132,
        "t1s_belastning": 516135,
        "t1s_spendingvl1": 516140,
        "t1s_spendingv12": 516141,
        "t1s_spendingv13": 516142,
        "t1s_sptrinvisning": 516143,
        "t1s_cosphitrf1": 516137,
        "t1s_10kvskinnespend": 516133,
        "ols_aktiveffektny": 516119,
        "ols_reaktiveffektny": 516124,
        "dal_aktiveffektny": 516105,
        "dal_reaktiveffektny": 516110,
        "gud_belastning": 516116
    },
    "HASLE": {
        "t1s_aktiveffekt": 515910,
        "t1s_reaktiveffekt": 515941,
        "t2s_aktiveffekt": 515922,
        "t2s_reaktiveffekt": 515927,
        "t1p_belastningill": 515905,
        "t1p_belastningil2": 515906,
        "t1p_belastningil3": 515907,
        "t1s_belastning": 515940,
        "t2p_belastningill": 515918,
        "t2p_belastningil2": 515919,
        "t2p_belastningil3": 515920,
        "t2s_belastning": 515923,
        "t1s_spendingvl1": 515912,
        "t1s_spendingv12": 515913,
        "t1s_spendingv13": 515914,
        "t2s_spendingvl1": 515929,
        "t2s_spendingv12": 515930,
        "t2s_spendingv13": 515931,
        "t1s_sptrinvisning": 515916,
        "t2s_sptrinvisning": 515933,
        "t1s_trp1pf": 516445,
        "t2s_trp2pf": 516446,
        "sok_mwbelastning": 515900,
        "sok_mvarbelastning": 515898,
        "sno_aktiveffekt": 515869,
        "sno_reaktiveffekt": 515884,
        "ols_aktiveffektny": 515763,
        "ols_reaktiveffektny": 515841,
        "lok_belastning": 515756
    },
    "SNORREBAKKEN": {
        "t1s_aktiveffekt": 516257,
        "t1s_reaktiveffekt": 516262,
        "t1p_belastningill": 516253,
        "t1p_belastningil2": 516254,
        "t1p_belastningil3": 516255,
        "t1s_belastning": 516258,
        "t1s_spendingvl1": 516263,
        "t1s_spendingv12": 516264,
        "t1s_spendingv13": 516265,
        "t1s_sptrinvisning": 516266,
        "t1s_cosphitrf1": 516260,
        "t1s_10kvskinnespend": 516256,
        "var_aktiveffektny": 516267,
        "var_reaktiveffektny": 516272,
        "has_aktiveffektny": 516236,
        "has_reaktiveffektny": 516241
    },
    "VAERKET": {
        "k01_effekt": 516332,
        "k01_reaktiveffekt": 516333,
        "k01_10kvspending": 516329,
        "k01_belastning": 516330,
        "k01_cosphi": 516331,
        "k02_effekt": 516338,
        "k02_reaktiveffekt": 516394,
        "k02_10kvskcspending": 516334,
        "k02_10kvspending": 516335,
        "k02_belastning": 516336,
        "k02_cosphi": 516337,
        "k04_effekt": 516343,
        "k04_reaktiveffekt": 516344,
        "k04_10kvspending": 516340,
        "k04_belastning": 516341,
        "k04_cosphi": 516342,
        "b15_mwblok5": 516316,
        "bl6_blok6el": 516317,
        "b16_blok6fjernvarme": 516318,
        "blok5_mw": 516319,
        "fel_verkettotalmw": 516327,
        "tip_belastning": 516383,
        "t2_sp60kvbelastning": 516384,
        "t2_sptrinvisning": 516385,
        "t5p_belastning": 516386,
        "t6p_60kvliniespend": 516387,
        "t6p_belastning": 516388,
        "has_spendingvl12": 516328,
        "ves_60kvliniespend": 516389,
        "ves_belastning": 516390,
        "sno_aktiveffektny": 516371,
        "sno_reaktiveffektny": 516377,
        "sno_60kvliniespend": 516370,
        "sno_spendingvl12": 516378,
        "sno_belastning": 516372,
        "nli_aktiveffekt": 516348,
        "nli_reaktiveffekt": 516355,
        "rsy_belastning": 516366,
        "rsy_60kvliniespend": 516365,
        "via_belastning": 516392,
        "via_60kvliniespend": 516391
    },
    "VESTHAVNEN": {
        "t1s_belastning": 795792,
        "t1s_sptrinvisning": 795797,
        "t1s_cosphitrf1": 795793,
        "rno_liniespending": 795796,
        "rno_belastning": 795791,
        "var_belastning": 795794,
        "var_liniespending": 795798
    },
    "VIADUKTEN": {
        "t1s_belastning": 516427,
        "t1s_sptrinvisning": 516430,
        "t1s_cosphitrf1": 516429,
        "t1s_10kvskinnespend": 516426,
        "var_60kvliniespend": 516432,
        "var_belastning": 516433
    },
    "RØNNE_SYD": {
        "t1s_belastning": 516224,
        "t1s_sptrinvisning": 516227,
        "t1s_cosphitrf1": 516226,
        "lok_10kvskinnespend": 516220,
        "aki_aktiveffektny": 516196,
        "aki_reaktiveffektny": 516207,
        "aki_60kvliniebelastning": 516193,
        "aki_60kvliniespend": 516194,
        "var_60kvliniebelastning": 516228,
        "var_60kvliniespend": 516229,
        "via_60kvliniespend": 516231,
        "via_60kvliniebelastning": 516230
    },
    "RØNNE_NORD": {
        "t1s_belastning": 516183,
        "t1s_sptrinvisning": 516186,
        "t1s_cosphitrf1": 516185,
        "via_belastning": 516188,
        "via_60kvliniespend": 516187,
        "has_belastning": 516178,
        "has_60kvliniespend": 516177
    },
    "AAKIRKEBY": {
        "t1s_effekt": 515527,
        "t1s_reaktiveffekt": 515529,
        "t1s_belastning": 515522,
        "t1s_cosphitrf1": 515525,
        "t1s_sptrinvisning": 515531,
        "t1s_spending": 515530,
        "tip_belastning": 515518,
        "t1p_liniespending": 515520,
        "t2s_effekt": 515543,
        "t2s_reaktiveffekt": 515545,
        "t2p_belastning": 515534,
        "t2s_cosphitrf2": 515541,
        "t2s_spending": 515546,
        "t2p_liniespending": 515537,
        "t2s_belastning": 515539,
        "t2s_sptrinvisning": 515547,
        "bod_aktiveffektny": 515441,
        "bod_reaktiveffektny": 515447,
        "bod_spendingvliny": 515449,
        "bod_spendingvl2ny": 515450,
        "bod_spendingvl3ny": 515451,
        "bod_belastningiliny": 515442,
        "bod_belastningil2ny": 515443,
        "bod_belastningil3ny": 515444,
        "rsy_aktiveffektny": 515486,
        "rsy_reaktiveffektny": 515492,
        "rsy_belastningiliny": 515487,
        "rsy_belastningil2ny": 515488,
        "rsy_belastningil3ny": 515489,
        "rsy_spendingvliny": 515494,
        "rsy_spendingv12ny": 515495,
        "rsy_spendingv13ny": 515496
    },
    "BODILSKER": {
        "t1s_aktiveffekt": 515696,
        "t1s_reaktiveffekt": 515702,
        "lok_10kvskinnespend": 516444,
        "t1p_belastningill": 515692,
        "t1p_belastningil2": 515693,
        "t1p_belastningil3": 515694,
        "t1s_belastning": 515697,
        "t1s_spendingvl1": 515704,
        "t1s_spendingv12": 515705,
        "t1s_spendingv13": 515706,
        "t1s_sptrinvisning": 515707,
        "t1s_cosphitrf1": 515699,
        "pou_60kvliniespend": 515683,
        "pou_belastning": 515684,
        "nex_60kvliniespend": 515674,
        "nex_belastning": 515675,
        "aki_aktiveffektny": 515634,
        "aki_reaktiveffektny": 515641,
        "aki_belastningillny": 515635,
        "aki_belastningil2ny": 515636,
        "aki_belastningil3ny": 515637,
        "aki_spendingvliny": 515643,
        "aki_spendingvl2ny": 515644,
        "aki_spendingvl3ny": 515645,
        "aki_spendingv112": 515642
    },
    "GUDHJEM": {
        "t1s_belastning": 515734,
        "t1s_sptrinvisning": 515740,
        "t1s_10kvskinnespend": 515733,
        "t1s_cosphitrf1": 515736
    },
    "NEXO": {
        "t1s_aktiveffekt": 515975,
        "t1s_reaktiveffekt": 515979,
        "t1p_belastningill": 515971,
        "t1p_belastningil2": 515972,
        "t1p_belastningil3": 515973,
        "t1s_belastning": 515976,
        "t1s_sptrinvisning": 515985,
        "t1s_spendingvl1": 515981,
        "t1s_spendingv12": 515982,
        "t1s_spendingv13": 515983,
        "t1s_cosphitrf1": 516007,
        "t2s_aktiveffekt": 515991,
        "t2s_reaktiveffekt": 515997,
        "t2p_belastningill": 515987,
        "t2p_belastningil2": 515988,
        "t2p_belastningil3": 515989,
        "t2s_sptrinvisning": 516003,
        "t2s_belastning": 515992,
        "t2s_spendingvl1": 515999,
        "t2s_spendingv12": 516000,
        "t2s_spendingv13": 516001,
        "t2s_cosphitrf2": 515994,
        "sva_60kvliniespend": 515968,
        "sva_belastning": 515969,
        "bod_60kvliniespend": 515945,
        "bod_belastning": 515946
    },
    "SVANEKE": {
        "t1s_aktiveffekt": 516303,
        "t1s_reaktiveffekt": 516308,
        "t1s_sptrinvisning": 516312,
        "t1s_belastning": 516304,
        "t1p_belastningill": 516300,
        "t1p_belastningil2": 516301,
        "t1p_belastningil3": 516302,
        "t1s_spendingvl1": 516309,
        "t1s_spendingv12": 516310,
        "t1s_spendingv13": 516311,
        "t1s_cosphitrf1": 516306,
        "dal_aktiveffektny": 516281,
        "dal_reaktiveffektny": 516287,
        "dal_spendingvliny": 516289,
        "dal_spendingvl2ny": 516290,
        "dal_spendingv13ny": 516291,
        "dal_belastningiliny": 516283,
        "dal_belastningil2ny": 516284,
        "dal_60kvliniespend": 516280,
        "nex_belastning": 516297,
        "nex_60kvliniespend": 516296
    },
    "POULSKER" : {
        "t1s_aktiveffekt": 516161,
        "t1s_reaktiveffekt": 516166,
        "t1s_sptrinvisning": 516170,
        "t1s_belastning": 516162,
        "t1s_cosphitrf1": 516164,
        "t1p_belastningil1": 516157,
        "t1p_belastningil2": 516158,
        "t1p_belastningil3": 516159,
        "t1s_spendingvl1": 516167,
        "t1s_spendingvl2": 516168,
        "t1s_spendingvl3": 516169,
        "t1s_10kvskinnespend": 516160,
        "bod_belastning": 516147,
        "bod_60kvliniespend": 516146
    }
}

def get_ids_by_substation(*substation: str) -> List[int]:
    """
    Return the datastream_id for a given substation and parameter using the metadata dictionary.
    If '*' is passed as a substation, it returns IDs for all substations.
    Returns an empty list if no IDs are found or if a specified substation is not in metadata.
    """
    ID_list = []
    
    # Check if '*' is among the requested substations
    if '*' in substation:
        # If '*' is present, collect IDs from all substations
        for sub_data in metadata.values():
            ID_list.extend(sub_data.values())
        # If '*' is the only argument, we are done.
        # If other specific substations are also provided with '*',
        # the behavior here is to get all, effectively ignoring others.
        # A more complex logic could be implemented if a mix of '*' and specific
        # substations should result in a union or difference.
        # For this request, '*' implies "all".
        return ID_list
    
    # If '*' is not present, process each specified substation
    for sub in substation:
        if sub in metadata:
            ID_list.extend(metadata[sub].values())
        else:
            print(f"Substation '{sub}' not found in metadata.")
            
    return ID_list


def get_datastream_metadata():
    """Return a dictionary mapping datastream_id to substation and parameter names."""
    return {
        # Allinge – two transformers
        515615: {"substation": "Allinge", "parameter": "t1s_cosphitrf1"},
        515623: {"substation": "Allinge", "parameter": "t2s_cosphitrf2"},
        515613: {"substation": "Allinge", "parameter": "t1s_belastning"},
        515621: {"substation": "Allinge", "parameter": "t2s_belastning"},
        515618: {"substation": "Allinge", "parameter": "t1s_sptrinvisning"},
        515626: {"substation": "Allinge", "parameter": "t2s_sptrinvisning"},
        516443: {"substation": "Allinge", "parameter": "lok_10kvskinnespend"},
        
        # Olsker – two transformers
        516063: {"substation": "Olsker", "parameter": "t1s_aktiveffekt"},
        516069: {"substation": "Olsker", "parameter": "t1s_reaktiveffekt"},
        516081: {"substation": "Olsker", "parameter": "t2s_aktiveffekt"},
        516087: {"substation": "Olsker", "parameter": "t2s_reaktiveeffekt"},
        516059: {"substation": "Olsker", "parameter": "t1p_belastningil1"},
        516060: {"substation": "Olsker", "parameter": "t1p_belastningil2"},
        516061: {"substation": "Olsker", "parameter": "t1p_belastningil3"},
        516064: {"substation": "Olsker", "parameter": "t1s_belastning"},
        516077: {"substation": "Olsker", "parameter": "t2p_belastningil1"},
        516078: {"substation": "Olsker", "parameter": "t2p_belastningil2"},
        516079: {"substation": "Olsker", "parameter": "t2p_belastningil3"},
        516082: {"substation": "Olsker", "parameter": "t2s_belastning"},
        516071: {"substation": "Olsker", "parameter": "t1s_spendingvl1"},
        516072: {"substation": "Olsker", "parameter": "t1s_spendingvl2"},
        516073: {"substation": "Olsker", "parameter": "t1s_spendingvl3"},
        516089: {"substation": "Olsker", "parameter": "t2s_spendingvl1"},
        516090: {"substation": "Olsker", "parameter": "t2s_spendingvl2"},
        516091: {"substation": "Olsker", "parameter": "t2s_spendingvl3"},
        516075: {"substation": "Olsker", "parameter": "t1s_sptrinvisning"},
        516093: {"substation": "Olsker", "parameter": "t2s_sptrinvisning"},
        516066: {"substation": "Olsker", "parameter": "t1s_cosphitrf1"},
        516084: {"substation": "Olsker", "parameter": "t2s_cosphitrf2"},
        516043: {"substation": "Olsker", "parameter": "ost_aktiveffektny"},
        516050: {"substation": "Olsker", "parameter": "ost_reaktiveffektny"},
        516022: {"substation": "Olsker", "parameter": "has_aktiveffektny"},
        516029: {"substation": "Olsker", "parameter": "has_reaktiveffektny"},
        516018: {"substation": "Olsker", "parameter": "all_belastning"},
        516017: {"substation": "Olsker", "parameter": "all_60kvliniespend"},
        516040: {"substation": "Olsker", "parameter": "lok_10kvskinnespend"},
        
        # Osterlars
        516134: {"substation": "Osterlars", "parameter": "t1s_aktiveffekt"},
        516139: {"substation": "Osterlars", "parameter": "t1s_reaktiveffekt"},
        516130: {"substation": "Osterlars", "parameter": "t1p_belastningil1"},
        516131: {"substation": "Osterlars", "parameter": "t1p_belastningil2"},
        516132: {"substation": "Osterlars", "parameter": "t1p_belastningil3"},
        516135: {"substation": "Osterlars", "parameter": "t1s_belastning"},
        516140: {"substation": "Osterlars", "parameter": "t1s_spendingvl1"},
        516141: {"substation": "Osterlars", "parameter": "t1s_spendingvl2"},
        516142: {"substation": "Osterlars", "parameter": "t1s_spendingvl3"},
        516143: {"substation": "Osterlars", "parameter": "t1s_sptrinvisning"},
        516137: {"substation": "Osterlars", "parameter": "t1s_cosphitrf1"},
        516133: {"substation": "Osterlars", "parameter": "t1s_10kvskinnespend"},
        516119: {"substation": "Osterlars", "parameter": "ols_aktiveffektny"},
        516124: {"substation": "Osterlars", "parameter": "ols_reaktiveffektny"},
        516105: {"substation": "Osterlars", "parameter": "dal_aktiveffektny"},
        516110: {"substation": "Osterlars", "parameter": "dal_reaktiveffektny"},
        516116: {"substation": "Osterlars", "parameter": "gud_belastning"},
        
        # Hasle – two transformers
        515910: {"substation": "Hasle", "parameter": "t1s_aktiveffekt"},
        515941: {"substation": "Hasle", "parameter": "t1s_reaktiveffekt"},
        515922: {"substation": "Hasle", "parameter": "t2s_aktiveffekt"},
        515927: {"substation": "Hasle", "parameter": "t2s_reaktiveffekt"},
        515905: {"substation": "Hasle", "parameter": "t1p_belastningil1"},
        515906: {"substation": "Hasle", "parameter": "t1p_belastningil2"},
        515907: {"substation": "Hasle", "parameter": "t1p_belastningil3"},
        515940: {"substation": "Hasle", "parameter": "t1s_belastning"},
        515918: {"substation": "Hasle", "parameter": "t2p_belastningil1"},
        515919: {"substation": "Hasle", "parameter": "t2p_belastningil2"},
        515920: {"substation": "Hasle", "parameter": "t2p_belastningil3"},
        515923: {"substation": "Hasle", "parameter": "t2s_belastning"},
        515912: {"substation": "Hasle", "parameter": "t1s_spendingvl1"},
        515913: {"substation": "Hasle", "parameter": "t1s_spendingvl2"},
        515914: {"substation": "Hasle", "parameter": "t1s_spendingvl3"},
        515929: {"substation": "Hasle", "parameter": "t2s_spendingvl1"},
        515930: {"substation": "Hasle", "parameter": "t2s_spendingvl2"},
        515931: {"substation": "Hasle", "parameter": "t2s_spendingvl3"},
        515916: {"substation": "Hasle", "parameter": "t1s_sptrinvisning"},
        515933: {"substation": "Hasle", "parameter": "t2s_sptrinvisning"},
        516445: {"substation": "Hasle", "parameter": "t1s_trp1pf"},
        516446: {"substation": "Hasle", "parameter": "t2s_trp2pf"},
        515900: {"substation": "Hasle", "parameter": "sok_mwbelastning"},
        515898: {"substation": "Hasle", "parameter": "sok_mvarbelastning"},
        515869: {"substation": "Hasle", "parameter": "sno_aktiveffekt"},
        515884: {"substation": "Hasle", "parameter": "sno_reaktiveffekt"},
        515763: {"substation": "Hasle", "parameter": "ols_aktiveffektny"},
        515841: {"substation": "Hasle", "parameter": "ols_reaktiveffektny"},
        515756: {"substation": "Hasle", "parameter": "lok_belastning"},
        
        # Snorrebakken
        516257: {"substation": "Snorrebakken", "parameter": "t1s_aktiveffekt"},
        516262: {"substation": "Snorrebakken", "parameter": "t1s_reaktiveffekt"},
        516253: {"substation": "Snorrebakken", "parameter": "t1p_belastningil1"},
        516254: {"substation": "Snorrebakken", "parameter": "t1p_belastningil2"},
        516255: {"substation": "Snorrebakken", "parameter": "t1p_belastningil3"},
        516258: {"substation": "Snorrebakken", "parameter": "t1s_belastning"},
        516263: {"substation": "Snorrebakken", "parameter": "t1s_spendingvl1"},
        516264: {"substation": "Snorrebakken", "parameter": "t1s_spendingvl2"},
        516265: {"substation": "Snorrebakken", "parameter": "t1s_spendingvl3"},
        516266: {"substation": "Snorrebakken", "parameter": "t1s_sptrinvisning"},
        516260: {"substation": "Snorrebakken", "parameter": "t1s_cosphitrf1"},
        516256: {"substation": "Snorrebakken", "parameter": "t1s_10kvskinnespend"},
        516267: {"substation": "Snorrebakken", "parameter": "var_aktiveffektny"},
        516272: {"substation": "Snorrebakken", "parameter": "var_reaktiveffektny"},
        516236: {"substation": "Snorrebakken", "parameter": "has_aktiveffektny"},
        516241: {"substation": "Snorrebakken", "parameter": "has_reaktiveffektny"},
        
        # Vaerket – 2 transformers
        516332: {"substation": "Vaerket", "parameter": "k01_effekt"},
        516333: {"substation": "Vaerket", "parameter": "k01_reaktiveffekt"},
        516329: {"substation": "Vaerket", "parameter": "k01_10kvspending"},
        516330: {"substation": "Vaerket", "parameter": "k01_belastning"},
        516331: {"substation": "Vaerket", "parameter": "k01_cosphi"},
        516338: {"substation": "Vaerket", "parameter": "k02_effekt"},
        516394: {"substation": "Vaerket", "parameter": "k02_reaktiveffekt"},
        516334: {"substation": "Vaerket", "parameter": "k02_10kvskcspending"},
        516335: {"substation": "Vaerket", "parameter": "k02_10kvspending"},
        516336: {"substation": "Vaerket", "parameter": "k02_belastning"},
        516337: {"substation": "Vaerket", "parameter": "k02_cosphi"},
        516343: {"substation": "Vaerket", "parameter": "k04_effekt"},
        516344: {"substation": "Vaerket", "parameter": "k04_reaktiveffekt"},
        516340: {"substation": "Vaerket", "parameter": "k04_10kvspending"},
        516341: {"substation": "Vaerket", "parameter": "k04_belastning"},
        516342: {"substation": "Vaerket", "parameter": "k04_cosphi"},
        516316: {"substation": "Vaerket", "parameter": "bl5_mwblok5"},
        516317: {"substation": "Vaerket", "parameter": "bl6_blok6el"},
        516318: {"substation": "Vaerket", "parameter": "bl6_blok6fjernvarme"},
        516319: {"substation": "Vaerket", "parameter": "blok5_mw"},
        516327: {"substation": "Vaerket", "parameter": "fel_verkettotalmw"},
        516383: {"substation": "Vaerket", "parameter": "t1p_belastning"},
        516384: {"substation": "Vaerket", "parameter": "t2_sp60kvbelastning"},
        516385: {"substation": "Vaerket", "parameter": "t2_sptrinvisning"},
        516386: {"substation": "Vaerket", "parameter": "t5p_belastning"},
        516387: {"substation": "Vaerket", "parameter": "t6p_60kvliniespend"},
        516388: {"substation": "Vaerket", "parameter": "t6p_belastning"},
        516328: {"substation": "Vaerket", "parameter": "has_spendingvl12"},
        516389: {"substation": "Vaerket", "parameter": "ves_60kvliniespend"},
        516390: {"substation": "Vaerket", "parameter": "ves_belastning"},
        516371: {"substation": "Vaerket", "parameter": "sno_aktiveffektny"},
        516377: {"substation": "Vaerket", "parameter": "sno_reaktiveffektny"},
        516370: {"substation": "Vaerket", "parameter": "sno_60kvliniespend"},
        516378: {"substation": "Vaerket", "parameter": "sno_spendingvl12"},
        516372: {"substation": "Vaerket", "parameter": "sno_belastning"},
        516348: {"substation": "Vaerket", "parameter": "nli_aktiveffekt"},
        516355: {"substation": "Vaerket", "parameter": "nli_reaktiveffekt"},
        516366: {"substation": "Vaerket", "parameter": "rsy_belastning"},
        516365: {"substation": "Vaerket", "parameter": "rsy_60kvliniespend"},
        516392: {"substation": "Vaerket", "parameter": "via_belastning"},
        516391: {"substation": "Vaerket", "parameter": "via_60kvliniespend"},
        
        # Vesthavnen
        795792: {"substation": "Vesthavnen", "parameter": "t1s_belastning"},
        795797: {"substation": "Vesthavnen", "parameter": "t1s_sptrinvisning"},
        795793: {"substation": "Vesthavnen", "parameter": "t1s_cosphitrf1"},
        795796: {"substation": "Vesthavnen", "parameter": "rno_liniespending"},
        795791: {"substation": "Vesthavnen", "parameter": "rno_belastning"},
        795794: {"substation": "Vesthavnen", "parameter": "var_belastning"},
        795798: {"substation": "Vesthavnen", "parameter": "var_liniespending"},
        
        # Viadukten
        516427: {"substation": "Viadukten", "parameter": "t1s_belastning"},
        516430: {"substation": "Viadukten", "parameter": "t1s_sptrinvisning"},
        516429: {"substation": "Viadukten", "parameter": "t1s_cosphitrf1"},
        516426: {"substation": "Viadukten", "parameter": "t1s_10kvskinnespend"},
        516432: {"substation": "Viadukten", "parameter": "var_60kvliniespend"},
        516433: {"substation": "Viadukten", "parameter": "var_belastning"},
        
        # Rønne Syd
        516224: {"substation": "Rønne Syd", "parameter": "t1s_belastning"},
        516227: {"substation": "Rønne Syd", "parameter": "t1s_sptrinvisning"},
        516226: {"substation": "Rønne Syd", "parameter": "t1s_cosphitrf1"},
        516220: {"substation": "Rønne Syd", "parameter": "lok_10kvskinnespend"},
        516196: {"substation": "Rønne Syd", "parameter": "aki_aktiveffektny"},
        516207: {"substation": "Rønne Syd", "parameter": "aki_reaktiveffektny"},
        516193: {"substation": "Rønne Syd", "parameter": "aki_60kvliniebelastning"},
        516194: {"substation": "Rønne Syd", "parameter": "aki_60kvliniespend"},
        516228: {"substation": "Rønne Syd", "parameter": "var_60kvliniebelastning"},
        516229: {"substation": "Rønne Syd", "parameter": "var_60kvliniespend"},
        516231: {"substation": "Rønne Syd", "parameter": "via_60kvliniespend"},
        516230: {"substation": "Rønne Syd", "parameter": "via_60kvliniebelastning"},
        
        # Rønne Nord
        516183: {"substation": "Rønne Nord", "parameter": "t1s_belastning"},
        516186: {"substation": "Rønne Nord", "parameter": "t1s_sptrinvisning"},
        516185: {"substation": "Rønne Nord", "parameter": "t1s_cosphitrf1"},
        516188: {"substation": "Rønne Nord", "parameter": "via_belastning"},
        516187: {"substation": "Rønne Nord", "parameter": "via_60kvliniespend"},
        516178: {"substation": "Rønne Nord", "parameter": "has_belastning"},
        516177: {"substation": "Rønne Nord", "parameter": "has_60kvliniespend"},
        
        # Aakirkeby – two transformers
        515527: {"substation": "Aakirkeby", "parameter": "t1s_effekt"},
        515529: {"substation": "Aakirkeby", "parameter": "t1s_reaktiveffekt"},
        515522: {"substation": "Aakirkeby", "parameter": "t1s_belastning"},
        515525: {"substation": "Aakirkeby", "parameter": "t1s_cosphitrf1"},
        515531: {"substation": "Aakirkeby", "parameter": "t1s_sptrinvisning"},
        515530: {"substation": "Aakirkeby", "parameter": "t1s_spending"},
        515518: {"substation": "Aakirkeby", "parameter": "t1p_belastning"},
        515520: {"substation": "Aakirkeby", "parameter": "t1p_liniespending"},
        515543: {"substation": "Aakirkeby", "parameter": "t2s_effekt"},
        515545: {"substation": "Aakirkeby", "parameter": "t2s_reaktiveffekt"},
        515534: {"substation": "Aakirkeby", "parameter": "t2p_belastning"},
        515541: {"substation": "Aakirkeby", "parameter": "t2s_cosphitrf2"},
        515546: {"substation": "Aakirkeby", "parameter": "t2s_spending"},
        515537: {"substation": "Aakirkeby", "parameter": "t2p_liniespending"},
        515539: {"substation": "Aakirkeby", "parameter": "t2s_belastning"},
        515547: {"substation": "Aakirkeby", "parameter": "t2s_sptrinvisning"},
        515441: {"substation": "Aakirkeby", "parameter": "bod_aktiveffektny"},
        515447: {"substation": "Aakirkeby", "parameter": "bod_reaktiveffektny"},
        515449: {"substation": "Aakirkeby", "parameter": "bod_spendingvl1ny"},
        515450: {"substation": "Aakirkeby", "parameter": "bod_spendingvl2ny"},
        515451: {"substation": "Aakirkeby", "parameter": "bod_spendingvl3ny"},
        515442: {"substation": "Aakirkeby", "parameter": "bod_belastningil1ny"},
        515443: {"substation": "Aakirkeby", "parameter": "bod_belastningil2ny"},
        515444: {"substation": "Aakirkeby", "parameter": "bod_belastningil3ny"},
        515486: {"substation": "Aakirkeby", "parameter": "rsy_aktiveffektny"},
        515492: {"substation": "Aakirkeby", "parameter": "rsy_reaktiveffektny"},
        515487: {"substation": "Aakirkeby", "parameter": "rsy_belastningil1ny"},
        515488: {"substation": "Aakirkeby", "parameter": "rsy_belastningil2ny"},
        515489: {"substation": "Aakirkeby", "parameter": "rsy_belastningil3ny"},
        515494: {"substation": "Aakirkeby", "parameter": "rsy_spendingvl1ny"},
        515495: {"substation": "Aakirkeby", "parameter": "rsy_spendingvl2ny"},
        515496: {"substation": "Aakirkeby", "parameter": "rsy_spendingvl3ny"},
        
        # Bodilsker
        515696: {"substation": "Bodilsker", "parameter": "t1s_aktiveffekt"},
        515702: {"substation": "Bodilsker", "parameter": "t1s_reaktiveffekt"},
        516444: {"substation": "Bodilsker", "parameter": "lok_10kvskinnespend"},
        515692: {"substation": "Bodilsker", "parameter": "t1p_belastningil1"},
        515693: {"substation": "Bodilsker", "parameter": "t1p_belastningil2"},
        515694: {"substation": "Bodilsker", "parameter": "t1p_belastningil3"},
        515697: {"substation": "Bodilsker", "parameter": "t1s_belastning"},
        515704: {"substation": "Bodilsker", "parameter": "t1s_spendingvl1"},
        515705: {"substation": "Bodilsker", "parameter": "t1s_spendingvl2"},
        515706: {"substation": "Bodilsker", "parameter": "t1s_spendingvl3"},
        515707: {"substation": "Bodilsker", "parameter": "t1s_sptrinvisning"},
        515699: {"substation": "Bodilsker", "parameter": "t1s_cosphitrf1"},
        515683: {"substation": "Bodilsker", "parameter": "pou_60kvliniespend"},
        515684: {"substation": "Bodilsker", "parameter": "pou_belastning"},
        515674: {"substation": "Bodilsker", "parameter": "nex_60kvliniespend"},
        515675: {"substation": "Bodilsker", "parameter": "nex_belastning"},
        515634: {"substation": "Bodilsker", "parameter": "aki_aktiveffektny"},
        515641: {"substation": "Bodilsker", "parameter": "aki_reaktiveffektny"},
        515635: {"substation": "Bodilsker", "parameter": "aki_belastningil1ny"},
        515636: {"substation": "Bodilsker", "parameter": "aki_belastningil2ny"},
        515637: {"substation": "Bodilsker", "parameter": "aki_belastningil3ny"},
        515643: {"substation": "Bodilsker", "parameter": "aki_spendingvl1ny"},
        515644: {"substation": "Bodilsker", "parameter": "aki_spendingvl2ny"},
        515645: {"substation": "Bodilsker", "parameter": "aki_spendingvl3ny"},
        515642: {"substation": "Bodilsker", "parameter": "aki_spendingvl12"},
        
        # Gudhjem
        515734: {"substation": "Gudhjem", "parameter": "t1s_belastning"},
        515740: {"substation": "Gudhjem", "parameter": "t1s_sptrinvisning"},
        515733: {"substation": "Gudhjem", "parameter": "t1s_10kvskinnespend"},
        515736: {"substation": "Gudhjem", "parameter": "t1s_cosphitrf1"},
        
        # Nexo – 2 transformer
        515975: {"substation": "Nexo", "parameter": "t1s_aktiveffekt"},
        515979: {"substation": "Nexo", "parameter": "t1s_reaktiveffekt"},
        515971: {"substation": "Nexo", "parameter": "t1p_belastningil1"},
        515972: {"substation": "Nexo", "parameter": "t1p_belastningil2"},
        515973: {"substation": "Nexo", "parameter": "t1p_belastningil3"},
        515976: {"substation": "Nexo", "parameter": "t1s_belastning"},
        515985: {"substation": "Nexo", "parameter": "t1s_sptrinvisning"},
        515981: {"substation": "Nexo", "parameter": "t1s_spendingvl1"},
        515982: {"substation": "Nexo", "parameter": "t1s_spendingvl2"},
        515983: {"substation": "Nexo", "parameter": "t1s_spendingvl3"},
        516007: {"substation": "Nexo", "parameter": "t1s_cosphitrf1"},
        515991: {"substation": "Nexo", "parameter": "t2s_aktiveffekt"},
        515997: {"substation": "Nexo", "parameter": "t2s_reaktiveffekt"},
        515987: {"substation": "Nexo", "parameter": "t2p_belastningil1"},
        515988: {"substation": "Nexo", "parameter": "t2p_belastningil2"},
        515989: {"substation": "Nexo", "parameter": "t2p_belastningil3"},
        516003: {"substation": "Nexo", "parameter": "t2s_sptrinvisning"},
        515992: {"substation": "Nexo", "parameter": "t2s_belastning"},
        515999: {"substation": "Nexo", "parameter": "t2s_spendingvl1"},
        516000: {"substation": "Nexo", "parameter": "t2s_spendingvl2"},
        516001: {"substation": "Nexo", "parameter": "t2s_spendingvl3"},
        515994: {"substation": "Nexo", "parameter": "t2s_cosphitrf2"},
        515968: {"substation": "Nexo", "parameter": "sva_60kvliniespend"},
        515969: {"substation": "Nexo", "parameter": "sva_belastning"},
        515945: {"substation": "Nexo", "parameter": "bod_60kvliniespend"},
        515946: {"substation": "Nexo", "parameter": "bod_belastning"},
        
        # Svaneke
        516303: {"substation": "Svaneke", "parameter": "t1s_aktiveffekt"},
        516308: {"substation": "Svaneke", "parameter": "t1s_reaktiveffekt"},
        516312: {"substation": "Svaneke", "parameter": "t1s_sptrinvisning"},
        516304: {"substation": "Svaneke", "parameter": "t1s_belastning"},
        516300: {"substation": "Svaneke", "parameter": "t1p_belastningil1"},
        516301: {"substation": "Svaneke", "parameter": "t1p_belastningil2"},
        516302: {"substation": "Svaneke", "parameter": "t1p_belastningil3"},
        516309: {"substation": "Svaneke", "parameter": "t1s_spendingvl1"},
        516310: {"substation": "Svaneke", "parameter": "t1s_spendingvl2"},
        516311: {"substation": "Svaneke", "parameter": "t1s_spendingvl3"},
        516306: {"substation": "Svaneke", "parameter": "t1s_cosphitrf1"},
        516281: {"substation": "Svaneke", "parameter": "dal_aktiveffektny"},
        516287: {"substation": "Svaneke", "parameter": "dal_reaktiveffektny"},
        516289: {"substation": "Svaneke", "parameter": "dal_spendingvl1ny"},
        516290: {"substation": "Svaneke", "parameter": "dal_spendingvl2ny"},
        516291: {"substation": "Svaneke", "parameter": "dal_spendingvl3ny"},
        516283: {"substation": "Svaneke", "parameter": "dal_belastningil1ny"},
        516284: {"substation": "Svaneke", "parameter": "dal_belastningil2ny"},
        516280: {"substation": "Svaneke", "parameter": "dal_belastningil3ny"},
        516297: {"substation": "Svaneke", "parameter": "dal_spendingvl12"},
        516296: {"substation": "Svaneke", "parameter": "dal_belastning"},

        # Poulsker
        516161: {"substation": "Poulsker", "parameter": "t1s_aktiveffekt"},
        516166: {"substation": "Poulsker", "parameter": "t1s_reaktiveffekt"},
        516170: {"substation": "Poulsker", "parameter": "t1s_sptrinvisning"},
        516162: {"substation": "Poulsker", "parameter": "t1s_belastning"},
        516164: {"substation": "Poulsker", "parameter": "t1s_cosphitrf1"},
        516157: {"substation": "Poulsker", "parameter": "t1p_belastningil1"},
        516158: {"substation": "Poulsker", "parameter": "t1p_belastningil2"},
        516159: {"substation": "Poulsker", "parameter": "t1p_belastningil3"},
        516167: {"substation": "Poulsker", "parameter": "t1s_spendingvl1"},
        516168: {"substation": "Poulsker", "parameter": "t1s_spendingvl2"},
        516169: {"substation": "Poulsker", "parameter": "t1s_spendingvl3"},
        516160: {"substation": "Poulsker", "parameter": "t1s_10kvskinnespend"},
        516147: {"substation": "Poulsker", "parameter": "bod_belastning"},
        516146: {"substation": "Poulsker", "parameter": "bod_60kvliniespend"}
    } 


def group_ids_by_substation():
    """Group datastream IDs by substation from the metadata DataFrame.
    Returns a dictionary where keys are substation names and values are lists of datastream IDs.
    """
    metadata_df = pd.DataFrame.from_dict(get_datastream_metadata(), orient='index')
    return metadata_df.groupby("substation")["id"].apply(list).to_dict()

def get_id_by_substation_and_parameter(substation: str, parameter: str) -> str:
    """
    Return the datastream_id for a given substation and parameter using the metadata dictionary.
    Returns None if not found.
    """
    metadata = get_datastream_metadata()
    # Build a reverse mapping: (substation, parameter) -> id
    reverse_map = { (v['substation'], v['parameter']): k for k, v in metadata.items() }
    return reverse_map.get((substation, parameter))

def define_timespan(start: str, end: str):
    date_obj = datetime.strptime(start, '%Y-%m-%d')
    start = date_obj.strftime('%Y-%m-%dT00:00:00')
    date_obj = datetime.strptime(end, '%Y-%m-%d')
    end = date_obj.strftime('%Y-%m-%dT00:00:00')
    return start, end

def convert_datatypes(df, verbose=True):
    """
    Convert dataframe columns to appropriate data types for EDDK transformer data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: datastream_id, value, substation, parameter, timestamp
    verbose : bool, optional (default=True)
        If True, print conversion summary
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted data types
        
    Raises:
    -------
    ValueError
        If required columns are missing

    """
    # Check if required columns exist
    required_columns = ["datastream_id", "value", "substation", "parameter", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert data types
    try:
        df = df.astype({
            "datastream_id": "int",
            "value": "float",
            "substation": "string",
            "parameter": "string"
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValueError(f"Error converting data types: {e}")
    
    return df

def fetch_timespan_values(token: str, datastream_ids: List[int], start, end) -> List[dict]:
    """
    Fetch datastream values for a specific time period from Energy Data Service API.
    
    Args:
        token (str): API authentication token
        datastream_ids (List[int]): List of datastream IDs to fetch
        start (str): Start datetime in ISO format (e.g., "2022-08-01T00:00:00")
        end (str): End datetime in ISO format (e.g., "2022-08-02T00:00:00")
    
    Returns:
        List[dict]: List of records with keys: datastream_id, timestamp, value, 
                    substation, parameter. Returns empty list on error.
    
    Example:
        data = fetch_timespan_values(token, [1205512, 1205513], 
                                    "2022-08-01T00:00:00", "2022-08-02T00:00:00")
    """
    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "from": start,
        "to": end
    }
    
    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "from": start,
        "to": end
    }

    #print("Fetching data from URL:", url)                              # Debugging line, can be removed later
    #print("Headers:", headers)
    #print("Parameters:", params)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print("Status Code:", response.status_code)
        #print("Response Text (first 500 chars):", response.text[:500]) # Debugging line, can be removed later
        response.raise_for_status()

    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
    except Exception as e:
        print("HTTP Request failed:", e)
        return []  # Ensures an empty list, not None

    metadata = get_datastream_metadata()
    result = []

    csv_file = StringIO(response.text)
    reader = csv.reader(csv_file)

    for row in reader:
        if len(row) == 3:
            datastream_id, timestamp, value = row
            try:
                datastream_id = int(datastream_id)
                value = float(value)
                timestamp = timestamp.strip()
            except ValueError as ve:
                print("Skipping malformed row:", row, "| Error:", ve)
                continue

            substation = metadata.get(datastream_id, {}).get("substation", "Unknown")
            parameter = metadata.get(datastream_id, {}).get("parameter", "Unknown")

            #print(f"Substation: {substation}, Parameter: {parameter}, Value: {value}, Timestamp: {timestamp}")
            result.append({
                "datastream_id": datastream_id,
                "timestamp": timestamp,
                "value": value,
                "substation": substation,
                "parameter": parameter
            })

    return result

def shift_timespan(data: List[dict], new_start: str) -> List[dict]:
    """
    Shift all timestamps in the data so they begin at new_start,
    preserving the original relative time intervals.

    Args:
        data (List[dict]): List of records with 'timestamp' field in ISO format.
        new_start (str): New start timestamp (e.g., '2023-01-01T00:00:00').

    Returns:
        List[dict]: Modified list with shifted timestamps.
    """
    if not data:
        return []

    # Parse original and new start times
    original_start = datetime.fromisoformat(data[0]['timestamp'])
    new_start_dt = datetime.fromisoformat(new_start)

    # Compute time delta
    delta = new_start_dt - original_start

    # Shift all timestamps
    for entry in data:
        original_ts = datetime.fromisoformat(entry['timestamp'])
        new_ts = original_ts + delta
        entry['timestamp'] = new_ts.isoformat()

    print(f"Shifted {len(data)} timestamps to start from {new_start}")
    return data


def store_data(df: pd.DataFrame, filename: str, path: str = "./Data/Stage") -> None:
    """
    Stores the DataFrame as a CSV file in the specified path.

    Args:
        df (pd.DataFrame): The data to store.
        filename (str): Name of the output CSV file (without extension).
        path (str): Directory to store the file. Defaults to './Data/Stage'.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}.csv")

    try:
        df.to_csv(full_path, index=False)
        print(f"Data stored at: {full_path} ({len(df)} rows)")
    except Exception as e:
        print(f"Failed to store data to {full_path}:", e)


#############################################
### Transformer metadata functions
#############################################


def get_datastream_metadata_transformer_p_c():
    """Return a dictionary mapping datastream_id to substation and parameter names."""
    return {

        1205489: {"substation": "Gudhjem", "parameter": "production"},
        1205490: {"substation": "Hasle", "parameter": "consumption"},
        1205491: {"substation": "Hasle", "parameter": "production"},
        1205492: {"substation": "Nexoe", "parameter": "consumption"},
        1205493: {"substation": "Nexoe", "parameter": "production"},
        1205494: {"substation": "Olsker", "parameter": "consumption"},
        1205495: {"substation": "Olsker", "parameter": "production"},
        1205496: {"substation": "Povlsker", "parameter": "consumption"},
        1205497: {"substation": "Povlsker", "parameter": "production"},
        1205498: {"substation": "Roenne Nord", "parameter": "consumption"},
        1205499: {"substation": "Roenne Nord", "parameter": "production"},
        1205500: {"substation": "Roenne Syd", "parameter": "consumption"},
        1205501: {"substation": "Roenne Syd", "parameter": "production"},
        1205502: {"substation": "Snorrebakken", "parameter": "consumption"},
        1205503: {"substation": "Snorrebakken", "parameter": "production"},
        1205504: {"substation": "Svaneke", "parameter": "consumption"},
        1205505: {"substation": "Svaneke", "parameter": "production"},
        1205506: {"substation": "Vesthavnen", "parameter": "consumption"},
        1205507: {"substation": "Vesthavnen", "parameter": "production"},
        1205508: {"substation": "Viadukten", "parameter": "consumption"},
        1205509: {"substation": "Viadukten", "parameter": "production"},
        1205510: {"substation": "Vaerket", "parameter": "consumption"},
        1205511: {"substation": "Vaerket", "parameter": "production"},
        1205512: {"substation": "aakirkeby", "parameter": "consumption"},
        1205513: {"substation": "aakirkeby", "parameter": "production"},
        1205514: {"substation": "oesterlars", "parameter": "consumption"},
        1205515: {"substation": "oesterlars", "parameter": "production"},
        1205516: {"substation": "Allinge", "parameter": "consumption"},
        1205517: {"substation": "Allinge", "parameter": "production"},
        1205518: {"substation": "Bodilsker", "parameter": "consumption"},
        1205519: {"substation": "Bodilsker", "parameter": "production"},
        1205520: {"substation": "Gudhjem", "parameter": "consumption"}
    }


def fetch_timespan_values_transformer_p_c(token: str, datastream_ids: List[int], start, end) -> List[dict]:
    """
    Fetch datastream values for a specific time period from 
    Energy Data Service API transformer production and consumption dataset.
    
    Args:
        token (str): API authentication token
        datastream_ids (List[int]): List of datastream IDs to fetch
        start (str): Start datetime in ISO format (e.g., "2022-08-01T00:00:00")
        end (str): End datetime in ISO format (e.g., "2022-08-02T00:00:00")
    
    Returns:
        List[dict]: List of records with keys: datastream_id, timestamp, value, 
                    substation, parameter. Returns empty list on error.
    """

    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "from": start,
        "to": end
    }

    #print("Fetching data from URL:", url)                              # Debugging line, can be removed later
    #print("Headers:", headers)
    #print("Parameters:", params)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print("Status Code:", response.status_code)
        #print("Response Text (first 500 chars):", response.text[:500]) # Debugging line, can be removed later
        response.raise_for_status()

    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
    except Exception as e:
        print("HTTP Request failed:", e)
        return []  # Ensures an empty list, not None

    metadata = get_datastream_metadata_transformer_p_c()
    result = []

    csv_file = StringIO(response.text)
    reader = csv.reader(csv_file)

    for row in reader:
        if len(row) == 3:
            datastream_id, timestamp, value = row
            try:
                datastream_id = int(datastream_id)
                value = float(value)
                timestamp = timestamp.strip()
            except ValueError as ve:
                print("Skipping malformed row:", row, "| Error:", ve)
                continue

            substation = metadata.get(datastream_id, {}).get("substation", "Unknown")
            parameter = metadata.get(datastream_id, {}).get("parameter", "Unknown")

            #print(f"Substation: {substation}, Parameter: {parameter}, Value: {value}, Timestamp: {timestamp}")
            result.append({
                "datastream_id": datastream_id,
                "timestamp": timestamp,
                "value": value,
                "substation": substation,
                "parameter": parameter
            })

    return result

def fetch_latest_values_transformer_p_c(token: str, datastream_ids: List[int]) -> List[dict]:
    """
    Fetch the latest (most recent) values from transformer production 
    and consumption datastreams.
    
    Args:
        token (str): API authentication token
        datastream_ids (List[int]): List of datastream IDs to fetch latest values for
    
    Returns:
        List[dict]: List of records with keys: datastream_id, timestamp, value, 
                    substation, parameter. Returns empty list on error.
    """
    
    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "latest": "true"
    }

    #print("Fetching data from URL:", url)                              # Debugging line, can be removed later
    #print("Headers:", headers)
    #print("Parameters:", params)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print("Status Code:", response.status_code)
        #print("Response Text (first 500 chars):", response.text[:500]) # Debugging line, can be removed later
        response.raise_for_status()

    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
    except Exception as e:
        print("HTTP Request failed:", e)
        return []  # Ensures an empty list, not None

    metadata = get_datastream_metadata_transformer_p_c()
    result = []

    csv_file = StringIO(response.text)
    reader = csv.reader(csv_file)

    for row in reader:
        if len(row) == 3:
            datastream_id, timestamp, value = row
            try:
                datastream_id = int(datastream_id)
                value = float(value)
                timestamp = timestamp.strip()
            except ValueError as ve:
                print("Skipping malformed row:", row, "| Error:", ve)
                continue

            substation = metadata.get(datastream_id, {}).get("substation", "Unknown")
            parameter = metadata.get(datastream_id, {}).get("parameter", "Unknown")

            #print(f"Substation: {substation}, Parameter: {parameter}, Value: {value}, Timestamp: {timestamp}")
            result.append({
                "datastream_id": datastream_id,
                "timestamp": timestamp,
                "value": value,
                "substation": substation,
                "parameter": parameter
            })

    return result