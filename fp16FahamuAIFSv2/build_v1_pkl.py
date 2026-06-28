#!/usr/bin/env python3
"""Minimal AIFS-ENS-1.0 (v1) input-state pkl builder — member 1, no upload.
Self-contained (v1 contract: 92 fields, no wave/sd/swvl, 13 levels)."""
import datetime, os, pickle, sys
from collections import defaultdict
import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from ecmwf.opendata import Client as OpendataClient

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]
SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2"}
G = 9.80665

def get_open_data(date, param, levelist=[], number=None, constant=False):
    fields = defaultdict(list)
    dates = [date] if constant else [date - datetime.timedelta(hours=6), date]
    for d in dates:
        if number is None:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param, levelist=levelist)
        else:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, number=[number], stream="enfo")
        for f in data:
            if not levelist and f.metadata("levtype") == "pl":
                continue
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist \
                else f.metadata("param")
            fields[name].append(values)
            if constant:
                fields[name].append(values)
    for name, values in fields.items():
        fields[name] = np.stack(values)
    return fields

date = OpendataClient("ecmwf").latest()
member = 1
print(f"v1 input prep | init {date} | member {member}", flush=True)
fields = {}
fields.update(get_open_data(date, param=PARAM_SFC, number=member))
fields.update(get_open_data(date, param=PARAM_SFC_FC, constant=True))
soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS, number=member)
for k, v in soil.items():
    fields[SOIL_RENAME[k]] = v
fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS, number=member))
for level in LEVELS:
    fields[f"z_{level}"] = fields.pop(f"gh_{level}") * G

state = dict(date=date, fields=fields)
shape = next(iter(fields.values())).shape
print(f"done — {len(fields)} fields, shape {shape}", flush=True)
os.makedirs("/scratch/input_states_v1", exist_ok=True)
out = "/scratch/input_states_v1/input_state_member_001.pkl"
with open(out, "wb") as fh:
    pickle.dump(state, fh)
print(f"saved {out}", flush=True)
