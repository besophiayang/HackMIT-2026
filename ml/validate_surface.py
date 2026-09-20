"""Reproducible recorded-data audit and physical-map regression checks."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import LABELS_PATH, RAW_DATA_DIR, DATA_DIR
from .fast_contact import FastContactGate
from .body_map import BodySurfaceMap
from .spatial_tracker import SpatialFingerprintTracker


def main():
    labels=pd.read_csv(LABELS_PATH)
    rows=labels[(labels.source=='real') & (labels.num_sensors==5)]
    waves=[np.load(RAW_DATA_DIR/f,allow_pickle=False) for f in rows.file]
    full=np.concatenate(waves).astype(float)
    report={'scope':'Replay of existing recordings, including training data; not a new held-out accuracy estimate.',
        'recordings':len(rows),'correlation':np.corrcoef(full.T).round(5).tolist(),
        'gpio6_gpio7_equal_fraction':float(np.mean(full[:,2]==full[:,3])),
        'gpio5_gpio15_equal_fraction':float(np.mean(full[:,1]==full[:,4]))}
    gate=FastContactGate.load()
    features=[]
    group=[]
    from .fast_contact import fast_contact_features
    for i,wave in enumerate(waves):
        idle=wave[:60].astype(float)
        noise=np.maximum(1.4826*np.median(abs(idle-np.median(idle,axis=0)),axis=0),0.5)
        for end in range(12,len(wave)+1,5):
            features.append(fast_contact_features(wave[end-12:end],noise))
            group.append(i)
    begin=time.perf_counter()
    probabilities=gate.model.predict_proba(np.asarray(features))[:,list(gate.model.classes_).index('touch')]
    report['batch_gate_evaluation_seconds']=time.perf_counter()-begin
    detections={}
    idle_duty=[]
    fast_detections={}
    fast_idle=[]
    for i,row in enumerate(rows.itertuples(index=False)):
        scores=probabilities[np.asarray(group)==i]
        p=scores>=gate.threshold
        runs=p[2:] & p[1:-1] & p[:-2]
        accelerated=runs | (scores[2:]>=0.97)
        if row.touch_type=='no_touch':
            idle_duty.extend(runs.tolist())
            fast_idle.extend(accelerated.tolist())
        else:
            detections.setdefault(row.location,[]).append(bool(np.any(runs)))
            fast_detections.setdefault(row.location,[]).append(bool(np.any(accelerated)))
    report['recording_detection_fraction']={name:float(np.mean(v)) for name,v in detections.items()}
    report['idle_positive_frame_fraction']=float(np.mean(idle_duty))
    report['high_confidence_onset_recording_fraction']={k:float(np.mean(v)) for k,v in fast_detections.items()}
    report['high_confidence_onset_idle_frame_fraction']=float(np.mean(fast_idle))
    tracker=SpatialFingerprintTracker.from_real_dataset(5,BodySurfaceMap.load())
    tracker.motion_model=None # isolate geometry tests from learned stroke predictions
    anchors=[]
    for placement in tracker.by_channel:
        tracker.reset()
        strengths=np.zeros(5)
        strengths[placement.channel]=20*tracker.sensor_scales[placement.channel]
        estimate=tracker.estimate(strengths)
        assert abs(estimate.u-placement.u)<1e-6
        assert abs(((estimate.v-placement.v+0.5)%1)-0.5)<1e-6
        anchors.append(dict(pin=placement.pin,u=estimate.u,v=estimate.v))
    report['isolated_channel_anchors']=anchors
    sweeps={}
    for side,front,rear in [('left',4,0),('right',3,1)]:
        path=[]
        for fraction in np.linspace(0,1,21):
            tracker.reset()
            values=np.zeros(5)
            values[front]=1-fraction
            values[rear]=fraction
            e=tracker.estimate(values*tracker.sensor_scales*20)
            path.append(e.u)
        assert np.min(np.diff(path))>=-0.005, (side,path)
        assert 0.35<path[10]<0.7
        sweeps[side]=np.round(path,4).tolist()
    report['synthetic_gain_corrected_interpolation']=sweeps
    # Single-frame prediction cost, not batch timings or an end-to-end claim.
    sample=waves[0][-12:]
    times=[]
    for _ in range(100):
        begin=time.perf_counter(); gate.probability(sample,np.ones(5))
        times.append((time.perf_counter()-begin)*1000)
    report['gate_ms_p50_p95']=np.percentile(times,[50,95]).tolist()
    output=DATA_DIR/'surface_validation.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Saved',output)


if __name__=='__main__':
    main()
