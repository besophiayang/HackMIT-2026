"""Sixteen short, known-position recordings; no full gesture recollection required."""
import argparse
import json
from uuid import uuid4
import numpy as np
from .body_map import BodySurfaceMap
from .config import HARDWARE_NUM_SENSORS, SERIAL_BAUD
from .source_serial import SerialTouchSource
from .surface_calibration import SURFACE_DIR


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default='COM5')
    parser.add_argument('--count',type=int,default=2)
    args=parser.parse_args()
    if args.count<1:
        parser.error('--count must be positive')
    points=[(p.name,p.u,p.v) for p in BodySurfaceMap.load().placements]
    points += [('left_middle',0.52,0.75),('right_middle',0.52,0.25),('top_middle',0.52,0.0)]
    print(f'{len(points)*args.count} recordings. Keep robot powered on. Close dashboard first.')
    print('At each point: one light tap on trial 1; a small gentle rub at the SAME spot on trial 2.')
    print('Use comparable strength. Do not stroke along the body in this calibration.')
    SURFACE_DIR.mkdir(parents=True,exist_ok=True)
    try:
        source=SerialTouchSource(args.port,SERIAL_BAUD,HARDWARE_NUM_SENSORS,900)
    except ConnectionError as error:
        raise SystemExit(f'Close the dashboard/Serial Monitor to release {args.port}. {error}') from None
    try:
        for name,u,v in points:
            for trial in range(args.count):
                input(f'{name}, trial {trial+1}/{args.count}: hand away, press Enter...')
                source.reset_input_buffer()
                wave=source.get_triggered_touch(10,baseline_samples=200,pretrigger_samples=40,
                    on_armed=lambda: print('GO: tap or rub the marked spot now.',flush=True))
                stem='point_'+uuid4().hex
                np.save(SURFACE_DIR/(stem+'.npy'),wave,allow_pickle=False)
                row=dict(file=stem+'.npy',name=name,u=u,v=v)
                (SURFACE_DIR/(stem+'.json')).write_text(json.dumps(row),encoding='utf-8')
                centered=wave-np.median(wave[:40],axis=0)
                rms=np.sqrt(np.mean(centered.astype(float)**2,axis=0))
                print('Saved. GPIO 4,5,6,7,15 RMS:',np.round(rms,1).tolist())
                for a,b in [(2,3),(1,4)]:
                    if min(rms[a],rms[b])>2 and np.corrcoef(centered[:,a],centered[:,b])[0,1]>0.98:
                        print(f'CHECK: channels {[4,5,6,7,15][a]} and {[4,5,6,7,15][b]} closely duplicate each other.')
    except KeyboardInterrupt:
        print('\nSaved trials are preserved; rerunning appends recordings.')
    finally:
        source.close()
    print('Restart dashboard. It loads the measured surface interpolation automatically.')


if __name__=='__main__':
    main()
