"""Measured response-field interpolation from a handful of known surface points."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay, QhullError
from .config import DATA_DIR

SURFACE_DIR = DATA_DIR / 'surface_calibration'


class SurfaceCalibration:
    def __init__(self, scales, directory=SURFACE_DIR):
        samples = {}
        trials = []
        for path in sorted(Path(directory).glob('*.json')):
            row = json.loads(path.read_text())
            wave = np.load(Path(directory) / row['file'], allow_pickle=False).astype(float)
            if wave.ndim != 2 or wave.shape[1] != len(scales) or len(wave) < 72 or not np.isfinite(wave).all():
                raise ValueError(f'Invalid surface recording: {path.name}')
            responses = []
            noise = np.maximum(1.4826*np.median(abs(wave[:40]-np.median(wave[:40],axis=0)),axis=0),0.5)
            for start in range(40,len(wave)-31,16):
                segment=wave[start:start+32]
                segment=segment-np.median(segment,axis=0)
                rms=np.maximum(np.sqrt(np.mean(segment**2,axis=0))-noise,0)/scales
                responses.append(rms)
            responses=np.asarray(responses)
            levels=np.linalg.norm(responses,axis=1)
            keep=responses[levels>=np.percentile(levels,65)]
            if np.max(levels)<1e-6:
                continue
            signature=np.median(keep/(np.linalg.norm(keep,axis=1,keepdims=True)+1e-12),axis=0)
            uv=(float(row['u']),((float(row['v'])+0.5)%1)-0.5)
            samples.setdefault(uv,[]).append(signature)
            trials.append((uv,signature))
        self.signatures=self.coordinates=None
        self.validation_error=None
        if len(samples)<5:
            return
        points=np.asarray(list(samples))
        templates=np.stack([np.median(samples[tuple(point)],axis=0) for point in points])
        templates/=np.linalg.norm(templates,axis=1,keepdims=True)+1e-12
        try:
            triangulation=Delaunay(points)
        except QhullError:
            # Partial/collinear recordings cannot define a 2D surface yet.
            return
        coords,signatures=[],[]
        for indices in triangulation.simplices:
            for i in range(17):
                for j in range(17-i):
                    weights=np.array([i,j,16-i-j])/16
                    coords.append(weights@points[indices])
                    signature=weights@templates[indices]
                    signatures.append(signature/(np.linalg.norm(signature)+1e-12))
        self.coordinates=np.asarray(coords)
        self.signatures=np.asarray(signatures)
        # Collected responses can be ambiguous (for example two duplicated
        # electrical channels). Never let a response field that cannot recover
        # its own known points override the physical energy interpolation.
        errors=[]
        for expected,signature in trials:
            estimated=self._estimate(signature)
            du=estimated[0]-expected[0]
            dv=((estimated[1]-expected[1]+0.5)%1)-0.5
            errors.append(float(np.hypot(du,dv)))
        self.validation_error=float(np.median(errors))
        if self.validation_error>0.16 or np.mean(np.asarray(errors)>0.30)>0.25:
            self.coordinates=self.signatures=None

    def estimate(self, values):
        if self.signatures is None:
            return None
        return self._estimate(values)

    def _estimate(self, values):
        signature=values/(np.linalg.norm(values)+1e-12)
        distance=np.sum((self.signatures-signature)**2,axis=1)
        best=int(np.argmin(distance))
        # Interpolate only within the winning local surface neighborhood. Never
        # average two ambiguous opposite-side solutions through the torso.
        local=np.linalg.norm(self.coordinates-self.coordinates[best],axis=1)<0.18
        weight=np.exp(-(distance-np.min(distance))/0.025)*local
        coordinate=weight@self.coordinates/(np.sum(weight)+1e-12)
        return float(coordinate[0]),float(coordinate[1]%1)
