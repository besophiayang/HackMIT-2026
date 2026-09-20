"""Hardware-free regression tests: python -m unittest ml.test_live_surface."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
import numpy as np
from .live_stream import LatestSamples
from .source_serial import SerialTouchSource
from .surface_calibration import SurfaceCalibration
from .body_map import BodySurfaceMap
from .spatial_tracker import SpatialFingerprintTracker
from .activity_envelope import ActivityEnvelope


class SurfaceTests(unittest.TestCase):
    def test_activity_is_above_idle_per_channel(self):
        rng=np.random.default_rng(4)
        idle=rng.normal(0,[1,2,3,4,5],size=(500,5)).astype(np.float32)
        gate=ActivityEnvelope(idle,np.ones(5))
        _,excess,score=gate.measure(idle[-12:])
        self.assertLess(score,1.5)
        touch=idle[-12:].copy()
        touch[:,2]+=np.array([0,0,0,0,20,-20,15,-15,10,-10,5,-5])
        _,excess,score=gate.measure(touch)
        self.assertGreater(score,1.5)
        self.assertEqual(int(np.argmax(excess)),2)

    def test_chunked_serial_and_reset(self):
        class Serial:
            data = b'garbage\nSensor1:4,Sensor2:5,Sensor3:6,Sensor4:7,Sensor5:15\n1,2,3,4,5,6\n'
            @property
            def in_waiting(self):
                return len(self.data)
            def read(self, count):
                result,self.data=self.data[:count],self.data[count:]
                return result
            def reset_input_buffer(self):
                self.data=b''
        source=SerialTouchSource.__new__(SerialTouchSource)
        source.num_sensors=5
        source._read_buffer=bytearray()
        source.serial=Serial()
        source._serial_module=type('Module',(),{'SerialException':OSError})
        np.testing.assert_equal(source.read_sample(1),[4,5,6,7,15])
        np.testing.assert_equal(source.read_sample(1),[2,3,4,5,6])
        source._read_buffer.extend(b'stale')
        source.reset_input_buffer()
        self.assertFalse(source._read_buffer)

    def test_acquisition_keeps_newest_not_backlog(self):
        class Source:
            count=0
            finished=Event()
            release=Event()
            def read_sample(self, **kwargs):
                self.count+=1
                if self.count>100:
                    self.finished.set()
                    self.release.wait(1)
                    raise OSError('closed')
                return np.full(5,self.count,dtype=np.float32)
            def close(self):
                self.release.set()
        source=Source()
        stream=LatestSamples(source,capacity=32).start()
        try:
            self.assertTrue(source.finished.wait(1))
            wave,seq,_=stream.snapshot()
            self.assertEqual(seq,100)
            self.assertEqual(wave.shape,(32,5))
            self.assertEqual(wave[0,0],69)
            self.assertEqual(wave[-1,0],100)
        finally:
            stream.close()

    def test_every_channel_and_both_stroke_directions(self):
        tracker=SpatialFingerprintTracker.from_real_dataset(5,BodySurfaceMap.load())
        tracker.motion_model=None
        for p in tracker.by_channel:
            tracker.reset()
            values=np.zeros(5)
            values[p.channel]=100
            result=tracker.estimate(values)
            self.assertAlmostEqual(result.u,p.u)
            self.assertAlmostEqual(result.v,p.v)
        for front,rear,v in [(4,0,.75),(3,1,.25)]:
            for fractions in [np.linspace(0,1,21),np.linspace(1,0,21)]:
                tracker.reset()
                positions=[]
                for f in fractions:
                    values=np.zeros(5)
                    values[front]=100*(1-f)
                    values[rear]=100*f
                    result=tracker.estimate(values*tracker.sensor_scales)
                    positions.append(result.u)
                    self.assertLess(abs(result.v-v),.1)
                self.assertTrue(np.all(np.diff(positions)*np.sign(fractions[-1]-fractions[0])>=-.005))

    def test_measured_surface_interpolation(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            coords=[(.2,.75),(.8,.75),(.2,.25),(.8,.25),(.8,0)]
            for i,(u,v) in enumerate(coords):
                wave=np.zeros((300,5),dtype=np.float32)
                wave[40:,i]=100*np.sin(np.arange(260)*.7)
                np.save(root/f'{i}.npy',wave)
                (root/f'{i}.json').write_text(json.dumps(dict(file=f'{i}.npy',u=u,v=v)))
            field=SurfaceCalibration(np.ones(5),root)
            u,v=field.estimate(np.array([1,1,0,0,0]))
            self.assertAlmostEqual(u,.5,delta=.04)
            self.assertAlmostEqual(v,.75,delta=.04)

    def test_slow_motion_does_not_block_heat(self):
        tracker=SpatialFingerprintTracker.from_real_dataset(5,BodySurfaceMap.load())
        started,release=Event(),Event()
        original=tracker._predict_motion
        def slow(sequence):
            started.set()
            release.wait(2)
            return original(sequence)
        tracker._predict_motion=slow
        tracker.enable_async_motion()
        try:
            # Build changing history to activate motion. The worker remains
            # deliberately blocked while estimate must keep returning.
            for i in range(20):
                values=np.array([80 if i%2 else 10,0,0,0,10 if i%2 else 80])
                self.assertIsNotNone(tracker.estimate(values))
            self.assertTrue(started.wait(.5))
            self.assertIsNotNone(tracker.estimate(np.array([40,0,0,0,60])))
            tracker.reset()
            self.assertIsNone(tracker._cached_motion)
        finally:
            release.set()
            tracker.close()


if __name__=='__main__':
    unittest.main()
