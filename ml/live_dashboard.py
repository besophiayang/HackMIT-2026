"""Latest-state streaming dashboard: acquisition, heat and gestures run separately."""
import argparse
from collections import deque
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
from queue import Queue, Empty, Full
from threading import Condition, Event, Thread
import time
import webbrowser
import numpy as np
from .body_map import BodySurfaceMap
from .config import HARDWARE_NUM_SAMPLES, HARDWARE_NUM_SENSORS, SERIAL_BAUD, PROJECT_ROOT
from .fast_contact import FastContactGate
from .inference import TouchClassifier
from .live_stream import LatestSamples
from .source_serial import SerialTouchSource
from .spatial_tracker import SpatialFingerprintTracker
from .activity_envelope import ActivityEnvelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--serial-port')
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Live activity threshold in gain-corrected ADC-RMS above idle (default: 1.5)')
    parser.add_argument('--baud', type=int, default=SERIAL_BAUD)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    changed, stopped = Condition(), Event()
    state = dict(mode='ml' if args.serial_port else 'demo', status='starting' if args.serial_port else 'demo',
                 message='Keep the robot untouched during startup.' if args.serial_port else 'Demo mode',
                 prediction=None, sequence=0)

    def publish(**updates):
        with changed:
            state.update(updates)
            state['sequence'] += 1
            changed.notify_all()

    def run_live():
        stream = None
        tracker = None
        jobs = Queue(maxsize=1)
        gesture = dict(label='tracking_touch', confidence=0.0, event=-1, completed=0.0)
        gesture_lock = Condition()
        try:
            tracker = SpatialFingerprintTracker.from_real_dataset(HARDWARE_NUM_SENSORS, BodySurfaceMap.load())
            tracker.enable_async_motion()
            gate = FastContactGate.load()

            def classify():
                try:
                    classifier = TouchClassifier(source='real')
                    while not stopped.is_set():
                        try:
                            waveform, event = jobs.get(timeout=0.5)
                        except Empty:
                            continue
                        label, confidence = classifier.predict_touch_type(waveform)
                        with gesture_lock:
                            gesture.update(label=label, confidence=confidence, event=event, completed=time.monotonic())
                except Exception as error:
                    publish(classification_warning=str(error))

            Thread(target=classify, daemon=True).start()
            source = SerialTouchSource(args.serial_port, args.baud, HARDWARE_NUM_SENSORS, HARDWARE_NUM_SAMPLES)
            stream = LatestSamples(source).start()
            baseline = noise = envelope = None
            idle_windows = deque(maxlen=100)
            sequence = 0
            last_contact, last_classification, last_baseline, last_location = -10.0, 0.0, 0.0, 0.0
            positive_since = None
            last_location_sequence = 0
            event, active, estimate = 0, False, None
            rate_sequence, rate_time, rate = 0, time.monotonic(), 0.0
            compute_times = deque(maxlen=200)
            while not stopped.is_set():
                frame = stream.snapshot(sequence)
                now = time.monotonic()
                if frame is None:
                    if now - stream.received_at > 0.25:
                        publish(status='waiting', prediction=None, message='Waiting for fresh sensor data.')
                    continue
                window, sequence, received_at = frame
                if baseline is None:
                    if len(window) < 500:
                        continue
                    baseline = np.median(window, axis=0)
                    noise = np.maximum(1.4826*np.median(abs(window-baseline),axis=0),0.5)
                    envelope = ActivityEnvelope(window[-500:],tracker.sensor_scales)
                begin = time.perf_counter()
                probability = gate.probability(window[-12:], noise)
                if probability >= gate.threshold:
                    positive_since = now if positive_since is None else positive_since
                else:
                    positive_since = None
                # Strong onsets need not wait through the debounce interval.
                classifier_detected = probability >= 0.97 or (positive_since is not None and now-positive_since >= 0.008)
                rms,strengths,activity_score = envelope.measure(window)
                # Physical vibration activity drives heat immediately. The ML
                # gate is a second detector, never the sole permission to draw.
                detected = activity_score >= args.threshold or classifier_detected
                if detected and np.max(strengths) > 0:
                    if not active:
                        event += 1
                    active, last_contact = True, now
                elif now-last_contact > 0.070:
                    if active:
                        tracker.reset()
                    active, estimate = False, None
                if not active:
                    idle_windows.append(window[-12:].copy())
                    if len(idle_windows)>=20 and now-last_baseline>0.5:
                        idle=np.concatenate(idle_windows)
                        baseline=np.median(idle,axis=0)
                        noise=np.maximum(1.4826*np.median(abs(idle-baseline),axis=0),0.5)
                        last_baseline=now
                else:
                    # Match the motion model's ten-sample training stride,
                    # independent of the board's actual (not assumed) rate.
                    if estimate is None or sequence-last_location_sequence>=10:
                        estimate=tracker.estimate(strengths)
                        last_location=now
                        last_location_sequence=sequence
                    if len(window)>=HARDWARE_NUM_SAMPLES and now-last_classification>=0.35:
                        try:
                            jobs.put_nowait((window[-HARDWARE_NUM_SAMPLES:].copy(),event))
                        except Full:
                            try:
                                jobs.get_nowait()
                            except Empty:
                                pass
                            jobs.put_nowait((window[-HARDWARE_NUM_SAMPLES:].copy(),event))
                        last_classification=now
                if now-rate_time>=1:
                    rate=(sequence-rate_sequence)/(now-rate_time)
                    rate_sequence,rate_time=sequence,now
                intensity,area=tracker.characterize(strengths)
                prediction=None
                if active and estimate is not None:
                    with gesture_lock:
                        label=dict(gesture)
                    current=label['event']==event and now-label['completed']<1
                    prediction=dict(touch_type=label['label'] if current else 'tracking_touch',
                        touch_confidence=float(label['confidence']) if current else 0.0,
                        location=estimate.location,location_confidence=estimate.confidence,
                        u=estimate.u,v=estimate.v,intensity=intensity,contact_area=area,
                        surface_side=estimate.side,
                        fast_contact_confidence=probability,strengths=strengths.tolist(),
                        event_id=event,timestamp=time.time())
                compute_times.append((time.perf_counter()-begin)*1000)
                publish(status='touch' if prediction else 'ready',message='Tracking contact' if prediction else 'Live touch ready',
                    prediction=prediction,diagnostics=dict(sample_rate_hz=round(rate,1),
                        sample_age_ms=round((time.monotonic()-received_at)*1000,2),
                        processing_p95_ms=round(float(np.percentile(compute_times,95)),2),
                        serial_pending_bytes=source.serial.in_waiting,raw=window[-1].tolist(),
                        rms=rms.tolist(),idle_rms=envelope.idle_rms.tolist(),strengths=strengths.tolist(),
                        activity_score=round(activity_score,3),activity_threshold=args.threshold,
                        surface_calibration_enabled=tracker.surface.signatures is not None,
                        surface_calibration_error=tracker.surface.validation_error,
                        noise=noise.tolist(),gate_probability=probability,gate_threshold=gate.threshold))
                stopped.wait(max(0,0.005-(time.perf_counter()-begin)))
        except Exception as error:
            publish(status='error',message=str(error),prediction=None)
        finally:
            if stream:
                stream.close()
            if tracker:
                tracker.close()

    class Handler(SimpleHTTPRequestHandler):
        protocol_version='HTTP/1.1'

        def __init__(self,*a,**kw):
            super().__init__(*a,directory=str(PROJECT_ROOT/'ui'/'dist'),**kw)

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):
                pass  # Normal when the browser refreshes a keep-alive connection.

        def send_json(self,value):
            data=json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def end_headers(self):
            self.send_header('Cache-Control','no-store')
            super().end_headers()

        def do_GET(self):
            try:
                path=self.path.split('?')[0]
                if path=='/api/events':
                    self.send_response(200)
                    self.send_header('Content-Type','text/event-stream')
                    self.end_headers()
                    last=-1
                    while not stopped.is_set():
                        with changed:
                            changed.wait_for(lambda: state['sequence']!=last or stopped.is_set(),timeout=1)
                            snapshot=dict(state)
                            last=snapshot['sequence']
                        self.wfile.write(('data: '+json.dumps(snapshot)+'\n\n').encode())
                        self.wfile.flush()
                        stopped.wait(1/60)
                    return
                if path=='/api/state':
                    with changed:
                        snapshot=dict(state)
                    self.send_json(snapshot)
                elif path=='/assets/sensor-map.json':
                    self.send_json(dict(placements=[vars(p) for p in BodySurfaceMap.load().placements]))
                else:
                    super().do_GET()
            except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):
                pass

        def log_message(self,format,*values):
            if not self.path.startswith('/api/'):
                super().log_message(format,*values)

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    if args.serial_port:
        Thread(target=run_live,daemon=True).start()
    address=f'http://localhost:{args.port}'
    print(f'Skinless dashboard: {address}\nCtrl+C stops the server and releases serial.')
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        with changed:
            changed.notify_all()
        server.server_close()


if __name__=='__main__':
    main()
