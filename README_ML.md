# Robot Dog Touch ML Pipeline

**Current live dashboard instructions and limitations:** see
[LIVE_TOUCH_AUDIT.md](LIVE_TOUCH_AUDIT.md). It supersedes older live-heat tuning
instructions below and includes the optional 16-recording surface workflow.

This pipeline classifies touch type and location from interchangeable simulated or
real sensor windows. The current ESP32-S3 hardware profile uses five piezos; older
two-sensor recordings and the original eight-sensor simulator remain isolated by
model metadata so incompatible data cannot be mixed accidentally.

See [ML_RESEARCH.md](ML_RESEARCH.md) for the peer-reviewed basis behind the signal
features, propagation timing, augmentation, evaluation, and confidence rejection.

## Final five-sensor workflow

Close the dashboard before data collection because only one program can own the
ESP32 serial port at a time. The wiring map is fixed: GPIO 4 rear-left, GPIO 5
rear-right, GPIO 6 top-rear butt, GPIO 7 front-right, and GPIO 15 front-left.
Calibration is not required.

1. Collect balanced five-sensor gesture data:

   ```powershell
   python -m ml.collect_five_sensor --port COM5 --threshold 10 --count 20
   ```

   The collector walks through every calibrated sensor and every gesture. Keep
   your hand off the shell, press Enter, wait for `ARMED`, then perform exactly
   the displayed gesture once. All five channels are recorded for every touch.

   To preview deletion of only the five-sensor real dataset, run
   `python -m ml.reset_dataset --source real --sensor-count 5`. Add `--confirm`
   only after checking the matched count. The command preserves other datasets,
   backs up `labels.csv`, and moves removed waveforms to a recoverable archive.

2. Teach the detector what the powered-on robot looks like when untouched:

   ```powershell
   python -m ml.collect_background --port COM5 --count 100
   ```

   Keep the robot powered on in its normal state, but do not touch the robot,
   table, cable, or sensors for about 30 seconds. These recordings are appended
   safely with the label `no_touch`; existing gesture data is not overwritten.
   For better motor-vibration coverage, repeat this once in each normal stance or
   operating mode.

3. Train only the current five-sensor recordings (older two-sensor files are
   intentionally excluded):

   ```powershell
   python -m ml.train --source real --sensor-count 5
   ```

   Training groups the detailed recordings into four robust output classes:
   `pat_stroke` (pet + stroke), `tap_poke` (tap + poke), `hard_tap`, and
   `scratch`. The original labels remain unchanged in `labels.csv`.

4. Start the spatial-only dashboard with the final model and ESP32 connected:

   ```powershell
   python -m ml.ui_server --serial-port COM5 --threshold 10
   ```

   Open `http://localhost:8080`. Do not click a browser serial button: the Python
   inference server owns the serial connection and sends structured ML predictions
   to the UI.

   Live detection first learns a robust per-sensor median/MAD baseline. A trained
   touch-presence gate then evaluates a sliding 300 ms window 20 times per second,
   with hysteresis to reject powered-on vibration without flicker. While that gate
   is active, a 32 ms spatial tracker updates up to 100 times per second. A fast
   transient path previews contact immediately while the presence and gesture
   models run on a separate thread, so model computation cannot block serial
   acquisition. It matches
   the changing five-channel energy ratio to measured location fingerprints and
   continuously interpolates them on the curved body. A front-to-back pet therefore
   moves the heat spot along the robot instead of producing one frozen event. The
   UI retains a short fading trail, so the direction and extent of the hand motion
   remain visible without permanently coloring the shell. Live signal amplitude
   controls delicate-to-hard heat intensity, while multi-sensor participation
   controls the displayed contact area; gesture names arrive asynchronously.

## Improve moving-touch localization

The sensor-centered dataset teaches five fixed points, but it does not fully teach
left/right shell propagation or stroke direction. Collect 20 examples of each of
four guided paths (left and right sides, in both directions):

```powershell
python -m ml.collect_motion_paths --port COM5 --threshold 10 --count 20
```

For every prompt, keep your hand away and press Enter. At `ARMED`, use one finger
to make one smooth torso stroke in the displayed direction, taking about half a
second. Stay entirely on the requested side. Existing touch data is not replaced.

Train the continuous spatial regressor and restart the dashboard:

```powershell
python -m ml.train_motion
python -m ml.ui_server --serial-port COM5 --threshold 10
```

Training holds out complete recordings rather than random slices and reports
front/rear error plus left/right accuracy. The dashboard automatically loads the
new spatial model while retaining its fast signal-driven heat response.

Every data source returns the same `np.float32` array with shape `(600, 8)`:

```python
waveform = source.get_touch()
```

## Windows setup in VS Code

Open the repository folder in VS Code, then run these commands in its PowerShell terminal.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` once in that terminal and try again.

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Generate, train, and try the demo

Generate 600 synthetic examples (30 for each touch/location combination):

```powershell
python -m ml.generate_dataset
```

The generator will not replace existing data by default. To intentionally rebuild a purely simulated dataset, use `python -m ml.generate_dataset --overwrite`. It refuses to overwrite a dataset containing rows marked as real.

Train and evaluate both random-forest models:

```powershell
python -m ml.train
```

Run the interactive demo:

```powershell
python -m ml.demo
```

For a non-interactive example:

```powershell
python -m ml.demo --touch-type scratch --location head
```

The accuracy printed during training measures only how well the model recognizes this simulator. It is **not** an estimate of accuracy on a real robot.

## Saving future recordings

Both simulated and real recordings use `.npy` files plus the same `data/labels.csv` index. To add a labeled real recording:

```python
from ml.storage import save_touch

save_touch(waveform, "tap", "head", source="real")
```

The helper validates the `(600, 8)` float32 contract, chooses a unique filename, saves it under `data/raw`, and appends its label. The training program reads both real and simulated rows without special cases.

## Switching between simulation and hardware

The simulator remains an eight-channel, 2,000 Hz development source. The current ESP32-S3 is a separate two-channel, approximately 1,000 Hz source. Both return arrays oriented as `(time, sensors)`, and the shared preprocessing and feature code discovers the sensor count from the array. Real data is never padded with fake channels.

The live commands below select hardware explicitly. After real models have been trained, `ml.demo` can also use them by setting `USE_HARDWARE = True` and updating `SERIAL_PORT` in `ml/config.py`.

## Current ESP32-S3 hardware workflow

The current hardware path expects five channels on GPIO 4, 5, 6, 7, and 15 at approximately 1,000 Hz. It accepts Arduino output such as `Sensor1:123,Sensor2:45,Sensor3:8,Sensor4:2,Sensor5:10`. Raw windows are `(300, 5)`. A median baseline is estimated independently for every channel from pre-trigger samples; no ADC midpoint is assumed.

First inspect live values. Add `--plot` for a scrolling graph:

```powershell
python -m ml.test_live --port COM5 --plot
```

Choose an event threshold by observing the idle noise and real touches. The threshold is an absolute ADC deviation and is deliberately not hard-coded. Capture one labeled example:

```powershell
python -m ml.capture --port COM5 --touch-type tap --location left --threshold 50
```

Replace `50` with a value measured for your circuit. Capture repeated examples with `--count`, changing labels as needed:

```powershell
python -m ml.capture --port COM5 --touch-type tap --location left --threshold 50 --count 30
```

Train only from rows marked `real`; this creates separate metadata-bearing model files and does not overwrite the synthetic models:

```powershell
python -m ml.train --source real
```

Start live detection with the same experimentally chosen threshold:

```powershell
python -m ml.live_predict --port COM5 --threshold 50
```

## Calibrating and viewing the Go2 body map

The torso is represented as a flat wrapped net. Horizontal position runs from the head (`u=0`) to the rear (`u=1`). Vertical position wraps around the curved body: top (`v=0/1`), right side (`v=0.25`), underside (`v=0.5`), and left side (`v=0.75`). Touch position is a continuous weighted estimate, so it can fall between physical sensors.

After mounting all five sensors at the marked locations, map each GPIO to its physical point. Replace `5` with the event threshold measured for the finished installation:

```powershell
python -m ml.calibrate_map --port COM5 --threshold 5
```

The program asks you to tap each marked sensor and saves `data/sensor_map.json`. Then open the live flat body map:

```powershell
python -m ml.live_map --port COM5 --threshold 5
```

The initial position is an energy-weighted estimate. Robot-body vibration propagation is not uniform, so accurate interpolation away from the five sensors will eventually require collecting calibration touches at additional known grid points and fitting a localization model.

## Three.js Go2 dashboard

The local dashboard loads the supplied Go2 GLB, snaps calibrated sensor markers to its curved mesh, and shows the same estimated touch on both the 3D robot and a flattened torso net. Start it from the repository root:

```powershell
python -m ml.ui_server
```

In the browser, click **Connect ESP32**, select the CP210x/ESP32 serial port, and touch the robot. Use desktop Chrome or Edge because Web Serial is required. Close Arduino Serial Monitor, Serial Plotter, and the Python live graph first so the browser can open the port. The server refreshes the dashboard's sensor map from `data/sensor_map.json` whenever it starts.

The supplied ESP32 firmware format needs no change. Close Arduino Serial Monitor/Plotter before running Python because only one program can normally own the COM port. The configured 1,000 Hz is approximate because the firmware uses `delayMicroseconds(1000)` and serial/ADC work adds overhead; if measured sampling differs materially, update `HARDWARE_SAMPLE_RATE` before collecting a dataset.

## Main files

- `ml/source_base.py`: the shared data-source contract
- `ml/source_simulated.py`: varied synthetic piezo signals
- `ml/source_serial.py`: lazy-loaded serial capture for the Teensy
- `ml/storage.py`: common storage for real and simulated touches
- `ml/features.py`: source-independent signal features
- `ml/train.py` and `ml/inference.py`: model training and prediction
- `ml/demo.py`: configuration-driven end-to-end example
