# Research basis for the Skinless tactile pipeline

Skinless uses a **hybrid signal-processing and machine-learning pipeline**. With five piezos and a modest dataset, physically meaningful features are more data-efficient and easier to debug than an oversized neural network.

## Why each component exists

### Robust baseline and event gating

Piezo sensors measure changes and vibration, not stable force. Each channel is
median-centered with a median-absolute-deviation noise estimate. Baseline removal
handles steady offsets, but changing motor vibration can still resemble contact.
Skinless therefore learns a separate binary `touch` versus `no_touch` gate from
powered-on negative recordings before it attempts gesture or location inference.
The gate operates on a sliding window and uses separate enter/leave thresholds
(hysteresis), so a single ADC spike cannot start a touch and a long pet does not
flicker off between movements. This two-stage structure also prevents the gesture
classifier—which only knows named gestures—from being forced to label ordinary
robot vibration as one of them.

### Time-domain shape

Peak, RMS, crest factor, duration, envelope, impulse count, periodicity, and early/late energy distinguish short impacts from sustained gestures. Social-touch work on a NAO robot extracted statistical, entropy, area, and peak features from tactile channels. See [Endowing a NAO Robot With Practical Social-Touch Perception](https://pmc.ncbi.nlm.nih.gov/articles/PMC9061995/).

### Time-frequency structure

Welch spectra, dominant frequency, spectral centroid, bandwidth, rolloff, entropy, and five relative frequency bands capture differences between a sharp tap, repeated scratch, and smoother pet. Piezoelectric tactile-skin research reports that multiresolution time-frequency features improve contact classification and regression. See [Leveraging Time-Frequency Features for Contact Classification and Regression with a Piezoelectric Tactile Skin](https://doi.org/10.1109/DSN-W65791.2025.00041).

### Spatial energy and propagation timing

Relative channel energy estimates proximity without depending on absolute ADC gain. Pairwise correlation and peak-time lag describe vibration propagation across the shell. Time-difference-of-arrival is a standard basis for piezo impact localization; see [Impact Location in an Isotropic Plate without Training](https://doi.org/10.1016/j.proeng.2017.04.471). Because the Go2 shell is curved and mechanically non-uniform, Skinless learns the mapping instead of assuming one wave velocity.

The dashboard uses the model's complete location-probability distribution rather
than raw sensor amplitude. Probabilities are sharpened and weak alternatives are
suppressed before calculating a circular surface centroid. This is an engineering
adaptation for visualization: it prevents mechanically widespread vibration from
being displayed as five simultaneous touches, while allowing interpolation when
two neighboring calibrated locations are both plausible. Its exponent and cutoff
must be validated by measuring physical localization error on held-out touch points.

### Motion features for strokes and petting

Each event is divided into six temporal bins. The model sees how energy moves between sensors through time, separating a moving stroke from repeated vibration at one point. Robot-skin research has combined spatial descriptors with temporal statistics and wavelet analysis for emotional-touch recognition; see [Textile Pressure Mapping Sensor for Emotional Touch Detection in Human-Robot Interaction](https://pmc.ncbi.nlm.nih.gov/articles/PMC5713507/).

For visualization, a separate low-latency 32 ms tracker compares the current
five-channel RMS-energy ratio to location fingerprints measured from the real
robot. A softmax over cosine similarities produces a continuous weighted body
coordinate rather than snapping to a sensor. The classifier's 300 ms window gives
the gate and gesture label stability, while the shorter tracker window makes the
heat spot follow a hand moving from front to rear.

Serial acquisition and ML classification run on separate threads with a
latest-frame queue. This is important for causality as well as speed: a slow model
must not allow unread serial samples to accumulate and make the display portray
old contact as if it were current. A short robust transient gate provides the
immediate heat response; the learned powered-on `no_touch` gate confirms or
retracts it asynchronously. Heat intensity is calibrated from real touch-amplitude
percentiles, and displayed area uses the gain-corrected sensor participation ratio.

The tracker also estimates a gain correction for each piezo from recordings made
at that sensor. It blends the learned propagation fingerprint with a
gain-corrected physical sensor centroid. This hybrid matters because the current
mountings differ substantially in sensitivity: fingerprint matching accounts for
shell cross-talk, while the physical centroid guarantees that energy moving from
front sensors toward rear sensors moves in the same direction in the visualization.

### Physically plausible augmentation

Training-only gain scaling models mounting pressure and ADC gain; per-channel scaling models sensitivity variation; small time shifts model trigger jitter; low Gaussian noise models electronics. Time-series augmentation research supports jittering and scaling while warning that transformations must respect the signal domain. See [An empirical survey of data augmentation for time series classification with neural networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049/).

### Complementary ensemble

Extra Trees captures nonlinear feature interactions. A robust-scaled RBF SVM supplies a smoother decision boundary. Their probability average is less dependent on either model's failure mode and remains appropriate for hundreds—not millions—of recordings.

The final gesture targets merge `pet` with `stroke` and `tap` with `poke`, while
keeping `hard_tap` and `scratch` distinct. This follows the sensor's observable
physics: piezos measure transient vibration rather than static force, so semantic
labels with strongly overlapping temporal signatures create avoidable label noise.
The detailed labels remain stored, allowing the grouping to be changed later if
additional sensing modalities make those distinctions identifiable.

### Leakage-resistant evaluation

Five consecutive recordings from one capture burst stay on the same side of the train/test split. Otherwise nearly identical adjacent windows can make accuracy look unrealistically good. `StratifiedGroupKFold` preserves class balance while preventing group overlap; see the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

### Confidence rejection

The model derives a conservative confidence floor from held-out correct predictions. A waveform below the floor returns `unknown` instead of forcing a label. Reliable confidence is distinct from raw accuracy; see [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).

## What must be validated experimentally

The existing real dataset was captured with two sensors. It can test touch-type code, but it cannot teach the current five-sensor geometry. Collect balanced five-sensor examples for every gesture and mapped surface point, from several people and separate recording sessions. Report gesture balanced accuracy and physical localization error in centimeters. Synthetic accuracy is only a software test and is not evidence of robot performance.
