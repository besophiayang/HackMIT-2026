import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const view = document.querySelector("#model-view");
const loading = document.querySelector("#loading");
const statusElement = document.querySelector("#status");
const messageElement = document.querySelector("#message");
const typeElement = document.querySelector("#touch-type");
const detailElement = document.querySelector("#touch-detail");
const sensorStrip = document.querySelector("#sensor-strip");
const sensorPins = [4,5,6,7,15];
sensorStrip.innerHTML = sensorPins.map((pin,index) => `<span class="sensor-meter" data-channel="${index}">GPIO${pin}<i></i></span>`).join('');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// Standard OrbitControls direction: drag left to orbit the view left.
controls.rotateSpeed = 0.78;
controls.enablePan = false;
controls.minDistance = 3;
controls.maxDistance = 14;
scene.add(new THREE.HemisphereLight(0xd8f8ff, 0x18202a, 2.5));
const light = new THREE.DirectionalLight(0xffffff, 2.8);
light.position.set(4, 7, 5);
scene.add(light);
const grid = new THREE.GridHelper(12, 24, 0x263848, 0x15212c);
scene.add(grid);

let placements = [];
let modelRoot = null;
let modelBounds = null;
let pointCloud = null;
let baseColors = null;
let heatUniforms = null;
let activeTouch = null;
let touchTrail = [];
let lastPredictionTimestamp = 0;
let lastEventId = null;
let lastArrival = 0;
let lastDiagnosticUpdate = 0;
let demoPreview = null;
let demoUntil = 0;
const surfaceCache = new Map();
const surfaceMeshes = [];
const raycaster = new THREE.Raycaster();
const sensorGroup = new THREE.Group();
scene.add(sensorGroup);

function resize() {
  const width = view.clientWidth;
  const height = view.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);

function surfacePosition({ u, v }) {
  if (!modelBounds) return new THREE.Vector3();
  const key = `${Math.round(u*100)},${Math.round(v*100)}`;
  if (surfaceCache.has(key)) return surfaceCache.get(key).clone();
  const size = modelBounds.getSize(new THREE.Vector3());
  const center = modelBounds.getCenter(new THREE.Vector3());
  const longitudinalIsX = size.x >= size.z;
  // Fixed physical orientation: u=0 is the head/front at screen-left and
  // u=1 is the rear at screen-right. The view is never coordinate-flipped.
  const longitudinalDirection = longitudinalIsX ? 1 : -1;
  const longitudinal = longitudinalDirection * (u - 0.5) * (longitudinalIsX ? size.x : size.z) * 0.69;
  const angle = v * Math.PI * 2;
  // With the default overhead camera, right-side sensors appear above the
  // centerline and left-side sensors appear below it, matching the real robot.
  const crossDirection = -1;
  const cross = crossDirection * Math.sin(angle) * (longitudinalIsX ? size.z : size.x) * 0.34;
  const y = modelBounds.max.y - size.y * 0.17 + Math.cos(angle) * size.y * 0.045;
  const approximate = longitudinalIsX
    ? new THREE.Vector3(center.x + longitudinal, y, center.z + cross)
    : new THREE.Vector3(center.x + cross, y, center.z + longitudinal);
  const axis = longitudinalIsX
    ? new THREE.Vector3(center.x + longitudinal, modelBounds.max.y - size.y * 0.17, center.z)
    : new THREE.Vector3(center.x, modelBounds.max.y - size.y * 0.17, center.z + longitudinal);
  const outward = approximate.clone().sub(axis).normalize();
  raycaster.set(axis.clone().addScaledVector(outward, Math.max(size.y, size.z) * 0.8), outward.clone().negate());
  const hit = raycaster.intersectObjects(surfaceMeshes, false)[0];
  const position = hit ? hit.point.clone().addScaledVector(outward, 0.03) : approximate;
  surfaceCache.set(key, position.clone());
  return position;
}

function excludedGeometry(object) {
  let current = object;
  while (current) {
    if (current.name?.startsWith("base_white") || current.name?.startsWith("base_black")) return true;
    current = current.parent;
  }
  const materials = Array.isArray(object.material) ? object.material : [object.material];
  return materials.some((material) => material?.name === "material_100100100");
}

function buildPointCloud(model) {
  model.updateMatrixWorld(true);
  const meshes = [];
  model.traverse((object) => {
    if (!object.isMesh || !object.geometry?.attributes?.position) return;
    if (excludedGeometry(object)) { object.visible = false; return; }
    meshes.push(object);
    surfaceMeshes.push(object);
    object.material = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, depthTest: true, side: THREE.DoubleSide });
  });
  const cells = new Map();
  const point = new THREE.Vector3();
  const voxel = 0.022;
  for (const mesh of meshes) {
    const attribute = mesh.geometry.attributes.position;
    for (let index = 0; index < attribute.count; index += 1) {
      point.fromBufferAttribute(attribute, index).applyMatrix4(mesh.matrixWorld);
      const key = `${Math.round(point.x / voxel)},${Math.round(point.y / voxel)},${Math.round(point.z / voxel)}`;
      if (!cells.has(key)) cells.set(key, [point.x, point.y, point.z]);
    }
  }
  const samples = [...cells.values()];
  const stride = Math.max(1, Math.ceil(samples.length / 90000));
  const positions = [];
  for (let index = 0; index < samples.length; index += stride) positions.push(...samples[index]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  // Angular surface distance prevents a broad Euclidean heat sphere from
  // passing through the thin torso and coloring the opposite panel.
  const boundsSize = modelBounds.getSize(new THREE.Vector3());
  const boundsCenter = modelBounds.getCenter(new THREE.Vector3());
  const torsoY = modelBounds.max.y - boundsSize.y * 0.17;
  const longX = boundsSize.x >= boundsSize.z;
  const surfaceAngle = positions.filter((_, i) => i % 3 === 0).map((_, i) => {
    const cross = longX ? positions[i*3+2]-boundsCenter.z : positions[i*3]-boundsCenter.x;
    return Math.atan2(-cross / ((longX ? boundsSize.z : boundsSize.x)*0.34),
      (positions[i*3+1]-torsoY)/(boundsSize.y*0.045));
  });
  geometry.setAttribute('surfaceAngle', new THREE.Float32BufferAttribute(surfaceAngle,1));
  heatUniforms = {
    uCount: { value: 0 },
    uCenters: { value: Array.from({ length: 8 }, () => new THREE.Vector3()) },
    uRadii: { value: new Float32Array(8) },
    uStrengths: { value: new Float32Array(8) },
    uAngles: { value: new Float32Array(8) },
  };
  const material = new THREE.ShaderMaterial({
    uniforms: heatUniforms,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    vertexShader: `
      uniform int uCount;
      uniform vec3 uCenters[8];
      uniform float uRadii[8];
      uniform float uStrengths[8];
      uniform float uAngles[8];
      attribute float surfaceAngle;
      varying float vHeat;
      void main() {
        float heat = 0.0;
        for (int i = 0; i < 8; i++) {
          if (i < uCount) {
            float radius = max(uRadii[i], 0.0001);
            float distanceSquared = dot(position - uCenters[i], position - uCenters[i]);
            float angularDistance = abs(atan(sin(surfaceAngle-uAngles[i]),cos(surfaceAngle-uAngles[i])));
            float sideMask = 1.0 - smoothstep(0.7, 1.3, angularDistance);
            heat = max(heat, sideMask * exp(-distanceSquared / (2.0 * radius * radius)) * uStrengths[i]);
          }
        }
        vHeat = heat;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        gl_PointSize = 1.45 + min(heat, 1.0) * 2.8;
      }
    `,
    fragmentShader: `
      varying float vHeat;
      void main() {
        vec2 point = gl_PointCoord - vec2(0.5);
        if (dot(point, point) > 0.25) discard;
        vec3 base = vec3(0.015, 0.15, 0.19);
        float heat = clamp(vHeat, 0.0, 1.0);
        vec3 red = vec3(1.0, 0.06, 0.0);
        vec3 whiteHot = vec3(1.0, 1.0, 0.88);
        vec3 color = heat < 0.58
          ? mix(base, red, sqrt(heat / 0.58))
          : mix(red, whiteHot, (heat - 0.58) / 0.42);
        gl_FragColor = vec4(color, 0.92);
      }
    `,
  });
  pointCloud = new THREE.Points(geometry, material);
  pointCloud.scale.setScalar(1.003);
  scene.add(pointCloud);
}

function buildSensorMarkers() {
  sensorGroup.clear();
  for (const placement of placements) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 16), new THREE.MeshBasicMaterial({ color: 0x45e6d0, transparent: true, opacity: 0.8 }));
    marker.position.copy(surfacePosition(placement));
    sensorGroup.add(marker);

    const canvas = document.createElement("canvas");
    canvas.width = 256; canvas.height = 80;
    const context = canvas.getContext("2d");
    context.fillStyle = "rgba(8, 11, 16, 0.82)";
    context.beginPath(); context.roundRect(18, 8, 220, 60, 20); context.fill();
    context.strokeStyle = "#45e6d0"; context.lineWidth = 3; context.stroke();
    context.fillStyle = "#f1f5f7"; context.font = "600 30px system-ui";
    context.textAlign = "center"; context.textBaseline = "middle";
    context.fillText(`GPIO ${placement.pin}`, 128, 39);
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false }));
    label.position.copy(marker.position);
    label.position.y += 0.07;
    label.center.set(0.5, -0.18);
    label.scale.set(0.5, 0.16, 1);
    label.renderOrder = 10;
    sensorGroup.add(label);
  }
}

function updateHeat() {
  if (!pointCloud || !heatUniforms) return;
  const now = Date.now() / 1000;
  touchTrail = touchTrail.filter((touch) => now - touch.time < 0.18).slice(-8);
  const baseRadius = modelBounds.getSize(new THREE.Vector3()).length() * 0.030;
  heatUniforms.uCount.value = touchTrail.length;
  for (let index = 0; index < 8; index += 1) {
    const touch = touchTrail[index];
    if (touch) {
      const age = now - touch.time;
      const decay = Math.max(0, 1 - age / 0.18);
      heatUniforms.uCenters.value[index].copy(touch.center);
      heatUniforms.uAngles.value[index] = touch.v * Math.PI * 2;
      heatUniforms.uRadii.value[index] = baseRadius * (0.72 + 0.9 * (touch.area || 0));
      heatUniforms.uStrengths.value[index] = decay * (0.12 + 1.08 * Math.sqrt(touch.intensity || 0));
    } else {
      heatUniforms.uRadii.value[index] = 0;
      heatUniforms.uStrengths.value[index] = 0;
    }
  }
}

function showPrediction(prediction) {
  if (!prediction || !Number.isFinite(prediction.u) || !Number.isFinite(prediction.v)) return;
  const now = Date.now() / 1000;
  if (prediction.event_id !== undefined && prediction.event_id !== lastEventId) {
    activeTouch = null;
    touchTrail = [];
    lastEventId = prediction.event_id;
  }
  // Do not draw a trail across the torso when the evidence switches sides.
  if (activeTouch && Math.abs(((prediction.v-activeTouch.v+1.5)%1)-0.5)>0.30) {
    activeTouch = null;
    touchTrail = [];
  }
  let movementSpeed = 0;
  if (activeTouch) {
    const elapsed = Math.max(now - activeTouch.time, 0.001);
    const du = prediction.u - activeTouch.u;
    const dv = ((prediction.v - activeTouch.v + 1.5) % 1) - 0.5;
    movementSpeed = Math.hypot(du, dv) / elapsed;
    activeTouch.u += (prediction.u - activeTouch.u) * 0.88;
    const wrappedV = ((prediction.v - activeTouch.v + 1.5) % 1) - 0.5;
    activeTouch.v = (activeTouch.v + wrappedV * 0.88 + 1) % 1;
    activeTouch.confidence = prediction.location_confidence;
    activeTouch.intensity = prediction.intensity;
    activeTouch.area = prediction.contact_area;
    activeTouch.time = now;
  } else {
    activeTouch = {
      u: prediction.u, v: prediction.v, confidence: prediction.location_confidence,
      intensity: prediction.intensity, area: prediction.contact_area, time: now,
    };
  }
  touchTrail.push({ ...activeTouch, time: now, center: surfacePosition(activeTouch) });
  touchTrail = touchTrail.slice(-8);
  const displayNames = {
    pat_stroke: "Pat / stroke",
    tap_poke: "Tap / poke",
    hard_tap: "Hard tap",
    scratch: "Scratch",
    tracking_touch: "Tracking touch",
  };
  const rawType = String(prediction.touch_type || "touch");
  const type = displayNames[rawType] || rawType.replaceAll("_", " ");
  typeElement.textContent = type[0].toUpperCase() + type.slice(1);
  const intensity = Number(prediction.intensity || 0);
  const hardness = intensity < 0.28 ? "delicate" : intensity < 0.62 ? "medium" : intensity < 0.86 ? "firm" : "hard";
  const motion = movementSpeed > 0.16 ? "moving" : "stationary";
  const area = Number(prediction.contact_area || 0) > 0.42 ? "broad" : "focused";
  detailElement.textContent = `${hardness} · ${area} · ${motion}`;
  lastPredictionTimestamp = Number(prediction.timestamp || Date.now() / 1000);
  updateHeat();
}

function receiveState(state) {
    lastArrival = performance.now();
    statusElement.className = `status ${state.status}`;
    statusElement.lastChild.textContent = state.status === "ready" ? " ML ready" : state.status === "touch" ? " Touch detected" : state.status === "error" ? " Model error" : " Demo mode";
    if (performance.now()-lastDiagnosticUpdate > 500) {
      const d = state.diagnostics;
      messageElement.textContent = state.message + (d ? ` · ${Math.round(d.sample_rate_hz)} samples/s · processing ${d.processing_p95_ms} ms` : '');
      lastDiagnosticUpdate = performance.now();
    }
    if (state.diagnostics?.strengths) {
      const threshold = Math.max(Number(state.diagnostics.activity_threshold || 1.5), .01);
      document.querySelectorAll('.sensor-meter').forEach((meter,index) => {
        const level = Math.min(1, Number(state.diagnostics.strengths[index] || 0) / (threshold*4));
        meter.style.setProperty('--level', level.toFixed(3));
      });
    }
    if (state.prediction && Number(state.prediction.timestamp) > lastPredictionTimestamp) {
      demoPreview = null; // Real input always takes priority over the preview.
      showPrediction(state.prediction);
    }
    else if (!state.prediction && state.mode !== 'demo') activeTouch = null;
}

function connectState() {
  const events = new EventSource('./api/events');
  events.onmessage = (event) => receiveState(JSON.parse(event.data));
  events.onerror = () => {
    activeTouch = null;
    touchTrail = [];
    statusElement.className = 'status error';
    statusElement.lastChild.textContent = ' Reconnecting';
  };
}

document.querySelector("#demo").addEventListener("click", () => {
  const placement = placements[Math.floor(Math.random() * placements.length)];
  if (!placement) return;
  demoPreview = { ...placement, intensity: 0.85, contact_area: 0.3, touch_type: "demo touch", touch_confidence: 0.94, location_confidence: 0.9 };
  demoUntil = Date.now() / 1000 + 2;
});
async function initialize() {
  const mapResponse = await fetch("./assets/sensor-map.json", { cache: "no-store" });
  if (!mapResponse.ok) throw new Error("Run five-sensor calibration first.");
  placements = (await mapResponse.json()).placements.sort((a, b) => a.channel - b.channel);
  new GLTFLoader().load("./assets/unitree-go2.glb", (gltf) => {
    const model = gltf.scene;
    const initial = new THREE.Box3().setFromObject(model);
    const center = initial.getCenter(new THREE.Vector3());
    const size = initial.getSize(new THREE.Vector3());
    model.position.sub(center);
    model.scale.setScalar(5.4 / Math.max(size.x, size.y, size.z));
    scene.add(model);
    modelRoot = model;
    modelBounds = new THREE.Box3().setFromObject(model);
    buildPointCloud(model);
    buildSensorMarkers();
    const finalSize = modelBounds.getSize(new THREE.Vector3());
    grid.position.y = modelBounds.min.y - 0.08;
    controls.target.set(0, modelBounds.min.y + finalSize.y * 0.48, 0);
    // This GLB's nose points toward +Z. Using -X as screen-up makes +Z run
    // right-to-left, so the real head/front is fixed on the left of the UI.
    // Keep the Y-up axis used when OrbitControls was constructed. View from
    // +X so the model's +Z nose appears left without rolling the camera.
    camera.up.set(0, 1, 0);
    const viewingSize = Math.max(finalSize.x, finalSize.z);
    camera.position.set(viewingSize*0.72, modelBounds.max.y+viewingSize*1.25, 0);
    controls.update();
    loading.remove();
  }, (event) => { if (event.total) loading.textContent = `Loading 3D model · ${Math.round(event.loaded / event.total * 100)}%`; }, () => { loading.textContent = "Could not load the model."; });
  connectState();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (demoPreview && Date.now()/1000 < demoUntil) {
    showPrediction({ ...demoPreview, timestamp: Date.now()/1000 });
  }
  if (Date.now() / 1000 - lastPredictionTimestamp > 0.18) {
    activeTouch = null;
    touchTrail = [];
    typeElement.textContent = "Waiting for touch";
    detailElement.textContent = "—";
    updateHeat();
  }
  updateHeat(); // fade every rendered frame, including when packets stop
  renderer.render(scene, camera);
}

initialize().catch((error) => { statusElement.className = "status error"; messageElement.textContent = error.message; });
animate();
