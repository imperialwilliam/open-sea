/* ============================================================================
   OPEN SEA — realtime procedural ocean
   WebGPURenderer + Three.js TSL node shaders. No textures, no build step.
   Sections: uniforms · waves · noise · sky · ocean · scene · controls ·
             time of day · UI · animation · init
   ============================================================================ */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import {
  Fn, pass, uniform, float, vec2, vec3, vec4,
  sin, cos, dot, cross, normalize, mix, pow, max, clamp,
  fract, floor, smoothstep, distance, reflect,
  positionLocal, positionWorld, cameraPosition
} from 'three/tsl';

/* ---------------------------------------------------------------------------
   Uniforms — shared scene state (ocean + sky read the same values)
   --------------------------------------------------------------------------- */
const uTime = uniform(0.0);
const uSea = uniform(0.925);
const uSunDir = uniform(new THREE.Vector3(0, 1, 0));
const uSunColor = uniform(new THREE.Color(1, 1, 1));      // base color × intensity
const uHorizonColor = uniform(new THREE.Color(0.52, 0.68, 0.82));
const uZenithColor = uniform(new THREE.Color(0.07, 0.2, 0.42));
const uDeepColor = uniform(new THREE.Color(0.015, 0.09, 0.11));
const uShallowColor = uniform(new THREE.Color(0.06, 0.32, 0.36));

/* ---------------------------------------------------------------------------
   Waves — five directional Gerstner components, constants precomputed on CPU
   --------------------------------------------------------------------------- */
const WAVE_DEFS = [
  { dir: [ 1.0,   0.0  ], wavelength: 60.0, steepness: 0.12 },
  { dir: [ 0.6,   0.8  ], wavelength: 31.0, steepness: 0.12 },
  { dir: [-0.7,   0.7  ], wavelength: 18.0, steepness: 0.09 },
  { dir: [ 0.3,  -0.95 ], wavelength:  9.5, steepness: 0.07 },
  { dir: [-0.35, -0.94 ], wavelength:  5.0, steepness: 0.05 }
];

const WAVES = WAVE_DEFS.map(({ dir, wavelength, steepness }) => {
  const len = Math.hypot(dir[0], dir[1]);
  const k = (Math.PI * 2) / wavelength;
  return {
    dir: vec2(dir[0] / len, dir[1] / len),
    k: float(k),
    c: float(Math.sqrt(9.8 * k)),      // angular propagation term
    amp: float(steepness / k),         // a = steepness / k (× live sea state)
    steep: float(steepness)            // a·k, used by analytic derivatives
  };
});

// Sum of Gerstner displacements. xz = surface coordinate, sea = live sea state.
const wavePosition = Fn(([xz, time, sea]) => {
  const pos = vec3(xz.x, 0.0, xz.y).toVar();
  for (const w of WAVES) {
    const a = w.amp.mul(sea);
    const f = w.k.mul(dot(w.dir, xz).sub(time.mul(w.c)));
    pos.x.addAssign(a.mul(w.dir.x).mul(cos(f)));
    pos.y.addAssign(a.mul(sin(f)));
    pos.z.addAssign(a.mul(w.dir.y).mul(cos(f)));
  }
  return pos;
});

// Analytic tangent/binormal accumulation → stable large-scale swell normal.
const waveNormal = Fn(([xz, time, sea]) => {
  const tangent = vec3(1.0, 0.0, 0.0).toVar();
  const binormal = vec3(0.0, 0.0, 1.0).toVar();
  for (const w of WAVES) {
    const wa = w.steep.mul(sea);       // = a·k with live sea state
    const f = w.k.mul(dot(w.dir, xz).sub(time.mul(w.c)));
    const sf = sin(f);
    const cf = cos(f);
    tangent.x.addAssign(wa.negate().mul(w.dir.x).mul(w.dir.x).mul(sf));
    tangent.y.addAssign(wa.mul(w.dir.x).mul(cf));
    tangent.z.addAssign(wa.negate().mul(w.dir.x).mul(w.dir.y).mul(sf));
    binormal.x.addAssign(wa.negate().mul(w.dir.x).mul(w.dir.y).mul(sf));
    binormal.y.addAssign(wa.mul(w.dir.y).mul(cf));
    binormal.z.addAssign(wa.negate().mul(w.dir.y).mul(w.dir.y).mul(sf));
  }
  return normalize(cross(binormal, tangent));
});

// Signed crest height — drives coloring, SSS glow and foam.
const waveCrest = Fn(([xz, time, sea]) => {
  const crest = float(0.0).toVar();
  for (const w of WAVES) {
    const f = w.k.mul(dot(w.dir, xz).sub(time.mul(w.c)));
    crest.addAssign(w.amp.mul(sea).mul(sin(f)));
  }
  return crest;
});

/* ---------------------------------------------------------------------------
   Noise — 2D gradient noise + 3-octave FBM, fully in TSL
   --------------------------------------------------------------------------- */
const hash2 = Fn(([p]) => {
  const v = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return fract(sin(v).mul(43758.5453)).mul(2.0).sub(1.0);
});

const gradNoise = Fn(([p]) => {
  const i = floor(p);
  const f = fract(p);
  // quintic fade: f³·(f·(f·6−15)+10)
  const u = f.mul(f).mul(f).mul(f.mul(f.mul(6.0).sub(15.0)).add(10.0));
  const n00 = dot(hash2(i), f);
  const n10 = dot(hash2(i.add(vec2(1.0, 0.0))), f.sub(vec2(1.0, 0.0)));
  const n01 = dot(hash2(i.add(vec2(0.0, 1.0))), f.sub(vec2(0.0, 1.0)));
  const n11 = dot(hash2(i.add(vec2(1.0, 1.0))), f.sub(vec2(1.0, 1.0)));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
});

const fbm = Fn(([p]) => {
  return gradNoise(p)
    .add(gradNoise(p.mul(2.04).add(vec2(17.3, 9.1))).mul(0.5))
    .add(gradNoise(p.mul(4.11).add(vec2(42.7, 28.6))).mul(0.25));
});

// Animated capillary-scale detail height field (two drifting FBM reads).
const detailHeight = Fn(([xz, time]) => {
  const driftA = vec2(time.mul(0.55), time.mul(0.32));
  const driftB = vec2(time.mul(-0.4), time.mul(0.5));
  return fbm(xz.mul(0.85).add(driftA)).add(fbm(xz.mul(2.1).add(driftB)).mul(0.45));
});

/* ---------------------------------------------------------------------------
   Sky — one analytic function shared by the dome and the water reflection
   --------------------------------------------------------------------------- */
const skyColor = Fn(([dir]) => {
  const d = normalize(dir);
  const up = clamp(d.y, -0.15, 1.0);
  const col = mix(uHorizonColor, uZenithColor, pow(max(up, 0.0), 0.42)).toVar();

  // below-horizon haze
  const haze = smoothstep(0.0, -0.15, d.y);
  col.assign(mix(col, uDeepColor.mul(1.4).add(uHorizonColor.mul(0.25)), haze));

  const s = max(dot(d, uSunDir), 0.0);
  col.addAssign(uSunColor.mul(pow(s, 10.0).mul(0.18)));                        // wide halo
  col.addAssign(uSunColor.mul(smoothstep(0.9994, 0.9998, s).mul(30.0)));      // sun disc
  return col;
});

// Sky dome color = shared sky + a slow procedural cloud band near the horizon.
const skyDomeColor = Fn(() => {
  const dir = normalize(positionLocal);
  const col = skyColor(dir).toVar();

  const band = smoothstep(0.03, 0.16, dir.y).mul(smoothstep(0.6, 0.22, dir.y));
  const cloudUV = dir.xz.div(dir.y.add(0.18)).mul(0.55);
  const cloudDrift = vec2(uTime.mul(0.006), uTime.mul(0.003));
  const cloudNoise = fbm(cloudUV.add(cloudDrift)).mul(0.5).add(0.5);
  const cloudMask = smoothstep(0.62, 0.95, cloudNoise).mul(band);

  // warm pale cloud tint derived from the sun hue (uSunColor carries intensity)
  const sunTint = normalize(uSunColor.add(vec3(0.0001)));
  const cloudColor = vec3(0.92, 0.90, 0.87).mul(mix(vec3(1.0), sunTint, 0.35));
  col.assign(mix(col, cloudColor, clamp(cloudMask, 0.0, 1.0).mul(0.6)));
  return vec4(col, 1.0);
});

/* ---------------------------------------------------------------------------
   Ocean — all apparent lighting built manually in a TSL color node
   --------------------------------------------------------------------------- */
const oceanColor = Fn(() => {
  const P = positionWorld;
  const xz = P.xz;

  // large-scale analytic swell normal + finite-difference FBM detail normal
  const N0 = waveNormal(xz, uTime, uSea);
  const eps = 0.1;
  const h0 = detailHeight(xz, uTime);
  const hx = detailHeight(xz.add(vec2(eps, 0.0)), uTime);
  const hz = detailHeight(xz.add(vec2(0.0, eps)), uTime);
  const detailGain = float(1.5).mul(uSea.mul(0.6).add(0.4));
  const N = normalize(N0.add(vec3(h0.sub(hx), 0.0, h0.sub(hz)).mul(detailGain)));

  const V = normalize(cameraPosition.sub(P));
  const crest = waveCrest(xz, uTime, uSea);

  const baseColor = mix(uDeepColor, uShallowColor, clamp(crest.mul(0.35).add(0.45), 0.0, 1.0));

  // back-lit crest glow, tinted by shallow water and sun, added pre-Fresnel
  const glow = pow(max(dot(V, uSunDir), 0.0), 3.0).mul(max(crest, 0.0)).mul(0.18);
  const waterColor = baseColor.add(uShallowColor.mul(uSunColor).mul(glow)).toVar();

  // Fresnel sky reflection (reflection ray kept above the horizon)
  const R = reflect(V.negate(), N);
  const RDir = normalize(vec3(R.x, max(R.y, 0.04), R.z));
  const reflection = skyColor(RDir);
  const fresnel = float(0.02).add(float(0.98).mul(pow(max(dot(N, V), 0.0).oneMinus(), 5.0)));
  const col = mix(waterColor, reflection, fresnel).toVar();

  // sun glitter: tight noise-modulated sparkle lobe + broad sheen
  const H = normalize(uSunDir.add(V));
  const ndh = max(dot(N, H), 0.0);
  const sparkleNoise = fbm(xz.mul(1.7).add(vec2(uTime.mul(0.9), uTime.mul(-0.7)))).mul(0.5).add(0.5);
  const sparkle = pow(ndh, 500.0).mul(mix(0.4, 3.4, sparkleNoise));
  const sheen = pow(ndh, 48.0).mul(0.12);
  col.addAssign(uSunColor.mul(sparkle.add(sheen)));

  // sparse breaking foam on wave crests
  const foamNoise = fbm(xz.mul(1.1).add(vec2(uTime.mul(0.22), uTime.mul(0.14)))).mul(0.5).add(0.5);
  const foamMask = smoothstep(0.5, 0.95, foamNoise).mul(smoothstep(1.0, 2.0, crest));
  col.assign(mix(col, vec3(0.82, 0.88, 0.90), clamp(foamMask.mul(0.85), 0.0, 1.0)));

  // horizon occlusion — hides the finite edge of the ocean plane
  const camDist = distance(cameraPosition, P);
  col.assign(mix(col, uHorizonColor, smoothstep(150.0, 290.0, camDist)));

  return vec4(col, 1.0);
});

/* ---------------------------------------------------------------------------
   Scene setup — renderer, camera, meshes, post-processing
   --------------------------------------------------------------------------- */
let renderer, scene, camera, controls, postProcessing;

function setupScene() {
  renderer = new THREE.WebGPURenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color('#05070a'); // safe init color

  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 8000);
  camera.position.set(0, 5.5, 17);

  // ocean
  const oceanGeometry = new THREE.PlaneGeometry(420, 420, 440, 440);
  oceanGeometry.rotateX(-Math.PI / 2);
  const oceanMaterial = new THREE.MeshBasicNodeMaterial();
  oceanMaterial.positionNode = wavePosition(positionLocal.xz, uTime, uSea);
  oceanMaterial.colorNode = oceanColor();
  const ocean = new THREE.Mesh(oceanGeometry, oceanMaterial);
  ocean.frustumCulled = false; // vertex displacement must never cull the mesh
  scene.add(ocean);

  // sky dome
  const skyGeometry = new THREE.SphereGeometry(4000, 48, 24);
  const skyMaterial = new THREE.MeshBasicNodeMaterial();
  skyMaterial.side = THREE.BackSide;
  skyMaterial.depthWrite = false;
  skyMaterial.colorNode = skyDomeColor();
  const sky = new THREE.Mesh(skyGeometry, skyMaterial);
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  scene.add(sky);

  // post-processing: scene pass + TSL bloom
  postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  postProcessing.outputNode = sceneColor.add(bloom(sceneColor, 0.4, 0.3, 0.9));
}

/* ---------------------------------------------------------------------------
   Controls — damped orbit / zoom, slow auto-rotation
   --------------------------------------------------------------------------- */
function setupControls() {
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 4;
  controls.maxDistance = 120;
  controls.minPolarAngle = 0.15;
  controls.maxPolarAngle = Math.PI * 0.495; // camera can never dip below the surface
  controls.target.set(0, 1.5, 0);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.25;
  controls.update();
}

/* ---------------------------------------------------------------------------
   Time of day — DUSK ↔ DAY palette interpolation driven by one slider
   --------------------------------------------------------------------------- */
const DAY = {
  zenith: new THREE.Color(0.07, 0.20, 0.42),
  horizon: new THREE.Color(0.52, 0.68, 0.82),
  sun: new THREE.Color(1.0, 0.93, 0.80),
  intensity: 1.6,
  deep: new THREE.Color(0.015, 0.09, 0.11),
  shallow: new THREE.Color(0.06, 0.32, 0.36)
};
const DUSK = {
  zenith: new THREE.Color(0.03, 0.05, 0.16),
  horizon: new THREE.Color(0.85, 0.36, 0.16),
  sun: new THREE.Color(1.0, 0.42, 0.14),
  intensity: 2.6,
  deep: new THREE.Color(0.02, 0.045, 0.075),
  shallow: new THREE.Color(0.09, 0.15, 0.20)
};

function timeLabel(t) {
  if (t < 0.12) return 'DUSK';
  if (t < 0.30) return 'GOLDEN HOUR';
  if (t < 0.62) return 'AFTERNOON';
  return 'MIDDAY';
}

function applyTimeOfDay(t) {
  const elevation = THREE.MathUtils.lerp(-0.05, 0.62, t);
  const azimuth = THREE.MathUtils.lerp(-0.9, 0.9, t);
  uSunDir.value.set(
    Math.cos(elevation) * Math.sin(azimuth),
    Math.sin(elevation),
    -Math.cos(elevation) * Math.cos(azimuth)
  );

  // daylight weight: cubic smoothstep of elevation from 0 → 0.42
  const x = THREE.MathUtils.clamp(elevation / 0.42, 0, 1);
  const w = x * x * (3 - 2 * x);

  uZenithColor.value.copy(DUSK.zenith).lerp(DAY.zenith, w);
  uHorizonColor.value.copy(DUSK.horizon).lerp(DAY.horizon, w);
  uDeepColor.value.copy(DUSK.deep).lerp(DAY.deep, w);
  uShallowColor.value.copy(DUSK.shallow).lerp(DAY.shallow, w);
  const intensity = THREE.MathUtils.lerp(DUSK.intensity, DAY.intensity, w);
  uSunColor.value.copy(DUSK.sun).lerp(DAY.sun, w).multiplyScalar(intensity);

  timeValueEl.textContent = timeLabel(t);
}

/* ---------------------------------------------------------------------------
   UI — DOM controls write straight into TSL uniforms (no rebuilds)
   --------------------------------------------------------------------------- */
const loaderEl = document.getElementById('loader');
const loaderTitleEl = document.getElementById('loader-title');
const loaderStatusEl = document.getElementById('loader-status');
const loaderMessageEl = document.getElementById('loader-message');
const seaRangeEl = document.getElementById('sea-range');
const seaValueEl = document.getElementById('sea-value');
const timeRangeEl = document.getElementById('time-range');
const timeValueEl = document.getElementById('time-value');
const driftButtonEl = document.getElementById('drift-button');
const fpsValueEl = document.getElementById('fps-value');

function setupUI() {
  const applySeaState = () => {
    uSea.value = 0.25 + (Number(seaRangeEl.value) / 100) * 1.5;
    seaValueEl.textContent = uSea.value.toFixed(2);
  };
  seaRangeEl.addEventListener('input', applySeaState);
  applySeaState();

  timeRangeEl.addEventListener('input', () => applyTimeOfDay(Number(timeRangeEl.value) / 100));
  applyTimeOfDay(Number(timeRangeEl.value) / 100);

  driftButtonEl.addEventListener('click', () => {
    controls.autoRotate = !controls.autoRotate;
    driftButtonEl.classList.toggle('active', controls.autoRotate);
    driftButtonEl.setAttribute('aria-pressed', String(controls.autoRotate));
  });
}

function showError(title, message) {
  loaderEl.classList.add('error');
  loaderTitleEl.textContent = title;
  loaderStatusEl.textContent = '';
  loaderMessageEl.textContent = message;
}

/* ---------------------------------------------------------------------------
   Animation — one loop, clamped dt, FPS telemetry, lifecycle handling
   --------------------------------------------------------------------------- */
let lastNow = 0;
let fpsTime = 0;
let fpsFrames = 0;
let revealed = false;

function revealUI() {
  revealed = true;
  loaderEl.classList.add('done');
  document.body.classList.add('ready');
}

async function frame() {
  const now = performance.now();
  const rawDt = (now - lastNow) / 1000;
  lastNow = now;
  uTime.value += Math.min(rawDt, 0.1); // clamp: no shader-time jumps after a stall

  controls.update();
  await postProcessing.renderAsync();

  fpsTime += rawDt;
  fpsFrames += 1;
  if (fpsTime >= 0.5) {
    fpsValueEl.textContent = String(Math.round(fpsFrames / fpsTime));
    fpsTime = 0;
    fpsFrames = 0;
  }

  if (!revealed) revealUI();
}

function setupLifecycle() {
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      renderer.setAnimationLoop(null);
    } else {
      lastNow = performance.now(); // reset so shader time never jumps
      renderer.setAnimationLoop(frame);
    }
  });
}

/* ---------------------------------------------------------------------------
   Init — WebGPU capability check, then boot
   --------------------------------------------------------------------------- */
async function init() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError(
      'WEBGPU UNAVAILABLE',
      'This experience renders with WebGPU. Please open it in a current version of Chrome or Edge with hardware acceleration enabled.'
    );
    return;
  }

  try {
    setupScene();
    setupControls();
    setupUI();
    setupLifecycle();
    await renderer.init();
    lastNow = performance.now();
    renderer.setAnimationLoop(frame);
  } catch (err) {
    console.error('[open-sea] initialization failed:', err);
    const detail = err && err.message ? err.message : String(err);
    showError(
      'INITIALIZATION FAILED',
      `${detail} — check that hardware acceleration is enabled and that your browser and GPU driver support WebGPU.`
    );
  }
}

init();
