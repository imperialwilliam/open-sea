/* ============================================================================
   OPEN SEA — realtime procedural ocean
   WebGPURenderer + Three.js TSL node shaders. No textures, no build step.
   Sections: uniforms · waves · noise · sky · ocean · scene · controls ·
             time of day · creatures · UI · animation · init
   ============================================================================ */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
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
const uWaterOpacity = uniform(0.72); // slightly more opaque — better visual balance

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
let creatureGroup, creatures = [], wildlifeVisible = true, sunLight, hemiLight;
const HEADING_OFFSET = 0; // assumed model forward = +Z

function setupScene() {
  renderer = new THREE.WebGPURenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.outputColorSpace = THREE.SRGBColorSpace;   // critical for PBR materials
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color('#05070a');

  // ── Procedural environment map (no external files needed) ──
  // PBR MeshStandardMaterial renders pure black without scene.environment.
  // We generate a soft sky gradient equirectangular texture via canvas.
  const envSize = 256;
  const envCanvas = document.createElement('canvas');
  envCanvas.width = envSize * 2;
  envCanvas.height = envSize;
  const ctx = envCanvas.getContext('2d');
  // vertical gradient: horizon warm → zenith cool blue
  const gradV = ctx.createLinearGradient(0, 0, 0, envSize);
  gradV.addColorStop(0.0, '#1a3050');    // zenith: deep blue
  gradV.addColorStop(0.4, '#4a7090');     // upper sky
  gradV.addColorStop(0.55, '#8ab4c8');    // horizon
  gradV.addColorStop(0.65, '#c8a070');    // warm horizon glow
  gradV.addColorStop(0.8, '#3a5060');     // below horizon fade
  gradV.addColorStop(1.0, '#0a1520');     // ground: dark
  ctx.fillStyle = gradV;
  ctx.fillRect(0, 0, envCanvas.width, envCanvas.height);
  // subtle horizontal brightness variation (sun side)
  for (let y = 0; y < envSize; y++) {
    const t = y / envSize;
    const bright = Math.exp(-Math.pow((t - 0.58) * 3, 2)) * 25;
    if (bright > 0.5) {
      const imgData = ctx.getImageData(0, y, envCanvas.width, 1);
      for (let x = 0; x < imgData.data.length; x += 4) {
        const hx = x / 4 / envCanvas.width;
        const sunGlow = Math.exp(-Math.pow((hx - 0.7) * 2.5, 2)) * bright;
        imgData.data[x] = Math.min(255, imgData.data[x] + sunGlow);
        imgData.data[x+1] = Math.min(255, imgData.data[x+1] + sunGlow * 0.85);
        imgData.data[x+2] = Math.min(255, imgData.data[x+2] + sunGlow * 0.6);
      }
      ctx.putImageData(imgData, 0, y);
    }
  }
  const envTex = new THREE.CanvasTexture(envCanvas);
  envTex.mapping = THREE.EquirectangularReflectionMapping;
  envTex.colorSpace = THREE.SRGBColorSpace;
  scene.environment = envTex;

  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 8000);
  camera.position.set(0, 5.5, 17);

  // ocean
  const oceanGeometry = new THREE.PlaneGeometry(420, 420, 440, 440);
  oceanGeometry.rotateX(-Math.PI / 2);
  const oceanMaterial = new THREE.MeshBasicNodeMaterial();
  oceanMaterial.positionNode = wavePosition(positionLocal.xz, uTime, uSea);
  oceanMaterial.colorNode = oceanColor();
  oceanMaterial.transparent = true;
  oceanMaterial.opacityNode = uWaterOpacity;
  oceanMaterial.depthWrite = false;
  const ocean = new THREE.Mesh(oceanGeometry, oceanMaterial);
  ocean.frustumCulled = false;
  ocean.renderOrder = 1;
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

  // creature group — populated asynchronously by setupCreatures()
  creatureGroup = new THREE.Group();
  scene.add(creatureGroup);

  // ── Lighting for marine-life models (WebGPU PBR needs strong + env) ──
  // Three-light system: ambient (base fill) + hemi (sky/ground) + sun (key)
  const ambientLight = new THREE.AmbientLight(0x8899aa, 0.5);
  scene.add(ambientLight);
  sunLight = new THREE.DirectionalLight(0xffeedd, 3.5);   // warm sun, strong
  scene.add(sunLight);
  hemiLight = new THREE.HemisphereLight(0x9fc6ff, 0x05202a, 1.2);
  scene.add(hemiLight);

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
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.target.set(0, 1.5, 0);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.25;
  controls.update();
}

/* ---------------------------------------------------------------------------
   Time of day — DUSK <-> DAY palette interpolation driven by one slider
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

  const x = THREE.MathUtils.clamp(elevation / 0.42, 0, 1);
  const w = x * x * (3 - 2 * x);

  uZenithColor.value.copy(DUSK.zenith).lerp(DAY.zenith, w);
  uHorizonColor.value.copy(DUSK.horizon).lerp(DAY.horizon, w);
  uDeepColor.value.copy(DUSK.deep).lerp(DAY.deep, w);
  uShallowColor.value.copy(DUSK.shallow).lerp(DAY.shallow, w);
  const intensity = THREE.MathUtils.lerp(DUSK.intensity, DAY.intensity, w);
  uSunColor.value.copy(DUSK.sun).lerp(DAY.sun, w).multiplyScalar(intensity);

  // drive marine-life lights from the same time-of-day state (brighter range)
  if (sunLight) {
    sunLight.position.copy(uSunDir.value).multiplyScalar(120);
    // Decouple: color = pure sun hue (no intensity), intensity = separate scalar
    const sunHue = DUSK.sun.clone().lerp(DAY.sun, w);
    const hLen = Math.sqrt(sunHue.r * sunHue.r + sunHue.g * sunHue.g + sunHue.b * sunHue.b);
    if (hLen > 1e-6) { sunHue.r /= hLen; sunHue.g /= hLen; sunHue.b /= hLen; }
    sunLight.color.copy(sunHue);
    sunLight.intensity = THREE.MathUtils.lerp(2.2, 3.5, w);  // stronger base
  }
  if (hemiLight) {
    hemiLight.color.copy(uHorizonColor.value);
    hemiLight.groundColor.copy(uDeepColor.value);
    hemiLight.intensity = THREE.MathUtils.lerp(0.8, 1.4, w);  // was 0.6-1.2
  }

  timeValueEl.textContent = timeLabel(t);
}

/* ===========================================================================
   MARINE LIFE — CC0 whale + shark with materials, realistic scale,
   wave-surface riding, smooth breach animation, and banking turns.

   ASSET PROVENANCE (evaluated 2026-07-30):
   Source: Quaternius "Animated Fish Bundle" (Poly Pizza mirror), Public Domain (CC0).
   - whale.glb : model id JGFwp6xWgk — clip "Armature|Swim", 20-node rig
   - shark.glb : model id 3LzFgI3GLO — clip "Armature|Swim", 20-node rig
   Why chosen over alternatives:
   - CC0 (no attribution, fully commercial) — beats Sketchfab CC-BY / CC-BY-NC
   - GLB/glTF — directly loadable by the project's GLTFLoader
   - Both species rigged + swimming animation — addresses "no reasonable physics"
   - Consistent low-poly style matches the stylized WebGPU ocean
   Rejected: Smithsonian scans (CC0 but 100s of MB, no rig, not game-ready);
   DigitalLife3D (ultra-HD but non-commercial license); Kenney (no sea-mammal pack).
   =========================================================================== */

// ── Species configs (real-world-inspired proportions & behaviour) ──
// Finback whale ~20m → targetLen 18 | Great white shark ~5m → targetLen 3.5
// Ratio ≈ 5:1 (previously 2:1 — shark was half the whale's size, wrong).
const CREATURE_CFG = {
  whale: {
    radius: 60, speed: 0.028, baseY: -3.5,     // deeper — whale swims below surface
    bobFreq: [0.37, 0.61], bobAmp: [0.25, 0.12], // gentler bob when submerged
    breach: false, targetLen: 18,   // breach disabled — user wants natural swimming only
    breachPeriod: 16, breachRiseFrac: 0.30, breachPeakFrac: 0.12,
    breachHeight: 10.0, breachPitch: 0.65,
    color: new THREE.Color(0x1a252e), roughness: 0.78, metalness: 0.06,
    bankAngle: 0.08
  },
  shark: {
    radius: 30, speed: 0.07, baseY: -1.4,
    bobFreq: [0.55, 0.91], bobAmp: [0.18, 0.09],
    breach: false, targetLen: 3.5,
    finBreak: true, finBreakAmp: 0.7, finBreakPeriod: 7,
    color: new THREE.Color(0x2e2e2e), roughness: 0.62, metalness: 0.10,
    bankAngle: 0.20
  }
};

// ── BOAT_CFG ──
// Fishing vessel = Quaternius "Ship" (CC0; mirrored on Poly Pizza as GLB so it
// drops in with no conversion). Rides ON the wave surface (baseY ≈ 0), tilts with
// the wave slope so it floats naturally, and never tips onto its side.
const BOAT_CFG = {
  radius: 45, speed: 0.011, baseY: 0.6,        // floats at surface, slow patrol
  bobFreq: [0.35, 0.62], bobAmp: [0.14, 0.08],
  targetLen: 14,
  keepMaterial: true,   // preserve boat's own PBR + texture (hull / sails)
  keepUp: true,         // Y stays up; longest horizontal axis aligns to +Z
  waveTilt: true, tiltGain: 1.6,  // pitch/roll follow wave slope → floats naturally
  bankAngle: 0
};

// ── Material post-processing ──
// glTF MeshStandardMaterial renders black in WebGPU when lights are weak or
// material properties are extreme. We force sane PBR values + tiny emissive floor.
function applyCreatureMaterial(mesh, cfg) {
  if (!mesh.isMesh) return;
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (let mat of mats) {
    if (!mat) { mat = new THREE.MeshStandardMaterial(); }
    mat.color.copy(cfg.color);
    mat.roughness = Math.min(Math.max(cfg.roughness, 0.01), 1.0);
    mat.metalness = Math.min(Math.max(cfg.metalness, 0.0), 1.0);
    mat.emissive = cfg.color.clone().multiplyScalar(0.06);   // visible base glow
    mat.emissiveIntensity = 1.0;
    mat.envMapIntensity = 1.2;    // stronger env reflection
    mat.needsUpdate = true;
  }
  if (Array.isArray(mesh.material)) return;
  mesh.material = mats[0]; // reassign in case we replaced it
}

// ── Easing helpers ──
function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function easeInCubic(t)  { return t * t * t; }

// ── Wave height sampler (JS mirror of TSL Gerstner waves) ──
// Returns approximate wave Y at (x, z, time). Close enough for creature riding.
const _WAVE_AMP = [0.12/0.105, 0.12/0.203, 0.09/0.349, 0.07/0.661, 0.05/1.257];
const _WAVE_K   = [0.105, 0.203, 0.349, 0.661, 1.257];
const _WAVE_C   = [1.013, 1.409, 1.850, 2.546, 3.509]; // = sqrt(9.8 * k), matches TSL WAVES[].c
const _WAVE_DIR = [[1,0],[0.6,0.8],[-0.7,0.7],[0.3,-0.95],[-0.35,-0.94]];

function sampleWaveY(x, z, time) {
  let y = 0;
  const sv = uSea.value;
  for (let i = 0; i < 5; i++) {
    const d = _WAVE_DIR[i], len = Math.hypot(d[0], d[1]);
    const f = _WAVE_K[i] * (d[0]/len*x + d[1]/len*z - time*_WAVE_C[i]);
    y += (_WAVE_AMP[i] * sv) * Math.sin(f);
  }
  return y;
}

// ── Creature factory ──
function addCreature(gltf, cfg) {
  const model = gltf.scene;

  // Apply materials to all meshes BEFORE scene graph manipulation
  // (boats keep their own PBR + texture via keepMaterial:true)
  if (!cfg.keepMaterial) model.traverse((child) => { applyCreatureMaterial(child, cfg); });

  model.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(model);
  const size = new THREE.Vector3(); box.getSize(size);
  const center = new THREE.Vector3(); box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const s = cfg.targetLen / maxDim;
  model.scale.setScalar(s);

  // Align longest axis to root-group +Z ("forward").
  // Boats use keepUp: keep Y up so a tall mast never tips the hull onto its side.
  if (cfg.keepUp) {
    if (size.x >= size.z) model.rotation.y = -Math.PI / 2;
  } else {
    if (size.x >= size.z && size.x >= size.y) model.rotation.y = -Math.PI / 2;
    else if (size.y >= size.x && size.y >= size.z) model.rotation.x = Math.PI / 2;
  }

  // Recenter AFTER rotation so the pivot aligns with the oriented bounding box
  model.updateMatrixWorld(true);
  const box2 = new THREE.Box3().setFromObject(model);
  const center2 = new THREE.Vector3(); box2.getCenter(center2);

  const root = new THREE.Group();
  root.add(model);
  model.position.copy(center2).negate(); // recenter: place bbox center at root origin
  creatureGroup.add(root);

  const c = {
    root, type: cfg.type, radius: cfg.radius, speed: cfg.speed,
    baseY: cfg.baseY, bobFreq: cfg.bobFreq, bobAmp: cfg.bobAmp,
    breach: cfg.breach, breachPeriod: cfg.breachPeriod||0,
    breachRiseFrac: cfg.breachRiseFrac||0, breachPeakFrac: cfg.breachPeakFrac||0,
    breachHeight: cfg.breachHeight||0, breachPitch: cfg.breachPitch||0,
    finBreak: cfg.finBreak||false, finBreakAmp: cfg.finBreakAmp||0,
    finBreakPeriod: cfg.finBreakPeriod||0, bankAngle: cfg.bankAngle||0,
    keepMaterial: cfg.keepMaterial||false, keepUp: cfg.keepUp||false,
    waveTilt: cfg.waveTilt||false, tiltGain: cfg.tiltGain||0,
    angle: Math.random() * Math.PI * 2, phase: Math.random() * Math.PI * 2,
    mixer: null
  };

  // Set up AnimationMixer for models with built-in clips (shark)
  if (gltf.animations && gltf.animations.length) {
    const mixer = new THREE.AnimationMixer(model);
    const clip = gltf.animations.find(a => /swim/i.test(a.name)) || gltf.animations[0];
    mixer.clipAction(clip).play();
    c.mixer = mixer;
  }
  creatures.push(c);
}

async function setupCreatures() {
  const loader = new GLTFLoader();
  try { addCreature(await loader.loadAsync('./assets/whale.glb'), { type:'whale', ...CREATURE_CFG.whale }); }
  catch (e) { console.error('[open-sea] whale failed:', e); }
  try { addCreature(await loader.loadAsync('./assets/shark.glb'), { type:'shark', ...CREATURE_CFG.shark }); }
  catch (e) { console.error('[open-sea] shark failed:', e); }
  try { addCreature(await loader.loadAsync('./assets/boat.glb'), { type:'boat', ...BOAT_CFG }); }
  catch (e) { console.error('[open-sea] boat failed:', e); }
}

// ── Per-frame creature update ──
function updateCreatures(time, dt) {
  creatureGroup.visible = wildlifeVisible;
  if (!wildlifeVisible) return;

  for (const c of creatures) {
    // Advance patrol angle
    c.angle += c.speed * dt;
    const r = c.radius;
    const x = Math.cos(c.angle) * r;
    const z = Math.sin(c.angle) * r;

    // Sample ocean wave height at this position → creature rides the surface
    const waveY = sampleWaveY(x, z, time);

    // Multi-frequency organic undulation (sum of 2 sine waves at different rates)
    const bob1 = Math.sin(time * c.bobFreq[0] + c.phase) * c.bobAmp[0];
    const bob2 = Math.sin(time * c.bobFreq[1] + c.phase * 1.7) * c.bobAmp[1];
    let y = c.baseY + waveY + bob1 + bob2;
    let pitch = 0, roll = 0;

    // ── Whale breach: smooth eased arc (slow rise → peak hang → fast fall) ──
    if (c.breach) {
      const P = c.breachPeriod;
      const cycle = (((time / P) + (c.phase / (Math.PI * 2))) % 1 + 1) % 1;
      if (cycle < c.breachRiseFrac) {
        // Slow ascent (ease-out cubic)
        const t = cycle / c.breachRiseFrac, arc = easeOutCubic(t);
        y += arc * c.breachHeight;
        pitch = -arc * c.breachPitch;
      } else if (cycle < c.breachRiseFrac + c.breachPeakFrac) {
        // Peak hang — nearly full height, slight forward-tilt decay
        const pt = (cycle - c.breachRiseFrac) / c.breachPeakFrac;
        const arc = 1.0 - ((pt * pt * (3 - 2 * pt))) * 0.08;  // smoothstep settle
        y += arc * c.breachHeight;
        pitch = -(1.0 - pt * 0.3) * c.breachPitch;
      } else {
        // Fast fall + submerged recovery (ease-in cubic, continuous from peak)
        const ft = (cycle - c.breachRiseFrac - c.breachPeakFrac) / (1 - c.breachRiseFrac - c.breachPeakFrac);
        const arc = 1.0 - easeInCubic(ft);  // starts at 1.0, decays smoothly to 0
        y += arc * c.breachHeight * Math.max(0, 1 - ft * 1.2);
        // Pitch transitions smoothly: nose-up at peak → level → slight nose-down
        pitch = (1 - ft) * -c.breachPitch * 0.4;  // continuous from peak pitch
      }
    }

    // ── Shark dorsal-fin surface break ──
    if (c.finBreak) {
      const fc = (((time / c.finBreakPeriod) + c.phase) % 1 + 1) % 1;
      if (fc < 0.15) {
        // Brief moment where dorsal fin breaks surface
        y += Math.sin((fc / 0.15) * Math.PI) * c.finBreakAmp;
      }
    }

    // Banking into turns (roll = Z-axis rotation)
    roll = Math.sin(time * 0.13 + c.phase * 2.0) * c.bankAngle;

    // ── Boat: pitch/roll follow local wave slope → floats on the surface ──
    if (c.waveTilt) {
      const dx = 2.5, dz = 2.5;
      const slopeX = (sampleWaveY(x + dx, z, time) - sampleWaveY(x - dx, z, time)) / (2 * dx);
      const slopeZ = (sampleWaveY(x, z + dz, time) - sampleWaveY(x, z - dz, time)) / (2 * dz);
      pitch += -slopeZ * c.tiltGain;   // nose up/down with the swell ahead
      roll  += -slopeX * c.tiltGain;   // heel with the cross-swell
    }

    // Apply transform
    c.root.position.set(x, y, z);
    const tx = -Math.sin(c.angle), tz = Math.cos(c.angle);
    c.root.rotation.set(pitch, Math.atan2(tx, tz) + HEADING_OFFSET, roll);

    // Advance animation mixer
    if (c.mixer) c.mixer.update(dt);
  }
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
const wildlifeButtonEl = document.getElementById('wildlife-button');
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

  wildlifeButtonEl.addEventListener('click', () => {
    wildlifeVisible = !wildlifeVisible;
    wildlifeButtonEl.classList.toggle('active', wildlifeVisible);
    wildlifeButtonEl.setAttribute('aria-pressed', String(wildlifeVisible));
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
  uTime.value += Math.min(rawDt, 0.1);

  updateCreatures(uTime.value, Math.min(rawDt, 0.1));

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
      lastNow = performance.now();
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
    setupCreatures(); // async; creatures pop in once models load
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
