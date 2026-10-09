/**
 * Akash Setu — 3D Earth & Satellite Orbit Visualization  (v2)
 * ============================================================
 * Changes from v1:
 *   - Earth no longer auto-rotates (cosmetic rotation removed)
 *   - Persistent orbitLines Map: paths never destroyed on selection change
 *   - All station orbits loaded at startup without any click required
 *   - Active satellite orbits loaded for the current display batch
 *   - Selection HIGHLIGHTS an existing orbit line (color/opacity change only)
 *   - Deselection restores normal appearance; all other paths remain visible
 *   - Camera target is always scene origin (Earth center stays fixed)
 *
 * Coordinate system (unchanged from v1):
 *   ECEF X -> Three.js +X  (0 lon, equator)
 *   ECEF Y -> Three.js +Z  (90E, equator)
 *   ECEF Z -> Three.js +Y  (north pole = up)
 *   Scale: 1 unit = 6371 km
 *
 * All SGP4 propagation is server-side (Flask /api/propagate_one,
 * /api/propagate, /api/orbit_batch). Positions are estimates only.
 */

// ─── Constants ────────────────────────────────────────────────────────────────
const API_BASE             = '';
const EARTH_RADIUS_KM      = 6371;
const SCALE                = 1 / EARTH_RADIUS_KM;
const REFRESH_INTERVAL_MS  = 30000;
const DEFAULT_SAT_LIMIT    = 300;   // batch propagation size
const ORBIT_DISPLAY_LIMIT  = 300;   // how many active-sat orbits to pre-render
const ORBIT_DIM_COLOR_SAT     = 0x2a5caa;  // normal active-sat orbit
const ORBIT_DIM_COLOR_STATION = 0x00995a;  // normal station orbit
const ORBIT_DIM_OPACITY    = 0.30;
const ORBIT_SEL_COLOR_SAT     = 0x80c0ff;  // selected active-sat orbit
const ORBIT_SEL_COLOR_STATION = 0x00ffaa;  // selected station orbit
const ORBIT_SEL_OPACITY    = 0.90;

// ─── State ────────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let earthMesh, atmosphereMesh;
let satPoints       = null;   // Points object for all satellite positions
let orbitLines      = new Map();   // norad_id -> THREE.Line  (persistent)
let selectedMarker  = null;   // pulsing sphere for the selected satellite
let satellites      = [];     // metadata from /api/satellites
let positions       = [];     // current ECEF positions from /api/propagate
let selectedNoradId = null;
let searchQuery     = '';
let sourceFilter    = 'all';
let refreshTimer    = null;

// ─── Initialization ───────────────────────────────────────────────────────────
async function init() {
    setupScene();
    buildEarth();
    buildStars();
    buildLighting();
    buildControls();
    animate();

    try {
        await loadSatellites();
        await refreshPositions();          // positions + batch orbits
        startAutoRefresh();
        hideLoading();
    } catch (err) {
        console.error('Init error:', err);
        const el = document.querySelector('.loading-text');
        if (el) el.textContent = 'Could not reach API. Run: python backend/app.py';
    }
    setupUIListeners();
}

// ─── Scene setup ─────────────────────────────────────────────────────────────
function setupScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 1000);
    camera.position.set(0, 0.5, 3.8);

    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('globe-canvas'),
        antialias: true,
    });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x000510);

    window.addEventListener('resize', () => {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
    });
}

// ─── Earth ───────────────────────────────────────────────────────────────────
function buildEarth() {
    const tc = document.createElement('canvas');
    tc.width = 2048; tc.height = 1024;
    const ctx = tc.getContext('2d');

    // Ocean gradient
    const oceanGrad = ctx.createLinearGradient(0, 0, 0, 1024);
    oceanGrad.addColorStop(0,   '#0c1e3a');
    oceanGrad.addColorStop(0.5, '#0e2244');
    oceanGrad.addColorStop(1,   '#091830');
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, 2048, 1024);

    // Continent patches (approximate procedural)
    const patches = [
        [[370,190],[500,170],[590,200],[620,290],[580,390],[510,420],[440,400],[375,350],[350,270]],
        [[520,420],[565,450],[590,530],[575,640],[548,730],[505,710],[480,620],[488,500]],
        [[910,175],[1000,160],[1060,195],[1050,265],[980,290],[920,255],[905,210]],
        [[945,300],[1010,275],[1075,310],[1100,450],[1060,570],[990,600],[945,535],[920,410]],
        [[1060,150],[1250,130],[1430,170],[1480,250],[1450,360],[1340,340],[1240,380],[1130,320],[1070,260]],
        [[1140,340],[1200,320],[1230,400],[1210,490],[1165,510],[1130,450],[1120,380]],
        [[1310,370],[1350,360],[1370,440],[1340,480],[1305,450]],
        [[1360,530],[1460,505],[1520,550],[1510,630],[1440,660],[1360,615],[1340,565]],
        [[650,110],[730,95],[790,130],[780,190],[720,210],[655,180]],
        [[0,920],[2048,920],[2048,1024],[0,1024]],
    ];
    for (const pts of patches) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
            const p = pts[i-1], q = pts[i];
            ctx.bezierCurveTo(
                p[0]+(q[0]-p[0])*0.3+(Math.random()-0.5)*18,
                p[1]+(q[1]-p[1])*0.1+(Math.random()-0.5)*14,
                p[0]+(q[0]-p[0])*0.7+(Math.random()-0.5)*18,
                p[1]+(q[1]-p[1])*0.9+(Math.random()-0.5)*14,
                q[0], q[1]);
        }
        ctx.closePath();
        const cg = ctx.createLinearGradient(0,0,0,1024);
        cg.addColorStop(0, '#1a4a8a'); cg.addColorStop(1, '#0e2e60');
        ctx.fillStyle = cg;
        ctx.fill();
        ctx.strokeStyle = 'rgba(80,160,255,0.55)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
    }
    // Grid lines
    ctx.strokeStyle = 'rgba(79,160,255,0.12)';
    ctx.lineWidth = 0.6;
    for (let y = 0; y < 1024; y += 1024/18) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(2048,y); ctx.stroke(); }
    for (let x = 0; x < 2048; x += 2048/36) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,1024); ctx.stroke(); }

    const tex = new THREE.CanvasTexture(tc);
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    const geo = new THREE.SphereGeometry(1, 64, 64);
    const mat = new THREE.MeshPhongMaterial({ map: tex, specular: new THREE.Color(0x223366), shininess: 8 });
    earthMesh = new THREE.Mesh(geo, mat);
    // Earth does NOT rotate — it stays fixed in the scene
    scene.add(earthMesh);

    // City lights overlay
    const ltc = document.createElement('canvas');
    ltc.width = 1024; ltc.height = 512;
    const lctx = ltc.getContext('2d');
    lctx.fillStyle = '#000'; lctx.fillRect(0,0,1024,512);
    const toXY = (lon,lat) => [(lon+180)/360*1024, (90-lat)/180*512];
    const clusters = [
        ...Array.from({length:80},()=>[-75+(Math.random()-.5)*15, 40+(Math.random()-.5)*8]),
        ...Array.from({length:100},()=>[10+(Math.random()-.5)*30, 50+(Math.random()-.5)*12]),
        ...Array.from({length:60},()=>[135+(Math.random()-.5)*15, 35+(Math.random()-.5)*10]),
        ...Array.from({length:70},()=>[80+(Math.random()-.5)*20, 22+(Math.random()-.5)*12]),
        ...Array.from({length:80},()=>[115+(Math.random()-.5)*25, 33+(Math.random()-.5)*15]),
        ...Array.from({length:30},()=>[105+(Math.random()-.5)*20, 15+(Math.random()-.5)*15]),
        ...Array.from({length:25},()=>[50+(Math.random()-.5)*20, 28+(Math.random()-.5)*10]),
        ...Array.from({length:30},()=>[-45+(Math.random()-.5)*15,-15+(Math.random()-.5)*15]),
        ...Array.from({length:20},()=>[10+(Math.random()-.5)*20, 7+(Math.random()-.5)*10]),
    ];
    for (const [lon,lat] of clusters) {
        const [px,py] = toXY(lon,lat);
        const g = lctx.createRadialGradient(px,py,0,px,py,1.5);
        g.addColorStop(0,'rgba(255,240,180,0.9)'); g.addColorStop(1,'rgba(255,200,100,0)');
        lctx.fillStyle = g; lctx.fillRect(px-3,py-3,6,6);
    }
    const lightsMat = new THREE.MeshBasicMaterial({
        map: new THREE.CanvasTexture(ltc), blending: THREE.AdditiveBlending,
        transparent: true, opacity: 0.4, depthWrite: false,
    });
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.001, 64, 64), lightsMat));

    // Atmosphere glow
    atmosphereMesh = new THREE.Mesh(
        new THREE.SphereGeometry(1.018, 64, 64),
        new THREE.ShaderMaterial({
            vertexShader:   `varying vec3 vNormal; void main() { vNormal = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
            fragmentShader: `varying vec3 vNormal; void main() { float rim = 1.0 - max(dot(vNormal, vec3(0,0,1)), 0.0); float intensity = pow(rim, 2.5); gl_FragColor = vec4(0.25,0.55,1.0,intensity*0.65); }`,
            side: THREE.BackSide, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
        })
    );
    scene.add(atmosphereMesh);

    // Equator ring
    scene.add(new THREE.Mesh(
        new THREE.RingGeometry(1.002, 1.004, 128),
        new THREE.MeshBasicMaterial({ color: 0x336699, side: THREE.DoubleSide, transparent: true, opacity: 0.3, depthWrite: false })
    ));
}

function buildStars() {
    const N = 8000, pos = new Float32Array(N*3);
    for (let i=0;i<N;i++) {
        const r=150+Math.random()*350, th=Math.random()*Math.PI*2, ph=Math.acos(2*Math.random()-1);
        pos[i*3]=r*Math.sin(ph)*Math.cos(th); pos[i*3+1]=r*Math.sin(ph)*Math.sin(th); pos[i*3+2]=r*Math.cos(ph);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
    scene.add(new THREE.Points(geo, new THREE.PointsMaterial({color:0xffffff,size:0.4,sizeAttenuation:true,transparent:true,opacity:0.7})));
}

function buildLighting() {
    scene.add(new THREE.AmbientLight(0x334466, 0.5));
    const sun = new THREE.DirectionalLight(0xffe8cc, 0.9);
    sun.position.set(4, 1, 3); scene.add(sun);
    const fill = new THREE.DirectionalLight(0x3355aa, 0.25);
    fill.position.set(-4,-2,-4); scene.add(fill);
}

function buildControls() {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);   // Earth center always locked as target
    controls.enableDamping  = true;
    controls.dampingFactor  = 0.06;
    controls.minDistance    = 1.15;
    controls.maxDistance    = 25;
    controls.rotateSpeed    = 0.5;
    controls.zoomSpeed      = 0.9;
    controls.enablePan      = false;   // panning would move target off Earth center
    controls.update();
}

// ─── Animation loop ───────────────────────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    // Earth does NOT rotate — earthMesh.rotation.y is never changed
    if (selectedMarker) {
        const s = 1 + 0.25 * Math.sin(Date.now() * 0.003);
        selectedMarker.scale.setScalar(s);
    }
    renderer.render(scene, camera);
}

// ─── Coordinate helper ────────────────────────────────────────────────────────
function ecefToThree(x, y, z) {
    return new THREE.Vector3(x * SCALE, z * SCALE, y * SCALE);
}

// ─── Data loading ─────────────────────────────────────────────────────────────
async function loadSatellites() {
    const r = await fetch(`${API_BASE}/api/satellites?limit=20000`);
    const d = await r.json();
    satellites = d.satellites;
    updateSatList();
    updateStats();
}

async function refreshPositions() {
    const ts = new Date().toISOString();
    const r  = await fetch(`${API_BASE}/api/propagate?ts=${encodeURIComponent(ts)}&limit=${DEFAULT_SAT_LIMIT}`);
    const d  = await r.json();
    positions = d.positions;
    updateTimeDisplay(d.timestamp);
    buildSatPoints();

    // Rebuild orbit paths for the currently displayed batch
    // (stations always included; active sats up to ORBIT_DISPLAY_LIMIT)
    await refreshOrbitBatch();

    // If a satellite is selected keep it highlighted
    if (selectedNoradId) highlightOrbit(selectedNoradId);
}

/**
 * Fetch and render orbit paths for all stations + current position batch.
 * Uses /api/orbit_batch to do it in two HTTP calls maximum.
 */
async function refreshOrbitBatch() {
    const ts = new Date().toISOString();

    // ── 1. Collect station NORAD IDs (always include all 22)
    const stationIds = satellites
        .filter(s => s.source === 'stations')
        .map(s => s.norad_id);

    // ── 2. Active-satellite IDs from current propagation batch
    const activeIds = positions
        .filter(p => p.source !== 'stations')
        .map(p => p.norad_id)
        .slice(0, ORBIT_DISPLAY_LIMIT);

    const allIds = [...new Set([...stationIds, ...activeIds])];
    if (allIds.length === 0) return;

    try {
        const resp = await fetch(
            `${API_BASE}/api/orbit_batch?norad_ids=${allIds.join(',')}&ts=${encodeURIComponent(ts)}&steps=90`
        );
        const data = await resp.json();

        for (const entry of data.orbits) {
            if (entry.error && (!entry.orbit_path || entry.orbit_path.length === 0)) {
                console.warn(`Orbit error norad=${entry.norad_id} (${entry.name || '?'}): ${entry.error}`);
                continue;
            }
            upsertOrbitLine(entry.norad_id, entry.source, entry.orbit_path, entry.stale);
        }
    } catch (err) {
        console.error('orbit_batch error:', err);
    }
}

/**
 * Create or update a persistent orbit line in the scene.
 * Does NOT remove/recreate if the line already exists — just replaces geometry.
 * Applies dim appearance by default; selection highlighting is separate.
 */
function upsertOrbitLine(noradId, source, orbitPath, stale) {
    if (!orbitPath || orbitPath.length < 2) return;

    const isStation = source === 'stations';
    const pts = orbitPath.map(p => ecefToThree(p.x, p.y, p.z));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);

    if (orbitLines.has(noradId)) {
        // Update geometry in-place; preserve material (may be highlighted)
        const existing = orbitLines.get(noradId);
        existing.geometry.dispose();
        existing.geometry = geo;
    } else {
        // First time — create with dim appearance
        const color   = isStation ? ORBIT_DIM_COLOR_STATION : ORBIT_DIM_COLOR_SAT;
        const opacity = ORBIT_DIM_OPACITY;
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
            color, transparent: true, opacity,
        }));
        line.userData = { noradId, source, isStation };
        scene.add(line);
        orbitLines.set(noradId, line);
    }
}

// ─── Satellite dot markers ────────────────────────────────────────────────────
function buildSatPoints() {
    if (satPoints) { scene.remove(satPoints); satPoints = null; }
    if (!positions.length) return;

    const geo     = new THREE.BufferGeometry();
    const posArr  = new Float32Array(positions.length * 3);
    const colArr  = new Float32Array(positions.length * 3);
    const cStn    = new THREE.Color(0x00e5a0);
    const cSat    = new THREE.Color(0x4f8cff);
    const cStale  = new THREE.Color(0xffa64d);

    for (let i = 0; i < positions.length; i++) {
        const p = positions[i];
        const v = ecefToThree(p.x, p.y, p.z);
        posArr[i*3]=v.x; posArr[i*3+1]=v.y; posArr[i*3+2]=v.z;
        const c = p.source==='stations' ? cStn : p.stale ? cStale : cSat;
        colArr[i*3]=c.r; colArr[i*3+1]=c.g; colArr[i*3+2]=c.b;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colArr, 3));

    satPoints = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.022, vertexColors: true, sizeAttenuation: true,
        transparent: true, opacity: 0.95,
        blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(satPoints);
}

// ─── Orbit highlighting ───────────────────────────────────────────────────────
/**
 * Highlight a single orbit line and dim everything else.
 * Only changes material properties — no geometry removal/recreation.
 */
function highlightOrbit(noradId) {
    for (const [nid, line] of orbitLines) {
        const isStation = line.userData.isStation;
        if (nid === noradId) {
            line.material.color.setHex(isStation ? ORBIT_SEL_COLOR_STATION : ORBIT_SEL_COLOR_SAT);
            line.material.opacity = ORBIT_SEL_OPACITY;
            line.renderOrder = 1;
        } else {
            line.material.color.setHex(isStation ? ORBIT_DIM_COLOR_STATION : ORBIT_DIM_COLOR_SAT);
            line.material.opacity = ORBIT_DIM_OPACITY;
            line.renderOrder = 0;
        }
    }
}

/** Restore all orbit lines to their normal dim appearance. */
function resetOrbitHighlights() {
    for (const [, line] of orbitLines) {
        const isStation = line.userData.isStation;
        line.material.color.setHex(isStation ? ORBIT_DIM_COLOR_STATION : ORBIT_DIM_COLOR_SAT);
        line.material.opacity = ORBIT_DIM_OPACITY;
        line.renderOrder = 0;
    }
}

// ─── Selected satellite marker ────────────────────────────────────────────────
async function selectSat(noradId) {
    // If re-clicking the same satellite, deselect
    if (noradId === selectedNoradId) { deselectSat(); return; }

    selectedNoradId = noradId;
    updateSatList();

    // Highlight orbit (in-place if it exists, else will appear after batch fetch)
    highlightOrbit(noradId);

    // Remove old pulsing marker
    if (selectedMarker) { scene.remove(selectedMarker); selectedMarker = null; }

    // Fetch detailed data for detail panel + pulsing marker position
    try {
        const ts   = new Date().toISOString();
        const resp = await fetch(`${API_BASE}/api/propagate_one?norad_id=${noradId}&ts=${encodeURIComponent(ts)}`);
        const data = await resp.json();
        if (data.error) { console.warn('propagate_one error:', data.error); }

        // If this satellite didn't have an orbit line yet, add it now
        if (data.orbit_path?.length > 1 && !orbitLines.has(noradId)) {
            upsertOrbitLine(noradId, data.source, data.orbit_path, data.stale);
            highlightOrbit(noradId);  // apply highlight to newly created line
        }

        // Pulsing marker at current position
        if (data.position) {
            const v = ecefToThree(data.position.x, data.position.y, data.position.z);
            const isStation = data.source === 'stations';
            selectedMarker = new THREE.Mesh(
                new THREE.SphereGeometry(0.018, 12, 12),
                new THREE.MeshBasicMaterial({ color: isStation ? 0x00e5a0 : 0x4f8cff, transparent: true, opacity: 0.95 })
            );
            selectedMarker.position.copy(v);
            scene.add(selectedMarker);
        }

        showDetail(data);
    } catch (err) {
        console.error('selectSat error:', err);
    }
}
window.selectSat = selectSat;

function deselectSat() {
    selectedNoradId = null;
    if (selectedMarker) { scene.remove(selectedMarker); selectedMarker = null; }
    resetOrbitHighlights();
    document.querySelector('.detail-panel')?.classList.add('hidden');
    updateSatList();
}
window.deselectSat = deselectSat;

// ─── Auto-refresh ─────────────────────────────────────────────────────────────
function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(refreshPositions, REFRESH_INTERVAL_MS);
}

// ─── UI updates ───────────────────────────────────────────────────────────────
function updateTimeDisplay(iso) {
    const el = document.getElementById('sim-time');
    if (el && iso) el.textContent = iso.replace('T',' ').slice(0,19) + ' UTC';
}

function updateStats() {
    const el = document.getElementById('stats-text');
    if (!el) return;
    const stations = satellites.filter(s => s.source==='stations').length;
    el.textContent = `${satellites.length.toLocaleString()} satellites · ${stations} stations`;
}

function updateSatList() {
    const container = document.getElementById('sat-list');
    if (!container) return;

    let list = satellites;
    if (sourceFilter !== 'all') list = list.filter(s => s.source===sourceFilter);
    if (searchQuery) {
        const q = searchQuery.toUpperCase();
        list = list.filter(s => s.name.toUpperCase().includes(q) || String(s.norad_id).includes(q));
    }

    const show = list.slice(0, 200);
    container.innerHTML = show.map(s => {
        const isStation = s.source === 'stations';
        const sel = s.norad_id === selectedNoradId;
        const pos = positions.find(p => p.norad_id === s.norad_id);
        const alt = pos ? Math.round(pos.alt_km) + ' km' : '—';
        return `<div class="sat-item${isStation?' station':''}${sel?' selected':''}"
                     data-norad="${s.norad_id}" onclick="selectSat(${s.norad_id})">
            <div class="sat-dot"></div>
            <div class="sat-info">
                <div class="sat-name">${esc(s.name)}</div>
                <div class="sat-meta">${s.norad_id} · ${isStation?'Station':'Satellite'}</div>
            </div>
            <div class="sat-alt">${alt}</div>
            ${s.stale?'<span class="stale-badge">STALE</span>':''}
        </div>`;
    }).join('');

    if (list.length > 200) {
        container.innerHTML += `<div style="text-align:center;padding:12px;color:var(--text-muted);font-size:12px;">
            Showing 200 of ${list.length.toLocaleString()} · Refine search to see more</div>`;
    }
}

function showDetail(d) {
    const panel = document.querySelector('.detail-panel');
    if (!panel) return;
    panel.classList.remove('hidden');
    const isStation = d.source === 'stations';
    const pos = d.position;
    const alt = pos ? pos.alt_km.toFixed(1)+' km' : 'Error';
    const posStr = pos ? `${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}` : '—';
    const ageStr = d.epoch_age_days != null ? d.epoch_age_days.toFixed(1)+' days' : '—';
    const accentColor = isStation ? 'var(--station)' : 'var(--accent)';
    panel.innerHTML = `
        <div class="detail-header">
            <div>
                <div class="detail-title" style="color:${accentColor}">${esc(d.name)}</div>
                <div class="detail-subtitle">${isStation ? '🏠 Space Station' : '🛰️ Satellite'}</div>
            </div>
            <button class="detail-close" onclick="deselectSat()">✕</button>
        </div>
        <div class="detail-grid">
            <div class="detail-card"><div class="detail-label">NORAD ID</div><div class="detail-value">${d.norad_id}</div></div>
            <div class="detail-card"><div class="detail-label">INTL DESIG</div><div class="detail-value">${esc(d.intl_desig||'—')}</div></div>
            <div class="detail-card"><div class="detail-label">ALTITUDE</div><div class="detail-value" style="color:${accentColor}">${alt}</div></div>
            <div class="detail-card"><div class="detail-label">PERIOD</div><div class="detail-value">${d.period_min ? d.period_min.toFixed(1)+' min' : '—'}</div></div>
            <div class="detail-card"><div class="detail-label">INCLINATION</div><div class="detail-value">${d.inclination.toFixed(2)}°</div></div>
            <div class="detail-card"><div class="detail-label">ECCENTRICITY</div><div class="detail-value">${d.eccentricity.toFixed(6)}</div></div>
            <div class="detail-card"><div class="detail-label">MEAN MOTION</div><div class="detail-value">${d.mean_motion.toFixed(4)} rev/d</div></div>
            <div class="detail-card"><div class="detail-label">EPOCH AGE</div>
                <div class="detail-value${d.stale?' warning':''}">${ageStr}${d.stale?' ⚠':''}</div>
            </div>
            <div class="detail-card full"><div class="detail-label">ECEF POSITION (km)</div>
                <div class="detail-value" style="font-size:12px">${posStr}</div></div>
            <div class="detail-card full"><div class="detail-label">EPOCH</div>
                <div class="detail-value" style="font-size:11px">${esc(d.epoch)}</div></div>
        </div>
        <div class="detail-disclaimer">
            ⚠ Positions are SGP4-propagated estimates from TLE data.
            TEME to ECEF uses approximate GMST (IAU 1982, &lt;1 km error).
            Not real-time tracking. Accuracy degrades with epoch age.
            ${d.position_error ? '<br>⛔ '+esc(d.position_error) : ''}
        </div>`;
}

function hideLoading() {
    const el = document.querySelector('.loading-overlay');
    if (el) { el.classList.add('fade-out'); setTimeout(()=>el.remove(), 600); }
}

function esc(s) {
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
        .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─── UI event listeners ───────────────────────────────────────────────────────
function setupUIListeners() {
    const input = document.getElementById('search-input');
    if (input) {
        let t;
        input.addEventListener('input', e => {
            clearTimeout(t);
            t = setTimeout(() => { searchQuery = e.target.value; updateSatList(); }, 200);
        });
    }
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.filter-tab').forEach(x => x.classList.remove('active'));
            tab.classList.add('active');
            sourceFilter = tab.dataset.filter;
            updateSatList();
        });
    });
}

window.addEventListener('DOMContentLoaded', init);
