/**
 * Akash Setu — 3D Satellite Tracking, Conjunction Screening & Avoidance Dashboard
 * ==============================================================================
 * Integrated 3D Earth, Orbit Trajectories, Conjunction Encounters,
 *          Nested Grid Refinements (0.01m), Avoidance Maneuver Planning,
 *          and ML Collision Risk Forecasting.
 *
 * Coordinate system:
 *   ECEF X -> Three.js +X  (0 lon, equator)
 *   ECEF Y -> Three.js +Z  (90E, equator)
 *   ECEF Z -> Three.js +Y  (north pole = up)
 *   Scale: 1 unit = 6371 km
 */

// ─── Constants ────────────────────────────────────────────────────────────────
const API_BASE             = (window.location.protocol === 'file:' || 
    (window.location.hostname === 'localhost' && window.location.port !== '5000') || 
    (window.location.hostname === '127.0.0.1' && window.location.port !== '5000'))
    ? 'http://127.0.0.1:5000'
    : '';
const EARTH_RADIUS_KM      = 6371;
const SCALE                = 1 / EARTH_RADIUS_KM;
const REFRESH_INTERVAL_MS  = 30000;
const DEFAULT_SAT_LIMIT    = 300;   // batch propagation size
const ORBIT_DISPLAY_LIMIT  = 300;   // how many active-sat orbits to pre-render

const ORBIT_DIM_COLOR_SAT     = 0x2a5caa;  // normal active-sat orbit
const ORBIT_DIM_COLOR_STATION = 0x00995a;  // normal station orbit
const ORBIT_DIM_OPACITY       = 0.30;
const ORBIT_SEL_COLOR_SAT     = 0x80c0ff;  // selected active-sat orbit
const ORBIT_SEL_COLOR_STATION = 0x00ffaa;  // selected station orbit
const ORBIT_SEL_OPACITY       = 0.90;

// Conjunction & Avoidance Colors
const CONJUNCTION_COLOR_1     = 0xff3355;  // primary satellite orbit
const CONJUNCTION_COLOR_2     = 0xffaa22;  // secondary satellite orbit
const MANEUVER_ORBIT_COLOR    = 0x00ffff;  // post-burn maneuver trajectory (cyan)

// ─── State ────────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let earthMesh, atmosphereMesh;
let satPoints          = null;        // Points object for all satellite positions
let orbitLines         = new Map();   // norad_id -> THREE.Line (persistent)
let selectedMarker     = null;        // pulsing sphere for the selected satellite
let conjunctionLines   = [];          // highlighted conjunction lines
let conjunctionMarker  = null;        // 3D group for conjunction encounter marker
let maneuverOrbitLine  = null;        // 3D line for post-burn maneuver trajectory

let satellites         = [];          // metadata from /api/satellites
let positions          = [];          // current ECEF positions from /api/propagate
let selectedNoradId    = null;
let currentMode        = 'tracker';   // 'tracker', 'screening', 'avoidance', 'ml_risk'
let activeAlertData    = null;        // selected conjunction alert
let searchQuery        = '';
let sourceFilter       = 'all';
let refreshTimer       = null;
let raycaster          = new THREE.Raycaster();
let mouse              = new THREE.Vector2();

// ─── Initialization ───────────────────────────────────────────────────────────
async function init() {
    setupScene();
    buildEarth();
    buildStars();
    buildLighting();
    buildControls();
    animate();
    setupUIListeners();

    try {
        await loadSatellites();
        await refreshPositions();
        await refreshOrbitBatch();
        await loadMLModelInfo();
        hideLoading();
        startAutoRefresh();
    } catch (err) {
        console.error('Initialization error:', err);
        const el = document.querySelector('.loading-text');
        if (el) {
            el.innerHTML = `⚠️ Failed to connect to API: ${err.message}<br><small style="color:var(--text-muted);display:block;margin-top:6px;">Please ensure the Flask backend is running on <code>http://127.0.0.1:5000</code>.<br>Start command: <code>python backend/app.py</code></small>`;
        }
        const statsEl = document.getElementById('stats-text');
        if (statsEl) {
            statsEl.textContent = 'Backend Offline';
            statsEl.style.color = 'var(--danger)';
        }
    }
}

function setupScene() {
    const canvas = document.getElementById('globe-canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060913);

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
    camera.position.set(0, 1.2, 3.2);

    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function buildEarth() {
    const geo = new THREE.SphereGeometry(1, 64, 64);

    // ------------------------------------------------------------------
    // Procedural "Blue Marble" fallback canvas (2048 x 1024)
    // Built immediately so the globe shows instantly while any
    // external texture loads in the background.
    // ------------------------------------------------------------------
    function makeProceduralTexture() {
        const W = 2048, H = 1024;
        const c = document.createElement('canvas');
        c.width = W; c.height = H;
        const ctx = c.getContext('2d');

        // --- ocean base ---
        const og = ctx.createLinearGradient(0, 0, 0, H);
        og.addColorStop(0.00, '#04101e');
        og.addColorStop(0.10, '#062040');
        og.addColorStop(0.28, '#0b2d58');
        og.addColorStop(0.50, '#0d3a72');
        og.addColorStop(0.72, '#0b2d58');
        og.addColorStop(0.90, '#062040');
        og.addColorStop(1.00, '#04101e');
        ctx.fillStyle = og;
        ctx.fillRect(0, 0, W, H);

        // ocean shimmer rows
        for (let r = 0; r < H; r++) {
            const v = Math.sin(r/H*Math.PI*18)*0.014 + Math.cos(r/H*Math.PI*9)*0.009;
            if (Math.abs(v) > 0.003) {
                ctx.fillStyle = v > 0 ? `rgba(30,90,180,${v*0.22})` : `rgba(2,10,30,${-v*0.18})`;
                ctx.fillRect(0, r, W, 1);
            }
        }

        function px(lon, lat) { return { x:(lon+180)/360*W, y:(90-lat)/180*H }; }
        function ell(cLon, cLat, rLon, rLat, color, n) {
            n = n||48;
            ctx.beginPath();
            const first = px(cLon+rLon, cLat);
            ctx.moveTo(first.x, first.y);
            for (let i=1; i<=n; i++) {
                const a = i/n*Math.PI*2;
                const p = px(cLon+rLon*Math.cos(a), cLat+rLat*Math.sin(a));
                ctx.lineTo(p.x, p.y);
            }
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        }

        // continents
        ell(22,4,27,37,'#2e5518'); ell(22,4,25,35,'#3d6222');
        ell(18,20,15,9,'#7e6232'); ell(22,26,16,8,'#8c7240');
        ell(15,53,25,17,'#2c4a1a'); ell(15,54,23,15,'#3e5a28');
        ell(80,47,67,29,'#263c18'); ell(80,47,65,27,'#3a5620');
        ell(56,25,13,9,'#b8a268'); ell(90,42,22,11,'#8c8462');
        ell(106,23,11,8,'#4c6e2a'); ell(138,36,7,7,'#3a5620');
        ell(78,30,11,7,'#7e8c72');
        ell(-98,47,41,27,'#273c18'); ell(-98,47,39,25,'#3c5620');
        ell(-110,40,15,11,'#7c6a42'); ell(-108,34,11,9,'#a47c4a');
        ell(-58,-12,22,35,'#2e6020'); ell(-58,-12,20,33,'#3c6628');
        ell(-66,-22,8,7,'#6c7244'); ell(-66,-38,7,9,'#5c6244');
        ell(134,-26,22,15,'#6c4c22'); ell(134,-26,20,13,'#8c6232');
        ell(148,-30,6,6,'#4c6228');
        ell(-42,72,19,13,'#ccdcec'); ell(-42,72,17,11,'#dde8f4');
        ctx.fillStyle='#c4d8e8';
        ctx.fillRect(0,H*0.84,W,H*0.16);

        // polar caps
        let g = ctx.createRadialGradient(W/2,0,0,W/2,0,H*0.14);
        g.addColorStop(0,'rgba(228,242,255,0.97)'); g.addColorStop(1,'rgba(200,225,248,0)');
        ctx.fillStyle=g; ctx.fillRect(0,0,W,H*0.18);
        g = ctx.createRadialGradient(W/2,H,0,W/2,H,H*0.15);
        g.addColorStop(0,'rgba(228,242,255,0.99)'); g.addColorStop(1,'rgba(200,225,248,0)');
        ctx.fillStyle=g; ctx.fillRect(0,H*0.83,W,H*0.17);

        // clouds
        ctx.globalAlpha = 0.42;
        let s = 137;
        function rng(){ s=(s*1664525+1013904223)&0xffffffff; return (s>>>0)/0xffffffff; }
        for (let i=0;i<120;i++){
            const cy=rng()*H, cx=rng()*W, cw=50+rng()*200, ch=10+rng()*28;
            const lat=90-(cy/H)*180;
            if(rng()>0.45+0.45*Math.abs(Math.sin(lat*Math.PI/90))) continue;
            const cg=ctx.createRadialGradient(cx,cy,0,cx,cy,cw*0.6);
            cg.addColorStop(0,'rgba(255,255,255,0.78)');
            cg.addColorStop(0.5,'rgba(240,248,255,0.32)');
            cg.addColorStop(1,'rgba(240,248,255,0)');
            ctx.save(); ctx.scale(1,ch/(cw*0.6));
            ctx.beginPath(); ctx.arc(cx,cy*(cw*0.6)/ch,cw*0.6,0,Math.PI*2);
            ctx.fillStyle=cg; ctx.fill(); ctx.restore();
        }
        ctx.globalAlpha=1;
        return c;
    }

    // ------------------------------------------------------------------
    // Specular map (ocean shiny, land matte, ice semi-shiny)
    // ------------------------------------------------------------------
    function makeSpecularMap() {
        const W=2048,H=1024;
        const c=document.createElement('canvas'); c.width=W; c.height=H;
        const ctx=c.getContext('2d');
        ctx.fillStyle='#5a7080'; ctx.fillRect(0,0,W,H); // ocean shiny
        function px(lon,lat){return{x:(lon+180)/360*W,y:(90-lat)/180*H};}
        function ell(cLon,cLat,rLon,rLat,n){
            n=n||48;
            ctx.beginPath();
            const first=px(cLon+rLon,cLat); ctx.moveTo(first.x,first.y);
            for(let i=1;i<=n;i++){const a=i/n*Math.PI*2;const p=px(cLon+rLon*Math.cos(a),cLat+rLat*Math.sin(a));ctx.lineTo(p.x,p.y);}
            ctx.closePath(); ctx.fillStyle='#080808'; ctx.fill();
        }
        ell(22,4,25,35); ell(15,54,23,15); ell(80,47,65,27);
        ell(56,25,13,9); ell(-98,47,39,25); ell(-58,-12,20,33);
        ell(134,-26,20,13); ell(-42,72,17,11);
        ctx.fillStyle='#080808'; ctx.fillRect(0,H*0.84,W,H*0.16);
        let g=ctx.createRadialGradient(W/2,0,0,W/2,0,H*0.14);
        g.addColorStop(0,'rgba(170,190,210,1)'); g.addColorStop(1,'rgba(170,190,210,0)');
        ctx.fillStyle=g; ctx.fillRect(0,0,W,H*0.18);
        g=ctx.createRadialGradient(W/2,H,0,W/2,H,H*0.15);
        g.addColorStop(0,'rgba(170,190,210,1)'); g.addColorStop(1,'rgba(170,190,210,0)');
        ctx.fillStyle=g; ctx.fillRect(0,H*0.83,W,H*0.17);
        return c;
    }

    // ------------------------------------------------------------------
    // Build initial material with procedural texture
    // ------------------------------------------------------------------
    const aniso = renderer.capabilities.getMaxAnisotropy();
    const procTex  = new THREE.CanvasTexture(makeProceduralTexture());
    procTex.anisotropy = aniso;
    const specTex   = new THREE.CanvasTexture(makeSpecularMap());

    const mat = new THREE.MeshPhongMaterial({
        map:          procTex,
        specularMap:  specTex,
        specular:     new THREE.Color(0x6699cc),
        shininess:    80,
        emissive:     new THREE.Color(0x020810),
        emissiveIntensity: 1.0,
    });

    earthMesh = new THREE.Mesh(geo, mat);
    scene.add(earthMesh);

    // ------------------------------------------------------------------
    // Attempt to load a high-quality Blue Marble texture.
    // Falls back silently to the procedural canvas if unreachable.
    // Sources are tried in order; first success wins.
    // ------------------------------------------------------------------
    const TEXTURE_URLS = [
        'https://cdn.jsdelivr.net/npm/three-globe@2.31.0/example/img/earth-blue-marble.jpg',
        'https://unpkg.com/three-globe@2.31.0/example/img/earth-blue-marble.jpg',
        'https://raw.githubusercontent.com/turban/webgl-earth/master/images/2_no_clouds_4k.jpg',
    ];

    (function tryLoad(urls) {
        if (!urls.length) return; // all failed, keep procedural
        const loader = new THREE.TextureLoader();
        loader.crossOrigin = 'anonymous';
        loader.load(
            urls[0],
            function(tex) {            // success
                tex.anisotropy = aniso;
                mat.map = tex;
                mat.needsUpdate = true;
                procTex.dispose();     // free memory
            },
            undefined,
            function() {               // error → try next
                tryLoad(urls.slice(1));
            }
        );
    })(TEXTURE_URLS);

    // Atmosphere glow removed per user request.
    // atmosphereMesh kept as null so other code referencing it stays safe.
    atmosphereMesh = null;
}

function buildStars() {
    const count = 1200;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i += 3) {
        const r = 40 + Math.random() * 40;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        pos[i]   = r * Math.sin(phi) * Math.cos(theta);
        pos[i+1] = r * Math.cos(phi);
        pos[i+2] = r * Math.sin(phi) * Math.sin(theta);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x88aacc, size: 0.15, transparent: true, opacity: 0.7 });
    scene.add(new THREE.Points(geo, mat));
}

function buildLighting() {
    // Soft dark-blue ambient so night side is very dark (matches reference)
    scene.add(new THREE.AmbientLight(0x12233a, 0.55));
    // Sun: warm-white, high intensity for strong day-side illumination
    const sun = new THREE.DirectionalLight(0xfff5e8, 3.0);
    sun.position.set(5, 2, 4);
    scene.add(sun);
    // Subtle earthshine: very dim cold blue on the night hemisphere
    const fill = new THREE.DirectionalLight(0x1a3a8a, 0.18);
    fill.position.set(-5, -1, -3);
    scene.add(fill);
}

function buildControls() {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.2;
    controls.maxDistance = 12;
    controls.target.set(0, 0, 0);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    if (selectedMarker) {
        const s = 1 + 0.15 * Math.sin(Date.now() * 0.005);
        selectedMarker.scale.set(s, s, s);
    }
    renderer.render(scene, camera);
}

function ecefToThree(x, y, z) {
    return new THREE.Vector3(x * SCALE, z * SCALE, y * SCALE);
}

// ─── Data Loading ─────────────────────────────────────────────────────────────

// ─── UI Helper Functions ──────────────────────────────────────────────────────
function updateTimeDisplay(iso) {
    const el = document.getElementById('sim-time');
    if (el && iso) el.textContent = iso.replace('T', ' ').slice(0, 19) + ' UTC';
}

function updateStats() {
    const el = document.getElementById('stats-text');
    if (!el) return;
    const stations = satellites.filter(s => s.source === 'stations').length;
    el.textContent = `${satellites.length.toLocaleString()} satellites · ${stations} stations`;
}

function updateSatList() {
    const container = document.getElementById('sat-list');
    if (!container) return;

    let list = satellites;
    if (sourceFilter !== 'all') list = list.filter(s => s.source === sourceFilter);
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
        return `<div class="sat-item${isStation ? ' station' : ''}${sel ? ' selected' : ''}"
                     data-norad="${s.norad_id}" onclick="selectSat(${s.norad_id})">
            <div class="sat-dot"></div>
            <div class="sat-info">
                <div class="sat-name">${esc(s.name)}</div>
                <div class="sat-meta">${s.norad_id} · ${isStation ? 'Station' : 'Satellite'}</div>
            </div>
            <div class="sat-alt">${alt}</div>
            ${s.stale ? '<span class="stale-badge">STALE</span>' : ''}
        </div>`;
    }).join('');

    if (list.length > 200) {
        container.innerHTML += `<div style="text-align:center;padding:12px;color:var(--text-muted);font-size:12px;">
            Showing 200 of ${list.length.toLocaleString()} · Refine search to see more</div>`;
    }
}

function hideLoading() {
    const el = document.getElementById('loading-overlay');
    if (el) {
        el.classList.add('fade-out');
        setTimeout(() => el.remove(), 600);
    }
}

function esc(s) {
    return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function clearManeuverLine() {
    clearManeuverTrajectory();
}

async function loadSatellites() {
    const res = await fetch(`${API_BASE}/api/satellites`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    satellites = data.satellites;
    updateStats();
    updateSatList();
}

async function refreshPositions() {
    const res = await fetch(`${API_BASE}/api/propagate?limit=${DEFAULT_SAT_LIMIT}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    positions = data.positions;
    updateTimeDisplay(data.timestamp);
    buildSatPoints();
}

async function refreshOrbitBatch() {
    if (!satellites || satellites.length === 0) return;
    const ids = satellites.slice(0, ORBIT_DISPLAY_LIMIT).map(s => s.norad_id).join(',');
    const res = await fetch(`${API_BASE}/api/orbit_batch?norad_ids=${ids}&steps=60`);
    if (!res.ok) return;
    const data = await res.json();
    for (const item of (data.orbits || [])) {
        upsertOrbitLine(item.norad_id, item.source, item.orbit_path, item.stale);
    }
}

function upsertOrbitLine(noradId, source, orbitPath, stale) {
    if (!orbitPath || orbitPath.length < 2) return;
    const isStation = source === 'stations';
    // API may return {x,y,z} objects (orbit_batch/propagate_one) or [x,y,z] arrays (trajectory).
    // Handle both defensively to prevent NaN geometry corruption.
    const points = orbitPath.map(p => {
        const x = (p.x !== undefined) ? p.x : p[0];
        const y = (p.y !== undefined) ? p.y : p[1];
        const z = (p.z !== undefined) ? p.z : p[2];
        return ecefToThree(x, y, z);
    }).filter(v => isFinite(v.x) && isFinite(v.y) && isFinite(v.z));
    if (points.length < 2) return;
    points.push(points[0].clone());
    const geo = new THREE.BufferGeometry().setFromPoints(points);

    if (orbitLines.has(noradId)) {
        const existing = orbitLines.get(noradId);
        existing.geometry.dispose();
        existing.geometry = geo;
        return;
    }

    const col = isStation ? ORBIT_DIM_COLOR_STATION : ORBIT_DIM_COLOR_SAT;
    const mat = new THREE.LineBasicMaterial({
        color: col,
        transparent: true,
        opacity: ORBIT_DIM_OPACITY,
        linewidth: 1
    });
    const line = new THREE.Line(geo, mat);
    line.userData = { noradId, source, stale, originalColor: col, originalOpacity: ORBIT_DIM_OPACITY };
    line.frustumCulled = false; // prevent culling issues from bounding sphere anomalies
    scene.add(line);
    orbitLines.set(noradId, line);
}

function buildSatPoints() {
    const valid = positions.filter(p => !p.error);
    const count = valid.length;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
        const p = valid[i];
        const v = ecefToThree(p.x, p.y, p.z);
        pos[i*3]   = v.x;
        pos[i*3+1] = v.y;
        pos[i*3+2] = v.z;

        if (p.norad_id === selectedNoradId) {
            col[i*3] = 1.0; col[i*3+1] = 0.9; col[i*3+2] = 0.2;
        } else if (p.source === 'stations') {
            col[i*3] = 0.0; col[i*3+1] = 0.9; col[i*3+2] = 0.6;
        } else {
            col[i*3] = 0.35; col[i*3+1] = 0.65; col[i*3+2] = 1.0;
        }
    }

    if (satPoints) {
        satPoints.geometry.dispose();
        scene.remove(satPoints);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    // Manually set bounding sphere to avoid NaN from outlier high-orbit satellites
    geo.computeBoundingBox();
    if (geo.boundingBox && isFinite(geo.boundingBox.min.x)) {
        geo.computeBoundingSphere();
    } else {
        // Fallback: bound sphere large enough for GEO orbit (≈7 units at this scale)
        geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 15);
    }
    const mat = new THREE.PointsMaterial({ size: 4.5, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.9 });
    satPoints = new THREE.Points(geo, mat);
    satPoints.frustumCulled = false; // avoid culling issues from large bounding sphere
    scene.add(satPoints);
}

function highlightOrbit(noradId) {
    resetOrbitHighlights();
    const line = orbitLines.get(noradId);
    if (!line) return;
    const isStation = line.userData.source === 'stations';
    line.material.color.setHex(isStation ? ORBIT_SEL_COLOR_STATION : ORBIT_SEL_COLOR_SAT);
    line.material.opacity = ORBIT_SEL_OPACITY;
    line.material.linewidth = 2;
}

function resetOrbitHighlights() {
    orbitLines.forEach(line => {
        line.material.color.setHex(line.userData.originalColor);
        line.material.opacity = line.userData.originalOpacity;
        line.material.linewidth = 1;
    });
}

async function selectSat(noradId) {
    selectedNoradId = noradId;
    highlightOrbit(noradId);
    buildSatPoints();

    const pos = positions.find(p => p.norad_id === noradId);
    if (pos && !pos.error) {
        const v = ecefToThree(pos.x, pos.y, pos.z);
        if (!selectedMarker) {
            const geo = new THREE.SphereGeometry(0.022, 16, 16);
            const mat = new THREE.MeshBasicMaterial({ color: 0xffd700 });
            selectedMarker = new THREE.Mesh(geo, mat);
            scene.add(selectedMarker);
        }
        selectedMarker.position.copy(v);
        selectedMarker.visible = true;
    }

    try {
        const res = await fetch(`${API_BASE}/api/propagate_one?norad_id=${noradId}`);
        if (res.ok) {
            const data = await res.json();
            upsertOrbitLine(noradId, data.source, data.orbit_path, data.stale);
            highlightOrbit(noradId);
            showSatelliteDetail(data);
        }
    } catch (e) {
        console.error(e);
    }
}

function deselectSat() {
    selectedNoradId = null;
    activeAlertData = null;
    resetOrbitHighlights();
    clearConjunctionOverlay();
    clearManeuverLine();
    buildSatPoints();
    if (selectedMarker) selectedMarker.visible = false;
    const panel = document.getElementById('detail-panel');
    if (panel) panel.classList.add('hidden');
}

function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(async () => {
        try {
            await refreshPositions();
        } catch (e) {
            console.warn('Auto refresh error:', e);
        }
    }, REFRESH_INTERVAL_MS);
}

// ── Conjunction Screening Integration ──
async function runScreening() {
    const btn = document.getElementById('btn-run-screen');
    const statusEl = document.getElementById('screening-status');
    const listEl = document.getElementById('conjunctions-list');

    const thresholdKm = parseFloat(document.getElementById('screen-threshold').value) || 50.0;
    const horizonHrs = parseFloat(document.getElementById('screen-horizon').value) || 3.0;
    const limit = parseInt(document.getElementById('screen-sample').value) || 60;

    btn.disabled = true;
    btn.innerHTML = '⏳ Screening Catalog…';
    statusEl.innerHTML = `Propagating candidate orbits over ${horizonHrs}h horizon (threshold: ${thresholdKm} km)…`;

    try {
        const horizonMin = horizonHrs * 60;
        const res = await fetch(`${API_BASE}/api/screen?threshold_km=${thresholdKm}&horizon_minutes=${horizonMin}&limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        const alerts = data.alerts || [];
        statusEl.innerHTML = `✅ Run Complete: Screened <strong>${data.total_pairs_screened || (limit*(limit-1)/2)}</strong> pairs. Found <strong>${alerts.length}</strong> close-approach alerts.`;

        if (alerts.length === 0) {
            listEl.innerHTML = '<div class="empty-state">No close approaches detected within threshold. Try increasing threshold km.</div>';
            return;
        }

        listEl.innerHTML = '';
        alerts.forEach((alert, idx) => {
            const isCritical = alert.miss_distance_km < 10.0;
            const badgeClass = isCritical ? 'alert-badge-critical' : 'alert-badge-warning';
            const badgeText = isCritical ? 'CRITICAL' : 'WARNING';

            const card = document.createElement('div');
            card.className = 'alert-card';
            card.dataset.index = idx;
            card.innerHTML = `
                <div class="alert-header">
                    <div class="alert-names">⚠️ ${esc(alert.name_1)} ↔ ${esc(alert.name_2)}</div>
                    <span class="alert-badge ${badgeClass}">${badgeText}</span>
                </div>
                <div class="alert-meta">
                    <span>Miss Dist: <span class="alert-miss-distance">${alert.miss_distance_km.toFixed(2)} km</span></span>
                    <span>Rel Speed: ${(alert.relative_speed_km_s || 0).toFixed(1)} km/s</span>
                </div>
                <div class="alert-meta" style="font-size:10px;">
                    <span>TCA: ${alert.tca_utc ? alert.tca_utc.replace('T', ' ').substring(0, 19) : 'T+'+alert.tca_minutes_from_start.toFixed(1)+'m'}</span>
                    <span style="color:var(--accent);font-weight:600;">View in 3D ➔</span>
                </div>
            `;
            card.addEventListener('click', () => {
                document.querySelectorAll('.alert-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                selectConjunction(alert);
            });
            listEl.appendChild(card);
        });

    } catch (err) {
        statusEl.innerHTML = `❌ Screening error: ${err.message}`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Run Collision Screening';
    }
}

async function selectConjunction(alert) {
    activeAlertData = alert;
    resetOrbitHighlights();
    clearConjunctionOverlay();
    clearManeuverLine();

    const n1 = alert.norad_id_1;
    const n2 = alert.norad_id_2;

    await ensureOrbitLoaded(n1);
    await ensureOrbitLoaded(n2);

    const line1 = orbitLines.get(n1);
    const line2 = orbitLines.get(n2);

    if (line1) {
        line1.material.color.setHex(CONJUNCTION_COLOR_1);
        line1.material.opacity = 1.0;
        line1.material.linewidth = 2;
    }
    if (line2) {
        line2.material.color.setHex(CONJUNCTION_COLOR_2);
        line2.material.opacity = 1.0;
        line2.material.linewidth = 2;
    }

    const pos1 = positions.find(p => p.norad_id === n1);
    const pos2 = positions.find(p => p.norad_id === n2);

    conjunctionMarker = new THREE.Group();
    if (pos1 && pos2 && !pos1.error && !pos2.error) {
        const v1 = ecefToThree(pos1.x, pos1.y, pos1.z);
        const v2 = ecefToThree(pos2.x, pos2.y, pos2.z);

        const mid = new THREE.Vector3().addVectors(v1, v2).multiplyScalar(0.5);
        const sphereGeo = new THREE.SphereGeometry(0.03, 16, 16);
        const sphereMat = new THREE.MeshBasicMaterial({ color: 0xff3355, transparent: true, opacity: 0.85 });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.copy(mid);
        conjunctionMarker.add(sphere);

        const lineGeo = new THREE.BufferGeometry().setFromPoints([v1, v2]);
        const lineMat = new THREE.LineDashedMaterial({ color: 0xff8598, dashSize: 0.02, gapSize: 0.01 });
        const connLine = new THREE.Line(lineGeo, lineMat);
        connLine.computeLineDistances();
        conjunctionMarker.add(connLine);

        scene.add(conjunctionMarker);
        controls.target.copy(mid);
    }

    showConjunctionDetail(alert);
}

function clearConjunctionOverlay() {
    if (conjunctionMarker) {
        scene.remove(conjunctionMarker);
        conjunctionMarker = null;
    }
}

async function ensureOrbitLoaded(noradId) {
    if (orbitLines.has(noradId)) return;
    try {
        const res = await fetch(`${API_BASE}/api/propagate_one?norad_id=${noradId}`);
        if (res.ok) {
            const data = await res.json();
            upsertOrbitLine(noradId, data.source, data.orbit_path, data.stale);
        }
    } catch (e) {
        console.warn(e);
    }
}

// ── Grid Refinement Handler ──
async function runGridRefinement(noradA, noradB) {
    const btn = document.getElementById('btn-grid-refine');
    const resultEl = document.getElementById('grid-refine-result');
    if (btn) { btn.disabled = true; btn.innerHTML = '⏳ Refining TCA (0.01m)…'; }

    try {
        const res = await fetch(`${API_BASE}/api/analyse/pair?norad_a=${noradA}&norad_b=${noradB}&threshold_km=50.0&target_resolution_m=0.01`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const analysis = (data.grid_analyses || [])[0];

        if (!analysis) {
            resultEl.innerHTML = '<div class="telemetry-item" style="color:var(--warning)">No grid alert produced for this pair.</div>';
            return;
        }

        resultEl.innerHTML = `
            <div class="telemetry-item">
                <div class="telemetry-label">Refined Miss Distance</div>
                <div class="telemetry-val" style="color:#00ffaa">${analysis.refined_miss_distance_m.toFixed(2)} m</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Spatial Resolution</div>
                <div class="telemetry-val">${(analysis.spatial_resolution_m * 100).toFixed(2)} cm</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Iterations Used</div>
                <div class="telemetry-val">${analysis.iterations_used} steps</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Convergence Status</div>
                <div class="telemetry-val" style="font-size:10px">${analysis.stop_reason}</div>
            </div>
            <div style="grid-column: span 2; font-size:10px; color:var(--text-muted); padding:4px 0;">
                Accuracy note: ${analysis.accuracy_disclaimer}
            </div>
        `;
    } catch (err) {
        resultEl.innerHTML = `<div style="color:#ff4d6a;font-size:11px;">Error: ${err.message}</div>`;
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = '🎯 Run 0.01m Nested Grid Refinement'; }
    }
}

// ── Avoidance Maneuver Trajectory Visualization ──
let activeManeuverLine = null;
let activeManeuverMarker = null;

function clearManeuverTrajectory() {
    if (activeManeuverLine && scene) {
        scene.remove(activeManeuverLine);
        if (activeManeuverLine.geometry) activeManeuverLine.geometry.dispose();
        if (activeManeuverLine.material) activeManeuverLine.material.dispose();
        activeManeuverLine = null;
    }
    if (activeManeuverMarker && scene) {
        scene.remove(activeManeuverMarker);
        if (activeManeuverMarker.geometry) activeManeuverMarker.geometry.dispose();
        if (activeManeuverMarker.material) activeManeuverMarker.material.dispose();
        activeManeuverMarker = null;
    }
}

function renderManeuverTrajectory(orbitPathCoords, colorHex = 0x00f0ff) {
    clearManeuverTrajectory();
    if (!orbitPathCoords || orbitPathCoords.length < 2) return;

    const points = [];
    for (const pt of orbitPathCoords) {
        points.push(ecefToThree(pt[0], pt[1], pt[2]));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: colorHex,
        linewidth: 2,
        transparent: true,
        opacity: 0.95
    });

    activeManeuverLine = new THREE.LineLoop(geometry, material);
    scene.add(activeManeuverLine);

    if (points.length > 0) {
        const markerGeo = new THREE.SphereGeometry(0.04, 16, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc, wireframe: true });
        activeManeuverMarker = new THREE.Mesh(markerGeo, markerMat);
        activeManeuverMarker.position.copy(points[0]);
        scene.add(activeManeuverMarker);
    }
}

// ── Maneuver Planning ──
async function planAvoidanceManeuver(customNoradA = null, customNoradB = null) {
    const noradA = customNoradA || document.getElementById('avoid-norad-a')?.value || '25544';
    const noradB = customNoradB || document.getElementById('avoid-norad-b')?.value || '36086';
    const dv = parseFloat(document.getElementById('avoid-dv')?.value || '0.5');
    const clearance = parseFloat(document.getElementById('avoid-clearance')?.value || '15.0');
    const container = document.getElementById('avoidance-candidates-list');
    const btn = document.getElementById('btn-plan-avoidance');

    if (!container) return;
    if (btn) { btn.disabled = true; btn.innerHTML = 'Optimizing Maneuvers...'; }
    container.innerHTML = '<div style="padding:12px; color:var(--text-muted); font-size:12px;">Computing impulsive Gauss variations and re-propagating SGP4 trajectories...</div>';

    try {
        const url = `/api/avoidance/plan?norad_a=${noradA}&norad_b=${noradB}&miss_threshold_km=${clearance}&dv_mag=${dv}`;
        const res = await fetch(url);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const plan = data.plan || data;
        const candidates = plan.candidates || [];

        if (!candidates || candidates.length === 0) {
            container.innerHTML = '<div style="padding:12px; color:var(--warning); font-size:12px;">No avoidance candidates calculated for this pair.</div>';
            return;
        }

        // plan.miss_threshold_km is not returned by the API — use the clearance value
        // captured from the form input above (variable `clearance`)
        const thresholdDisplay = (typeof plan.miss_threshold_km === 'number')
            ? plan.miss_threshold_km.toFixed(1)
            : clearance.toFixed(1);

        let html = `
            <div class="avoidance-summary-box">
                <div class="avoidance-summary-row">
                    <span>Baseline Miss Distance</span>
                    <strong style="color:#ff4d6a;">${plan.baseline_miss_distance_km != null ? plan.baseline_miss_distance_km.toFixed(3) + ' km' : '—'}</strong>
                </div>
                <div class="avoidance-summary-row">
                    <span>Required Clearance</span>
                    <strong style="color:var(--accent);">${thresholdDisplay} km</strong>
                </div>
                <div class="avoidance-summary-row">
                    <span>Baseline TCA</span>
                    <strong>${plan.baseline_tca_minutes != null ? plan.baseline_tca_minutes.toFixed(1) + ' min' : '—'}</strong>
                </div>
            </div>
            <div class="avoidance-candidates-title">CANDIDATE MANEUVERS (${plan.candidates.length})</div>
        `;

        // Safe numeric formatters - guard against undefined/null/NaN
        const fmtKm = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) + ' km' : 'N/A';
        const fmtMs = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) : '?';
        const fmtSign = (v) => (typeof v === 'number' && isFinite(v)) ? (v >= 0 ? '+' : '') + v.toFixed(2) + ' km' : '-';

        candidates.forEach((cand, idx) => {
            const isFeasible = cand.is_feasible !== undefined ? cand.is_feasible : cand.feasible;
            const badgeClass = isFeasible ? 'badge-low' : 'badge-critical';
            const badgeText = isFeasible ? 'FEASIBLE' : 'INSUFFICIENT';
            const dir = cand.maneuver_direction || cand.direction || 'prograde';
            // Use null-coalescing to try both API field name variants
            const postMiss = fmtKm(cand.after_miss_distance_km ?? cand.post_burn_miss_km);
            const deltaMiss = fmtSign(cand.miss_distance_improvement_km ?? cand.delta_miss_km);
            const leadTime = cand.maneuver_lead_time_minutes || cand.burn_lead_time_min || 15;

            const stratLabel = cand.strategy_name || dir.replace(/_/g, ' ');
            const clearsThreshold = cand.clears_threshold ?? cand.is_feasible;
            html += `
                <div class="candidate-card ${isFeasible ? 'candidate-feasible' : ''}" id="cand-card-${idx}">
                    <div class="candidate-header">
                        <span class="candidate-strategy-name">${stratLabel}</span>
                        <span class="badge ${badgeClass}">${badgeText}</span>
                    </div>
                    <div class="candidate-metric-grid">
                        <div class="candidate-metric-item">
                            <div class="candidate-metric-label">Delta-v</div>
                            <div class="candidate-metric-val">${fmtMs(cand.delta_v_m_s)} m/s</div>
                        </div>
                        <div class="candidate-metric-item">
                            <div class="candidate-metric-label">Post-Burn Miss</div>
                            <div class="candidate-metric-val" style="color:#00e5a0;">${postMiss}</div>
                        </div>
                        <div class="candidate-metric-item">
                            <div class="candidate-metric-label">Miss Gain</div>
                            <div class="candidate-metric-val" style="color:${(cand.miss_distance_improvement_km ?? 0) >= 0 ? '#00e5a0' : '#ffa64d'};">${deltaMiss}</div>
                        </div>
                        <div class="candidate-metric-item">
                            <div class="candidate-metric-label">Burn Advance</div>
                            <div class="candidate-metric-val">${(typeof leadTime === 'number' ? leadTime.toFixed(1) : leadTime)} min</div>
                        </div>
                    </div>
                    ${cand.feasibility_reasons && cand.feasibility_reasons.length ? `<div class="candidate-reasons">${cand.feasibility_reasons.slice(0,1).join('')}</div>` : ''}
                    <button class="action-btn btn-secondary" style="font-size:11px; padding:5px 8px; margin-top:2px;"
                            onclick="visualizeCandidateTrajectory('${noradA}', '${dir}', ${cand.delta_v_m_s})">
                        ↗ Show Post-Burn 3D Trajectory
                    </button>
                </div>
            `;
        });

        html += `
            <div class="avoidance-footnote">
                Maneuvers computed via impulsive Gauss equations and SGP4 re-propagation.
                Not an operationally validated mission assurance guarantee.
            </div>
        `;

        container.innerHTML = html;
    } catch (err) {
        container.innerHTML = `<div style="padding:12px; color:#ff4d6a; font-size:12px;">Failed to plan maneuvers: ${err.message}</div>`;
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = 'Plan Avoidance Maneuvers'; }
    }
}

async function visualizeCandidateTrajectory(noradId, direction, deltaVMs) {
    try {
        const url = `/api/avoidance/trajectory?norad_id=${noradId}&direction=${direction}&delta_v_m_s=${deltaVMs}&steps=180`;
        const res = await fetch(url);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        if (data.orbit_path && data.orbit_path.length > 0) {
            renderManeuverTrajectory(data.orbit_path, 0x00f0ff);
        }
    } catch (err) {
        console.error('Trajectory visualization error:', err);
        alert(`Could not fetch trajectory: ${err.message}`);
    }
}
// ============================================================================
// ── Machine Learning Risk Prediction ──
// ============================================================================
let mlModelMetadata = null;

async function loadMLModelInfo() {
    try {
        const res = await fetch('/api/v1/ml/model_info');
        if (res.ok) {
            mlModelMetadata = await res.json();
            console.log('ML Model Metadata loaded:', mlModelMetadata);
        }
    } catch (err) {
        console.warn('ML Model Info fetch skipped:', err);
    }
}

async function runMLPrediction(customPayload = null) {
    const resultCard = document.getElementById('ml-result-display');
    const btn = document.getElementById('btn-predict-ml');

    let payload;
    if (customPayload) {
        payload = customPayload;
    } else {
        const timeToTca = parseFloat(document.getElementById('ml-time-tca')?.value || '1.2');
        const missDist = parseFloat(document.getElementById('ml-miss-dist')?.value || '320.0');
        const relSpeed = parseFloat(document.getElementById('ml-rel-speed')?.value || '14200.0');
        const objType = document.getElementById('ml-obj-type')?.value || 'DEBRIS';
        const mahalanobis = parseFloat(document.getElementById('ml-mahalanobis')?.value || '1.8');

        payload = {
            features: {
                time_to_tca_hours: timeToTca,
                miss_distance_m: missDist,
                relative_speed_m_s: relSpeed,
                secondary_object_type: objType,
                mahalanobis_distance: mahalanobis
            }
        };
    }

    if (btn) { btn.disabled = true; btn.innerHTML = 'Running Model Inference...'; }

    try {
        const res = await fetch('/api/v1/ml/predict_risk', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const pred = data.prediction || data;

        if (resultCard) {
            resultCard.style.display = 'block';
            const isHigh = pred.is_high_risk !== undefined ? pred.is_high_risk : false;
            const badgeClass = isHigh ? 'badge-critical' : 'badge-low';
            const riskText = isHigh ? 'HIGH RISK' : 'LOW RISK';
            const predRisk = pred.predicted_risk_log10 !== undefined ? pred.predicted_risk_log10 : (pred.predicted_log10_risk !== undefined ? pred.predicted_log10_risk : -30.0);
            const probPct = pred.high_risk_probability !== undefined ? (pred.high_risk_probability * 100).toFixed(1) : '0.0';

            resultCard.innerHTML = `
                <div class="ml-result-header">
                    <span class="ml-result-title">ML Conjunction Risk</span>
                    <span class="ml-risk-badge ${isHigh ? 'high' : 'low'}">${riskText}</span>
                </div>
                <div class="ml-telemetry-grid">
                    <div class="ml-telemetry-item">
                        <div class="ml-telemetry-label">Predicted Risk (log10)</div>
                        <div class="ml-telemetry-value" style="color:#00e5a0;">${predRisk.toFixed(3)}</div>
                    </div>
                    <div class="ml-telemetry-item">
                        <div class="ml-telemetry-label">High-Risk Probability</div>
                        <div class="ml-telemetry-value" style="color:${isHigh ? '#ff4d6a' : 'var(--text-primary)'}">${probPct}%</div>
                    </div>
                    <div class="ml-telemetry-item">
                        <div class="ml-telemetry-label">Classification Threshold</div>
                        <div class="ml-telemetry-value">log10(R) &ge; &minus;6.0</div>
                    </div>
                    <div class="ml-telemetry-item">
                        <div class="ml-telemetry-label">Model Confidence</div>
                        <div class="ml-telemetry-value">${pred.confidence_level || 'Validated'}</div>
                    </div>
                </div>
                <div class="ml-input-summary">
                    <strong>Inputs:</strong> Miss ${payload.features.miss_distance_m} m &middot;
                    TCA ${payload.features.time_to_tca_hours} d &middot;
                    Speed ${(payload.features.relative_speed_m_s / 1000).toFixed(1)} km/s
                </div>
            `;
        }
    } catch (err) {
        if (resultCard) {
            resultCard.style.display = 'block';
            resultCard.innerHTML = `<div style="color:#ff4d6a; font-size:11px;">ML Prediction failed: ${err.message}</div>`;
        }
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = 'Predict Collision Risk'; }
    }
}

function onMLPresetChanged(presetKey) {
    const tcaInput = document.getElementById('ml-time-tca');
    const missInput = document.getElementById('ml-miss-dist');
    const speedInput = document.getElementById('ml-rel-speed');
    const typeInput = document.getElementById('ml-obj-type');
    const mahalInput = document.getElementById('ml-mahalanobis');

    if (presetKey === 'iss_debris') {
        if (tcaInput) tcaInput.value = '0.8';
        if (missInput) missInput.value = '180.0';
        if (speedInput) speedInput.value = '14800.0';
        if (typeInput) typeInput.value = 'DEBRIS';
        if (mahalInput) mahalInput.value = '1.2';
    } else if (presetKey === 'high_speed_graze') {
        if (tcaInput) tcaInput.value = '1.5';
        if (missInput) missInput.value = '420.0';
        if (speedInput) speedInput.value = '15300.0';
        if (typeInput) typeInput.value = 'DEBRIS';
        if (mahalInput) mahalInput.value = '2.1';
    } else if (presetKey === 'nominal_leo') {
        if (tcaInput) tcaInput.value = '5.2';
        if (missInput) missInput.value = '2450.0';
        if (speedInput) speedInput.value = '11200.0';
        if (typeInput) typeInput.value = 'PAYLOAD';
        if (mahalInput) mahalInput.value = '8.5';
    }
}
// ============================================================================
// DETAILS PANELS & INTEGRATED WORKFLOWS
// ============================================================================
function showSatelliteDetail(sat) {
    const panel = document.getElementById('detail-panel');
    if (!panel) return;

    panel.classList.remove('hidden');
    panel.innerHTML = `
        <div class="detail-header">
            <div>
                <div class="detail-title">${sat.name}</div>
                <div class="detail-subtitle">NORAD ID: ${sat.norad_id} &bull; INTDES: ${sat.intdes || 'N/A'}</div>
            </div>
            <button class="close-btn" onclick="closeDetailPanel()">&times;</button>
        </div>
        <div class="telemetry-grid">
            <div class="telemetry-item">
                <div class="telemetry-label">Altitude</div>
                <div class="telemetry-val">${sat.altitude_km ? sat.altitude_km.toFixed(1) + ' km' : 'LEO'}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Speed</div>
                <div class="telemetry-val">${sat.velocity_kms ? sat.velocity_kms.toFixed(2) + ' km/s' : '7.67 km/s'}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Inclination</div>
                <div class="telemetry-val">${sat.inclination_deg ? sat.inclination_deg.toFixed(1) + '°' : '51.6°'}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Period</div>
                <div class="telemetry-val">${sat.period_min ? sat.period_min.toFixed(1) + ' min' : '92.8 min'}</div>
            </div>
        </div>
        <div style="margin-top:16px;">
            <button class="action-btn btn-secondary" style="width:100%;" onclick="switchMode('avoidance'); document.getElementById('avoid-norad-a').value='${sat.norad_id}';">
                Plan Avoidance For This Satellite
            </button>
        </div>
    `;
}

function showConjunctionDetail(conj) {
    const panel = document.getElementById('detail-panel');
    if (!panel) return;

    panel.classList.remove('hidden');
    panel.innerHTML = `
        <div class="detail-header">
            <div>
                <div class="detail-title">Conjunction Encounter</div>
                <div class="detail-subtitle">NORAD ${conj.sat1_id} vs NORAD ${conj.sat2_id}</div>
            </div>
            <button class="close-btn" onclick="closeDetailPanel()">&times;</button>
        </div>
        <div class="telemetry-grid">
            <div class="telemetry-item">
                <div class="telemetry-label">Screening Miss Dist</div>
                <div class="telemetry-val" style="color:#ff4d6a;">${conj.min_distance_km.toFixed(2)} km</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Time to TCA</div>
                <div class="telemetry-val">${conj.time_to_tca_hours ? conj.time_to_tca_hours.toFixed(2) + ' hrs' : '-'}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">TCA Epoch</div>
                <div class="telemetry-val" style="font-size:10px;">${conj.tca_iso || '-'}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">Status</div>
                <div class="telemetry-val"><span class="badge badge-critical">CONJUNCTION</span></div>
            </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:14px;">
            <button class="action-btn btn-primary" style="font-size:11px;" onclick="runGridRefinement('${conj.sat1_id}', '${conj.sat2_id}')">
                Run 0.01m Nested Grid Refinement
            </button>
            <button class="action-btn btn-secondary" style="font-size:11px;" onclick="switchMode('avoidance'); document.getElementById('avoid-norad-a').value='${conj.sat1_id}'; document.getElementById('avoid-norad-b').value='${conj.sat2_id}'; planAvoidanceManeuver('${conj.sat1_id}', '${conj.sat2_id}');">
                Plan Avoidance Maneuvers
            </button>
            <button class="action-btn btn-secondary" style="font-size:11px;" onclick="switchMode('ml_risk'); populateMLFromConjunction('${conj.sat1_id}', '${conj.sat2_id}', ${conj.min_distance_km}, ${conj.time_to_tca_hours || 1.5});">
                Run ML Risk Assessment
            </button>
        </div>
        <div id="refine-result-${conj.sat1_id}-${conj.sat2_id}" class="telemetry-grid" style="margin-top:12px;"></div>
    `;
}

function closeDetailPanel() {
    const panel = document.getElementById('detail-panel');
    if (panel) panel.classList.add('hidden');
}

function populateMLFromConjunction(sat1, sat2, missKm, tcaHrs) {
    const tcaInput = document.getElementById('ml-time-tca');
    const missInput = document.getElementById('ml-miss-dist');
    const speedInput = document.getElementById('ml-rel-speed');
    const mahalInput = document.getElementById('ml-mahalanobis');

    if (tcaInput) tcaInput.value = tcaHrs ? tcaHrs.toFixed(2) : '1.5';
    if (missInput) missInput.value = (missKm * 1000).toFixed(0);
    if (speedInput) speedInput.value = '13500.0';
    if (mahalInput) mahalInput.value = (missKm < 1.0) ? '1.5' : '4.2';

    runMLPrediction();
}
// ============================================================================
// MODE NAVIGATION & UI EVENT LISTENERS
// ============================================================================
function switchMode(targetMode) {
    currentMode = targetMode;

    document.querySelectorAll('.mode-tab').forEach(tab => {
        if (tab.getAttribute('data-mode') === targetMode) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    const viewMap = {
        'tracker': 'view-tracker',
        'screening': 'view-screening',
        'avoidance': 'view-avoidance',
        'ml_risk': 'view-ml-risk'
    };

    Object.keys(viewMap).forEach(mode => {
        const viewEl = document.getElementById(viewMap[mode]);
        if (viewEl) {
            if (mode === targetMode) {
                viewEl.classList.add('active');
            } else {
                viewEl.classList.remove('active');
            }
        }
    });
}

function setupUIListeners() {
    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const mode = e.currentTarget.getAttribute('data-mode');
            if (mode) switchMode(mode);
        });
    });

    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.filter-tab').forEach(x => x.classList.remove('active'));
            tab.classList.add('active');
            sourceFilter = tab.dataset.filter || 'all';
            updateSatList();
        });
    });

    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        let t = null;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(t);
            t = setTimeout(() => {
                searchQuery = e.target.value;
                updateSatList();
            }, 150);
        });
    }

    const screenBtn = document.getElementById('btn-run-screen');
    if (screenBtn) {
        screenBtn.addEventListener('click', () => {
            const th = parseFloat(document.getElementById('screen-threshold')?.value || '50.0');
            const hz = parseFloat(document.getElementById('screen-horizon')?.value || '3.0');
            runScreening(th, hz);
        });
    }

    const avoidBtn = document.getElementById('btn-plan-avoidance');
    if (avoidBtn) {
        avoidBtn.addEventListener('click', () => {
            planAvoidanceManeuver();
        });
    }

    const mlBtn = document.getElementById('btn-predict-ml');
    if (mlBtn) {
        mlBtn.addEventListener('click', () => {
            runMLPrediction();
        });
    }

    const mlPresetSelect = document.getElementById('ml-preset');
    if (mlPresetSelect) {
        mlPresetSelect.addEventListener('change', (e) => {
            onMLPresetChanged(e.target.value);
        });
    }

    const resetCamBtn = document.getElementById('btn-reset-cam');
    if (resetCamBtn) {
        resetCamBtn.addEventListener('click', () => {
            if (camera && controls) {
                camera.position.set(0, 1.5, 3.2);
                controls.target.set(0, 0, 0);
                controls.update();
            }
        });
    }

    const clearSceneBtn = document.getElementById('btn-clear-scene');
    if (clearSceneBtn) {
        clearSceneBtn.addEventListener('click', () => {
            clearConjunctionOverlay();
            clearManeuverTrajectory();
            closeDetailPanel();
        });
    }

    if (renderer && renderer.domElement) {
        renderer.domElement.addEventListener('click', onCanvasClick);
    }
}

function onCanvasClick(event) {
    if (!camera || !scene || !satPoints) return;

    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(satPoints);

    if (intersects.length > 0) {
        const idx = intersects[0].index;
        const valid = positions.filter(p => !p.error);
        const posObj = valid[idx];
        if (posObj) {
            selectSat(posObj.norad_id);
        }
    }
}

// Global exposure for onclick handlers
window.switchMode = switchMode;
window.planAvoidanceManeuver = planAvoidanceManeuver;
window.visualizeCandidateTrajectory = visualizeCandidateTrajectory;
window.runGridRefinement = runGridRefinement;
window.selectConjunction = selectConjunction;
window.closeDetailPanel = closeDetailPanel;
window.showConjunctionDetail = showConjunctionDetail;
window.showSatelliteDetail = showSatelliteDetail;
window.populateMLFromConjunction = populateMLFromConjunction;

window.selectSat = selectSat;
window.deselectSat = deselectSat;

// Bootstrapping
window.addEventListener('DOMContentLoaded', init);






