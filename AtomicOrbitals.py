<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atomic Orbital Visualiser</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        #container {
            position: relative;
            width: 100%;
            height: 100vh;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            max-width: 250px;
        }
        #controls {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            max-width: 90%;
            max-height: 40vh;
            overflow-y: auto;
        }
        button {
            margin: 5px;
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 4px;
            border: 1px solid #ccc;
            background-color: #f8f8f8;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #e8e8e8;
        }
        button.active {
            background-color: #4CAF50;
            color: white;
        }
        .quality-selector {
            margin: 10px 0;
        }
        .quality-selector label {
            margin-right: 10px;
        }
        .orbital-group {
            margin-bottom: 10px;
            border: 1px solid #ddd;
            padding: 8px;
            border-radius: 5px;
            background-color: #fff;
        }
        .orbital-group h3 {
            margin-top: 0;
            margin-bottom: 8px;
        }
        .action-buttons {
            margin-bottom: 10px;
        }
        .add-group-button {
            font-size: 0.8em;
            padding: 4px 8px;
            margin-left: 8px;
        }
        #fps-counter {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
        }
        #language-switcher {
            position: absolute;
            top: 10px;
            right: 80px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 24px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            z-index: 1000;
        }
        .lang-flag {
            margin: 0 5px;
            cursor: pointer;
            opacity: 0.6;
            transition: opacity 0.3s;
        }
        .lang-flag:hover, .lang-flag.active {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            <h2 class="lang" data-en="Atomic Orbital Visualiser" data-tr="Atomun Orbitalleri">Atomic Orbital Visualiser</h2>
            <div id="orbital-info" class="lang" data-en="Electron Configuration: 1s" data-tr="Elektron Konfigürasyonu: 1s">Electron Configuration: 1s</div>
        </div>
        <div id="controls">
            <div class="action-buttons">
                <button id="clear-orbitals" class="lang" data-en="Clear All" data-tr="Tümünü Temizle">Clear All</button>
                <button id="add-all-orbitals" class="lang" data-en="Add All Orbitals" data-tr="Tüm Orbitalleri Ekle">Add All Orbitals</button>
            </div>
            <div class="quality-selector">
                <label class="lang" data-en="Quality:" data-tr="Kalite:">Quality:</label>
                <input type="radio" name="quality" id="low" value="low" checked> <label for="low" class="lang" data-en="Low" data-tr="Düşük">Low</label>
                <input type="radio" name="quality" id="medium" value="medium"> <label for="medium" class="lang" data-en="Medium" data-tr="Orta">Medium</label>
                <input type="radio" name="quality" id="high" value="high"> <label for="high" class="lang" data-en="High" data-tr="Yüksek">High</label>
            </div>
            <div class="quality-selector">
                <label class="lang" data-en="Performance:" data-tr="Performans:">Performance:</label>
                <input type="checkbox" id="pause-render" name="pause-render">
                <label for="pause-render" class="lang" data-en="Pause render during interaction" data-tr="Etkileşim sırasında renderi duraklat">Pause render during interaction</label>
            </div>
            
            <div class="orbital-groups" id="orbital-buttons">
            </div>
        </div>
        <div id="fps-counter">FPS: 0</div>
        <div id="language-switcher">
            <span class="lang-flag active" data-lang="en">En</span>
            <span class="lang-flag" data-lang="tr">Tr</span>
        </div>
    </div>

    <script>
        let scene, camera, renderer, controls;
        let meshes = [];
        let isRendering = true;
        let lastFrameTime = 0;
        let isPaused = false;
        let currentLanguage = 'en';
        
        // translation 
        const translations = {
            'en': {
                'title': 'Atomic Orbital Visualiser',
                'clearAll': 'Clear All',
                'addAllOrbitals': 'Add All Orbitals',
                'quality': 'Quality:',
                'low': 'Low',
                'medium': 'Medium',
                'high': 'High',
                'performance': 'Performance:',
                'pauseRender': 'Pause render during interaction',
                'activeOrbitals': 'Active Orbitals:',
                'noActiveOrbitals': 'No Active Orbitals',
                'addAll': 'Add All',
                'fps': 'FPS'
            },
            'tr': {
                'title': 'Atomun Orbitalleri',
                'clearAll': 'Tümünü Temizle',
                'addAllOrbitals': 'Tüm Orbitalleri Ekle',
                'quality': 'Grafikler:',
                'low': 'Düşük',
                'medium': 'Orta',
                'high': 'Yüksek',
                'performance': 'Performans:',
                'pauseRender': 'Etkileşim sırasında renderi duraklat',
                'activeOrbitals': 'Aktif Orbitaller:',
                'noActiveOrbitals': 'Aktif Orbital Yok',
                'addAll': 'Tümünü Ekle',
                'fps': 'FPS'
            }
        };
        
        // orbital calculations
        let orbitalCache = {};

        const availableOrbitals = [
            { n: 1, l: 0, m: 0, name: "1s", color: 0xff0000 },
            { n: 2, l: 0, m: 0, name: "2s", color: 0x00ff00 },
            { n: 2, l: 1, m: -1, name: "2p_x", color: 0x0000ff },
            { n: 2, l: 1, m: 0, name: "2p_y", color: 0x00ffff },
            { n: 2, l: 1, m: 1, name: "2p_z", color: 0xff00ff },
            { n: 3, l: 0, m: 0, name: "3s", color: 0xffff00 },
            { n: 3, l: 1, m: -1, name: "3p_x", color: 0x800080 },
            { n: 3, l: 1, m: 0, name: "3p_y", color: 0xa52a2a },
            { n: 3, l: 1, m: 1, name: "3p_z", color: 0xffc0cb },
            { n: 4, l: 0, m: 0, name: "4s", color: 0xffa500 },
            { n: 3, l: 2, m: -2, name: "3d_xy", color: 0x00ff00 },
            { n: 3, l: 2, m: -1, name: "3d_yz", color: 0x808080 },
            { n: 3, l: 2, m: 0, name: "3d_z²", color: 0x808000 },
            { n: 3, l: 2, m: 1, name: "3d_xz", color: 0x000080 },
            { n: 3, l: 2, m: 2, name: "3d_x²-y²", color: 0x008080 },
            { n: 4, l: 1, m: -1, name: "4p_x", color: 0xff7f50 },
            { n: 4, l: 1, m: 0, name: "4p_y", color: 0xffd700 },
            { n: 4, l: 1, m: 1, name: "4p_z", color: 0xee82ee }
        ];

        const qualitySettings = {
            'low': { resolution: 12 },
            'medium': { resolution: 20 },
            'high': { resolution: 32 }
        };

        let currentQuality = 'low';
        let currentOrbitals = [availableOrbitals[0]]; // start with 1s
        
        let geometryCache = {};
        let materialCache = {};

        // track fps
        let frameCount = 0;
        let lastFpsUpdate = 0;

        // Language switcher
        function switchLanguage(lang) {
            currentLanguage = lang;
            
            // Update active flag
            document.querySelectorAll('.lang-flag').forEach(flag => {
                flag.classList.remove('active');
                if (flag.dataset.lang === lang) {
                    flag.classList.add('active');
                }
            });
            
            // Update all elements with language class
            document.querySelectorAll('.lang').forEach(element => {
                if (element.dataset[lang]) {
                    element.textContent = element.dataset[lang];
                }
            });
            
            // Recreate orbital buttons with new language
            document.getElementById('orbital-buttons').innerHTML = '';
            createOrbitalButtons();
            
            // Update orbital info
            updateOrbitalInfo();
        }

        // organize orbitals by type
        function groupOrbitalsByType() {
            const groups = {};
            
            availableOrbitals.forEach(orbital => {
                const baseName = orbital.name.split('_')[0];
                if (!groups[baseName]) {
                    groups[baseName] = [];
                }
                groups[baseName].push(orbital);
            });
            
            return groups;
        }

        // create buttons for orbitals
        function createOrbitalButtons() {
            const orbitalGroups = groupOrbitalsByType();
            const container = document.getElementById('orbital-buttons');
            
            // sort groups by n and l values
            const sortedGroupNames = Object.keys(orbitalGroups).sort((a, b) => {
                const nA = parseInt(a[0]);
                const nB = parseInt(b[0]);
                
                if (nA !== nB) return nA - nB;
                
                const typeA = a.substr(1);
                const typeB = b.substr(1);
                const typeOrder = { 's': 0, 'p': 1, 'd': 2, 'f': 3 };
                return typeOrder[typeA] - typeOrder[typeB];
            });
            
            sortedGroupNames.forEach(groupName => {
                const orbitals = orbitalGroups[groupName];
                
                const groupDiv = document.createElement('div');
                groupDiv.className = 'orbital-group';
                
                const groupHeader = document.createElement('h3');
                groupHeader.textContent = groupName;

                const addGroupButton = document.createElement('button');
                addGroupButton.textContent = translations[currentLanguage]['addAll'];
                addGroupButton.className = 'add-group-button';
                addGroupButton.addEventListener('click', function() {
                    addOrbitalGroup(orbitals);
                });
                groupHeader.appendChild(addGroupButton);
                
                groupDiv.appendChild(groupHeader);
                
                orbitals.forEach(orbital => {
                    const button = document.createElement('button');
                    const displayName = orbital.name.includes('_') ? 
                        orbital.name.split('_')[1] : 
                        orbital.name;
                    
                    button.textContent = displayName;
                    button.dataset.n = orbital.n;
                    button.dataset.l = orbital.l;
                    button.dataset.m = orbital.m;
                    button.dataset.name = orbital.name;
                    
                    button.addEventListener('click', function() {
                        toggleOrbital(this, orbital);
                    });
                    
                    if (isOrbitalActive(orbital)) {
                        button.classList.add('active');
                    }
                    
                    groupDiv.appendChild(button);
                });
                
                container.appendChild(groupDiv);
            });
        }
        
        function isOrbitalActive(orbital) {
            return currentOrbitals.some(o => 
                o.n === orbital.n && o.l === orbital.l && o.m === orbital.m
            );
        }
        
        // orbital on/off
        function toggleOrbital(button, orbital) {
            const isActive = button.classList.contains('active');
            
            if (isActive) {
                // Remove orbital
                currentOrbitals = currentOrbitals.filter(o => 
                    !(o.n === orbital.n && o.l === orbital.l && o.m === orbital.m)
                );
                button.classList.remove('active');
            } else {
                if (!isOrbitalActive(orbital)) {
                    currentOrbitals.push(orbital);
                    button.classList.add('active');
                }
            }
            
            renderOrbitals();
        }
        
        // add all orbitals from a group
        function addOrbitalGroup(orbitals) {
            let changed = false;
            
            orbitals.forEach(orbital => {
                if (!isOrbitalActive(orbital)) {
                    currentOrbitals.push(orbital);
                    changed = true;
                    
                    const buttons = document.querySelectorAll(`button[data-name="${orbital.name}"]`);
                    buttons.forEach(button => button.classList.add('active'));
                }
            });
            
            if (changed) renderOrbitals();
        }
        
        function addAllOrbitals() {
            if (currentOrbitals.length === availableOrbitals.length &&
                availableOrbitals.every(o => isOrbitalActive(o))) {
                return; 
            }
            
            currentOrbitals = [...availableOrbitals];
            
            const buttons = document.querySelectorAll('#orbital-buttons button:not(.add-group-button)');
            buttons.forEach(button => button.classList.add('active'));
            
            renderOrbitals();
        }

        function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);

            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;

            // renderer
            renderer = new THREE.WebGLRenderer({ 
                antialias: false, // disable antialiasing for performance
                powerPreference: "high-performance" 
            });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio
            document.getElementById('container').appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.rotateSpeed = 0.5;
            
            controls.addEventListener('start', function() {
                if (document.getElementById('pause-render').checked) {
                    isPaused = true;
                }
            });
            
            controls.addEventListener('end', function() {
                isPaused = false;
                if (document.getElementById('pause-render').checked) {
                    render();
                }
            });

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);

            const nucleusGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const nucleusMaterial = new THREE.MeshPhongMaterial({ color: 0x333333 });
            const nucleus = new THREE.Mesh(nucleusGeometry, nucleusMaterial);
            scene.add(nucleus);

            // setup language switcher
            document.querySelectorAll('.lang-flag').forEach(flag => {
                flag.addEventListener('click', () => {
                    switchLanguage(flag.dataset.lang);
                });
            });

            // create orbital buttons
            createOrbitalButtons();

            // draw first orbital
            renderOrbitals();

            // event listeners
            document.getElementById('clear-orbitals').addEventListener('click', clearOrbitals);
            document.getElementById('add-all-orbitals').addEventListener('click', addAllOrbitals);
            document.getElementById('pause-render').addEventListener('change', function() {
                isPaused = false; // reset pause state when preference changes
            });
            
            const qualityRadios = document.querySelectorAll('input[name="quality"]');
            qualityRadios.forEach(radio => {
                radio.addEventListener('change', function() {
                    if (this.checked) {
                        currentQuality = this.value;
                        
                        // clear geometry cache on quality change
                        for (const key in geometryCache) {
                            if (geometryCache[key]) {
                                geometryCache[key].dispose();
                            }
                        }
                        geometryCache = {};
                        
                        // clear orbital cache on quality change
                        orbitalCache = {};
                        
                        renderOrbitals();
                    }
                });
            });

            let resizeTimeout;
            window.addEventListener('resize', function() {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(onWindowResize, 250);
            });

            // start animation loop
            animate();
        }

        function getCachedOrbitalData(n, l, m, resolution, scaleFactor) {
            const key = `${n}_${l}_${m}_${resolution}_${scaleFactor}`;
            
            if (orbitalCache[key]) {
                return orbitalCache[key];
            }
            
            const data = calculateOrbital(n, l, m, resolution, scaleFactor);
            orbitalCache[key] = data;
            return data;
        }

        function calculateOrbital(n, l, m, resolution, scaleFactor = 0.5) {
            const data = [];
            
            
            const nScaleFactor = n * n; 
                        
            for (let i = 0; i <= resolution; i++) {
                const theta = (i / resolution) * Math.PI;
                for (let j = 0; j <= resolution; j++) {
                    const phi = (j / resolution) * 2 * Math.PI;
                    
                    let Y = 0;
                    
                    if (l === 0) {
                        Y = 0.5; // s orbital
                    } else if (l === 1) {
                        if (m === 0) {
                            Y = Math.cos(theta); // p_z
                        } else if (m === 1 || m === -1) {
                            Y = Math.sin(theta) * (m === 1 ? Math.cos(phi) : Math.sin(phi)); // p_x, p_y
                        }
                    } else if (l === 2) {
                        if (m === 0) {
                            Y = (3 * Math.cos(theta) * Math.cos(theta) - 1) / 2; // d_z²
                        } else if (m === 1 || m === -1) {
                            Y = Math.sin(theta) * Math.cos(theta) * (m === 1 ? Math.cos(phi) : Math.sin(phi)); // d_xz, d_yz
                        } else if (m === 2 || m === -2) {
                            Y = Math.sin(theta) * Math.sin(theta) * (m === 2 ? Math.cos(2 * phi) : Math.sin(2 * phi)); // d_x²-y², d_xy
                        }
                    }
                    
                    Y = Math.abs(Y); 
                    
                   
                    const r = 1; 
                    let R = Math.exp(-r/(2*n)) * Math.pow(r/n, l);
                    
                    const psi = R * Y; 
                    
                    const rScale = scaleFactor * nScaleFactor;
                    
                    const x = rScale * psi * Math.sin(theta) * Math.cos(phi);
                    const y = rScale * psi * Math.sin(theta) * Math.sin(phi);
                    const z = rScale * psi * Math.cos(theta);
                    
                    data.push({ position: new THREE.Vector3(x, y, z), theta, phi });
                }
            }
            
            return data;
        }       
        function getGeometry(resolution) {
            const key = `geometry_${resolution}`;
            
            if (!geometryCache[key]) {
                geometryCache[key] = new THREE.SphereGeometry(1, resolution, resolution);
            }
            
            return geometryCache[key].clone(); 
        }
        
        function getMaterial(color) {
            const key = `material_${color}`;
            
            if (!materialCache[key]) {
                materialCache[key] = new THREE.MeshPhongMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.5,
                    side: THREE.DoubleSide,
                    depthWrite: false 
                });
            }
            
            return materialCache[key];
        }

        function renderOrbitals() {
            // if it stopped don't continue
            if (isPaused) return;
            
            // delete all meshes
            meshes.forEach(mesh => scene.remove(mesh));
            meshes = [];
            
            if (currentOrbitals.length === 0) {
                updateOrbitalInfo();
                render(); // forced render
                return;
            }
            
            const maxN = Math.max(...currentOrbitals.map(o => o.n));
            const resolution = qualitySettings[currentQuality].resolution;
            
            currentOrbitals.forEach(orbital => {
                const scale = 0.8 + 0.4 * (orbital.n / maxN);
                
                const orbitalData = getCachedOrbitalData(
                    orbital.n, orbital.l, orbital.m, 
                    resolution, scale
                );
                
                const geometry = getGeometry(resolution);
                const material = getMaterial(orbital.color);
                
                const mesh = new THREE.Mesh(geometry, material);
                
                // 2p and 3d orbitals render
                if (orbital.name.startsWith("2p") || orbital.name.startsWith("3d")) {
                    mesh.renderOrder = 1;
                } else {
                    mesh.renderOrder = 0;
                }
                
                // orbitals scale settings:
                let orbitalAdjustment = 1.0;
                if (orbital.name.startsWith("3p")) {
                    orbitalAdjustment = 2.8;  // 3p 
                } else if (orbital.name.startsWith("4p")) {
                    orbitalAdjustment = 4.0;  // 4p 
                } else if (orbital.l === 2) {
                    orbitalAdjustment = 10;  // 3d 
                } else if (orbital.name.startsWith("2p")) {
                    orbitalAdjustment = 2.6; 
                } else if (orbital.name.startsWith("2s")) {
                    orbitalAdjustment = 2
                } else if (orbital.name.startsWith("3s")) {
                    orbitalAdjustment = 1.5
                } else if (orbital.name.startsWith("4s")) {
                    orbitalAdjustment = 1.5
                }
                
                const positions = geometry.attributes.position;
                for (let i = 0; i < positions.count; i++) {
                    const data = orbitalData[i % orbitalData.length];
                    positions.setXYZ(i, 
                        data.position.x * orbitalAdjustment, 
                        data.position.y * orbitalAdjustment, 
                        data.position.z * orbitalAdjustment
                    );
                }
                
                geometry.computeVertexNormals();
                scene.add(mesh);
                meshes.push(mesh);
            });
            
            updateOrbitalInfo();
            render(); 
        }

        function updateOrbitalInfo() {
            const config = {};
            currentOrbitals.forEach(orbital => {
                const name = orbital.name.split('_')[0];
                if (!config[name]) {
                    config[name] = [];
                }
                config[name].push(orbital.name);
            });
            
            // principal quantum number
            const nGroups = {};
            for (const [orbitalType, orbitals] of Object.entries(config)) {
                const n = orbitalType[0];
                if (!nGroups[n]) {
                    nGroups[n] = [];
                } 
                nGroups[n].push(...orbitals);
            }
            
            let result = translations[currentLanguage]['activeOrbitals'] + "<br>";
            if (Object.keys(nGroups).length === 0) {
                result += translations[currentLanguage]['noActiveOrbitals'];
            } else {
                Object.keys(nGroups).sort().forEach(n => {
                    result += `n=${n}: ${nGroups[n].join(", ")}<br>`;
                });
            }
            
            document.getElementById('orbital-info').innerHTML = result;
        }

        // delete all orbitals
        function clearOrbitals() {
            if (currentOrbitals.length === 0) return; 
            
            currentOrbitals = [];
            
            // button
            const buttons = document.querySelectorAll('#orbital-buttons button:not(.add-group-button)');
            buttons.forEach(button => {
                button.classList.remove('active');
            });
            
            renderOrbitals();
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            render(); 
        }

        function render() {
            renderer.render(scene, camera);
        }

        function animate(time) {
            requestAnimationFrame(animate);
            
            controls.update();
            
            if (isPaused) return;
            
            frameCount++;
            if (time - lastFpsUpdate > 1000) {
                const fps = Math.round((frameCount * 1000) / (time - lastFpsUpdate));
                document.getElementById('fps-counter').textContent = `${translations[currentLanguage]['fps']}: ${fps}`;
                lastFpsUpdate = time;
                frameCount = 0;
            }
            
            render();
        }

        window.addEventListener('load', init);
    </script>
</body>
</html>
