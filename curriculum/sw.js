const CACHE_NAME = 'neurons-to-agents-v2';

// All module paths to pre-cache
const MODULES = [
  '', './index.html',
  'build-01-speak/', 'build-02-smarter/', 'build-03-window/',
  'build-04-talk/', 'build-05-act/', 'build-06-remember/',
  'build-07-see/', 'build-08-scale/', 'build-09-safe/', 'build-10-free/',
  'build-capstone/',
  'founders-briefing/',
  'oc-01-gateway/', 'oc-02-sessions/', 'oc-03-context/',
  'oc-04-channels/', 'oc-05-tools/', 'oc-06-multi-agent/',
  'oc-07-automation/', 'oc-08-nodes/', 'oc-09-security/', 'oc-10-deployment/',
  'module-01-prediction-game/', 'module-02-attention/', 'module-03-position/',
  'module-04-flash-attention/', 'module-05-scaling-laws/', 'module-06-alignment/',
  'module-07-reasoning/', 'module-08-mixture-of-experts/', 'module-09-open-source/',
  'module-10-interpretability/', 'module-11-embeddings/', 'module-12-multimodal/',
];

// Pre-cache on install
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      // Cache index files for each module
      const urls = MODULES.map(m => m.endsWith('') ? m + './index.html' : m);
      return cache.addAll(urls);
    }).then(() => self.skipWaiting())
  );
});

// Clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Cache-first, fallback to network
self.addEventListener('fetch', event => {
  // Skip non-GET and external requests (like Pyodide CDN)
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        // Cache new requests on the fly
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
