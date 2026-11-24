// MicroAlerta: funciona sin conexión en el stand.
// - Fotos, espectros, íconos y fuentes: primero la copia guardada (rápido y sin señal).
// - La app y las listas de reportes y galería: primero la red; si no hay señal, la copia guardada.
// - El resto de la API (iniciar sesión, enviar reportes) siempre va a la red.
const CACHE = "microalerta-v21";
const SHELL = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png", "/logo-mark.png", "/logo-full.jpg"];
const LISTAS = ["/api/gallery", "/api/reports", "/api/health", "/api/me"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

const guardar = (req, res) => {
  if (res && (res.ok || res.type === "opaque")) {
    const copia = res.clone();
    caches.open(CACHE).then(c => c.put(req, copia));
  }
  return res;
};

const primeroRed = req => fetch(req).then(r => guardar(req, r))
  .catch(() => caches.match(req, { ignoreSearch: req.mode === "navigate" }).then(r => r || caches.match("/")));

const primeroCopia = req => caches.match(req).then(r => r || fetch(req).then(res => guardar(req, res)));

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const fuente = url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com";
  if (url.origin !== location.origin && !fuente) return;
  if (fuente || /^\/(galeria|raman|icons|api\/galeria)\//.test(url.pathname) || /^\/logo-/.test(url.pathname)) {
    e.respondWith(primeroCopia(req));
  } else if (url.pathname.startsWith("/api/")) {
    if (LISTAS.includes(url.pathname)) e.respondWith(primeroRed(req));
  } else {
    e.respondWith(primeroRed(req));
  }
});
