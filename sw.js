// おちゃこAI Web版 サービスワーカー
// アプリの殻（HTML・語彙・ONNX Runtime一式）をキャッシュして、
// 機内モードでもページが開けるようにする。モデル本体はページ側の
// 「ブラウザに保存」が担う（キャッシュ名 ochako-model-v1）。
const SHELL_CACHE = "ochako-shell-v1";
const SHELL_URLS = [
  "./",
  "./index.html",
  "./vocab.json",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.mjs",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(SHELL_CACHE)
      .then((c) => c.addAll(SHELL_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("ochako-shell-") && k !== SHELL_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = e.request.url;
  const isShell = SHELL_URLS.some((u) => url === new URL(u, self.registration.scope).href) || url === self.registration.scope;
  if (isShell) {
    e.respondWith(
      caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request))
    );
  }
  // モデルやその他は通常どおり（モデル保存はページ側 Cache API が担当）
});
