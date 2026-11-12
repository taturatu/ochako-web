// おちゃこAI Web版 サービスワーカー
// 方針：
// - 殻ドキュメント（index.html / worker.js / vocab.json / ナビゲーション）は network-first。
//   オンライン時は常に最新を取得してキャッシュを更新し、オフライン時のみキャッシュを使う。
//   （キャッシュ優先だとサイトを更新しても保存済みユーザーに新しいレイアウトが届かないため）
// - CDN の ONNX Runtime 一式（バージョン固定URL）は cache-first（変わらないファイルなので）。
// - モデル本体はページ側の「ブラウザに保存」が担当（キャッシュ名 ochako-model-v1、ここでは触らない）。
const SHELL_CACHE = "ochako-shell-v4";
const DOC_URLS = ["./", "./index.html", "./vocab.json", "./worker.js"];
const CDN_URLS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.mjs",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(SHELL_CACHE)
      .then((c) => c.addAll([...DOC_URLS, ...CDN_URLS]))
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

function isDocRequest(request, url) {
  if (request.mode === "navigate") return true;
  const scope = self.registration.scope;
  return DOC_URLS.some((u) => url === new URL(u, scope).href);
}

self.addEventListener("fetch", (e) => {
  const url = e.request.url;
  if (CDN_URLS.includes(url)) {
    // バージョン固定のCDNファイルはキャッシュ優先
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request))
    );
    return;
  }
  if (isDocRequest(e.request, url)) {
    // 殻ドキュメントは network-first（最新を取りに行き、キャッシュも更新。失敗時のみキャッシュ）
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const clone = res.clone();
          caches.open(SHELL_CACHE).then((c) => c.put(e.request, clone));
          return res;
        })
        .catch(() => caches.match(e.request, { ignoreSearch: true }))
    );
    return;
  }
  // モデルやその他は通常どおり（モデル保存はページ側 Cache API が担当）
});
