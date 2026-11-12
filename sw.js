// おちゃこAI Web版 サービスワーカー
// 方針：
// - 殻ドキュメント（index.html / worker.js / vocab.json / モデル版数JSON / ナビゲーション）は network-first。
//   オンライン時は常に最新を取得してキャッシュを更新し、オフライン時のみキャッシュを使う。
// - CDN の ONNX Runtime 一式（バージョン固定URL）は cache-first。
// - ナビゲーション応答に COOP/COEP ヘッダを付与し、crossOriginIsolated を有効化する
//   （SharedArrayBuffer が使えるようになり、WASM の実マルチスレッドが効く。いわゆる coi-serviceworker 方式）
// - 新バージョンは即時適用せず待機させ、ページ側の「更新する」ボタンで SKIP_WAITING を受けて切り替える。
// - モデル本体はページ側の「ブラウザに保存」が担当（キャッシュ名 ochako-model-v1、ここでは触らない）。
const SHELL_CACHE = "ochako-shell-v5";
const DOC_URLS = ["./", "./index.html", "./vocab.json", "./worker.js", "./model-version.json"];
const CDN_URLS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.mjs",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(SHELL_CACHE).then((c) => c.addAll([...DOC_URLS, ...CDN_URLS]))
    // skipWaiting はしない（更新はユーザーのボタン操作で切り替える。初回導入時は待機相手がいないので自動で有効化される）
  );
});

self.addEventListener("message", (e) => {
  if (e.data === "SKIP_WAITING") self.skipWaiting();
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

// crossOriginIsolated 用ヘッダを付けた応答を作る
function withIsolationHeaders(res) {
  const headers = new Headers(res.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "credentialless");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

self.addEventListener("fetch", (e) => {
  const url = e.request.url;
  if (CDN_URLS.includes(url)) {
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request))
    );
    return;
  }
  if (isDocRequest(e.request, url)) {
    const isNav = e.request.mode === "navigate";
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const clone = res.clone();
          caches.open(SHELL_CACHE).then((c) => c.put(e.request, clone));
          return isNav ? withIsolationHeaders(res) : res;
        })
        .catch(async () => {
          const hit = await caches.match(e.request, { ignoreSearch: true });
          if (!hit) throw new Error("offline and not cached");
          return isNav ? withIsolationHeaders(hit) : hit;
        })
    );
    return;
  }
  // モデルやその他は通常どおり
});
