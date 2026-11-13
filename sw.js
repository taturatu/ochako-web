// おちゃこAI Web版 サービスワーカー
// 方針：
// - 殻ドキュメント（index.html / worker.js / vocab.json / モデル版数JSON / ナビゲーション）は network-first。
//   オンライン時は常に最新を取得してキャッシュを更新し、オフライン時のみキャッシュを使う。
// - CDN の ONNX Runtime 一式（バージョン固定URL）は cache-first。
// - ナビゲーション応答に COOP/COEP ヘッダを付与し、crossOriginIsolated を有効化する
//   （SharedArrayBuffer が使えるようになり、WASM の実マルチスレッドが効く。いわゆる coi-serviceworker 方式）
// - モデル本体はページ側の「ブラウザに保存」が担当（キャッシュ名 ochako-model-v1、ここでは触らない）。
//
// v6 での修正（2026-11-13、無限準備の原因切り分けを受けて）：
// - インストール時の事前キャッシュは「全URL一括（addAll）」をやめ、URLごとのベストエフォート方式に変更。
//   addAll は1件でも失敗すると全体が失敗して SW が永遠に有効化されない。実測では .wasm 応答の
//   cache.put が "Cache.put() encountered a network error" で落ちる環境があり、その場合に
//   「古いSW＋古いページ」に固定されて新しい修正が一切届かなくなるのが無限準備の温床だった。
// - 新バージョンは待機させず即時有効化（skipWaiting）する。古い版で固まった端末を次回訪問で救済するため。
//   ページ側の「更新する」ボタン方式は廃止（メッセージ自体は互換のため残す）。
// v8（2026-11-13）：v5.12（ターン境界スライディングウィンドウ＋文末完結バイアス）を届けるための版上げ。
const SHELL_CACHE = "ochako-shell-v8";
const DOC_URLS = ["./", "./index.html", "./vocab.json", "./worker.js", "./model-version.json"];
const CDN_URLS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort-wasm-simd-threaded.mjs",
];

// cache.put は応答のストリーム読み取りで network error になることがある（.wasm で実測）。
// 失敗したら本体を読み切ってから作り直した応答で再試行し、それも失敗したら諦める（ベストエフォート）。
async function cachePutSafe(cache, key, res) {
  try {
    await cache.put(key, res.clone());
    return true;
  } catch (e) {
    try {
      const buf = await res.arrayBuffer();
      const headers = new Headers(res.headers);
      await cache.put(key, new Response(buf, { status: res.status, statusText: res.statusText, headers }));
      return true;
    } catch (e2) {
      return false;
    }
  }
}

async function precacheBestEffort() {
  const cache = await caches.open(SHELL_CACHE);
  for (const u of [...DOC_URLS, ...CDN_URLS]) {
    try {
      const res = await fetch(u);
      if (res && res.ok) await cachePutSafe(cache, u, res);
    } catch (e) {
      // 1件の失敗でインストール全体を落とさない
    }
  }
}

self.addEventListener("install", (e) => {
  e.waitUntil(precacheBestEffort().then(() => self.skipWaiting()));
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
      caches.match(e.request).then((hit) => {
        if (hit) return hit;
        return fetch(e.request).then((res) => {
          // 初回取得時にベストエフォートでキャッシュ（失敗しても応答は返す）
          if (res && res.ok) {
            const clone = res.clone();
            caches.open(SHELL_CACHE).then((c) => cachePutSafe(c, e.request, clone)).catch(() => {});
          }
          return res;
        });
      })
    );
    return;
  }
  if (isDocRequest(e.request, url)) {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const clone = res.clone();
          caches.open(SHELL_CACHE).then((c) => cachePutSafe(c, e.request, clone)).catch(() => {});
          // 分離ヘッダはナビゲーションだけでなく worker.js 等の殻リソース全てに付ける。
          // crossOriginIsolated なページでは、COEP ヘッダのない Worker スクリプトは
          // 起動できず「準備しています」のまま固まる（2026-11-13 実測で原因特定）。
          // 304 応答は本体がなくヘッダ付け替えができないためそのまま返す。
          return res.status === 304 ? res : withIsolationHeaders(res);
        })
        .catch(async () => {
          const hit = await caches.match(e.request, { ignoreSearch: true });
          if (!hit) throw new Error("offline and not cached");
          return withIsolationHeaders(hit);
        })
    );
    return;
  }
  // モデルやその他は通常どおり
});
