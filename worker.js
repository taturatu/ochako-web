// おちゃこAI 生成ワーカー（UIスレッドを止めないため推論はここで実行）
// プロンプト形式は製品版アプリと同一：
//   ユーザー：{発話}\nAI：{返事}\n…\nAI：
// 停止マーカーは「\nユーザー：」「\nAI：」。
// モデルがKVキャッシュ版（入力に offset / past_* を持つ）の場合は高速モード、
// そうでなければ従来の全系列モードで動く（自動判定）。
// KV版では会話の事前計算（Prefix Cache）を再利用し、2ターン目以降は差分だけを計算する。
let session = null, itos = [], kvMode = false;
let stoi = new Map();

const MAX_SEQ = 512;
// デコード設定（標準は製品標準 B1.5。画面から変更できる。minP は v5.11 で追加した内部既定）
let CFG = { temperature: 0.65, topK: 40, repPenalty: 1.15, maxNew: 150, threads: 1, ep: 'auto', maxSeq: 512, lowMem: false, minP: 0.05 };
const effSeq = () => Math.min(CFG.maxSeq || MAX_SEQ, MAX_SEQ);
const BOS = 1, EOS = 2;
const NL = 12, NH = 10, HD = 64;
let cancelFlag = false;

// Prefix Cache: past がカバーするトークン列とKV本体
let cacheIds = null;   // number[]（past がカバーする範囲）
let cachePast = null;  // [{k: Float32Array, v: Float32Array} x NL]

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function buildPrompt(history) {
  let s = '';
  for (const m of history) {
    if (m.role === 'user') s += 'ユーザー：' + m.text.trim() + '\n';
    else s += 'AI：' + m.text.trim() + '\n';
  }
  return s + 'AI：';
}

function encode(prompt) {
  const ids = [BOS];
  for (const ch of prompt) { const t = stoi.get(ch); if (t !== undefined) ids.push(t); }
  return ids;
}

function encodeNoBos(text) {
  const ids = [];
  for (const ch of text) { const t = stoi.get(ch); if (t !== undefined) ids.push(t); }
  return ids;
}

// v5.12（Gemini案4）：文脈が上限を超えるときは、末尾を機械的に切るのではなく
// 「古いターンを丸ごと落とす」スライディングウィンドウにする。
// 先頭BOSと「ユーザー：」で始まる構造は絶対に壊さない。
function buildIds(history) {
  const budget = effSeq();
  const segs = []; // {role, ids}  ids は BOS を含まない
  for (const m of history) {
    const head = m.role === 'user' ? 'ユーザー：' : 'AI：';
    segs.push({ role: m.role === 'user' ? 'user' : 'ai', head, ids: encodeNoBos(head + m.text.trim() + '\n') });
  }
  const tail = { role: 'tail', head: 'AI：', ids: encodeNoBos('AI：') };
  const total = () => 1 + segs.reduce((a, s) => a + s.ids.length, 0) + tail.ids.length;
  // 古いターンからペア（ユーザー＋AI）単位で落とす
  while (total() > budget && segs.length > 1) {
    if (segs.length >= 3 && segs[0].role === 'user' && segs[1].role === 'ai') segs.splice(0, 2);
    else segs.splice(0, 1);
  }
  // それでも溢れる場合は、最も長いユーザー発話の本文を先頭側から削る
  // （「ユーザー：」の5トークンは必ず残す）
  while (total() > budget) {
    let li = -1, ll = 0;
    segs.forEach((s, i) => { if (s.role === 'user' && s.ids.length > ll) { ll = s.ids.length; li = i; } });
    if (li < 0 || ll <= 6) break;
    const over = total() - budget;
    const s = segs[li];
    const keepBody = Math.max(0, s.ids.length - 5 - over);
    s.ids = s.ids.slice(0, 5).concat(s.ids.slice(5 + (s.ids.length - 5 - keepBody)));
  }
  const ids = [BOS];
  for (const s of segs) ids.push(...s.ids);
  ids.push(...tail.ids);
  return ids.slice(0, budget);
}

// v5.12（Gemini案3）：生成が maxNew の8割を超えたら、文末方向へ小さなバイアスを掛けて
// 尻切れを防ぐ。コードらしい出力（タグや関数呼び出し等を含む）では無効化する。
const CODE_HINTS = ['<', 'def ', 'import ', 'print(', '`', '{', '}', '=>', 'function ', 'console.'];
function looksCode(text) {
  const t = text.slice(-200);
  return CODE_HINTS.some((h) => t.includes(h));
}

// ===== v5.11 デコード改善 =====
// (a) 繰り返しペナルティは「直近64トークン以内」だけに適用し、句読点と1文字の基本助詞は
//     対象から外す（日本語の文法上必要な文字まで罰して文を崩さないため）。
const PENALTY_EXEMPT = new Set(['、', '。', '！', '？', '!', '?', '・', '…', '「', '」', '『', '』', '（', '）',
  'は', 'が', 'の', 'に', 'を', 'て', 'で', 'と', 'も', 'へ', 'や']);
const PENALTY_WINDOW = 64;
// (b) 「わからない」系のNG表現。検知したら全文再生成はせず、その表現の開始位置まで
//     巻き戻して開始トークンをマスクし、別の表現へ分岐させる（生成ループ側で処理）。
const NG_PHRASES = ['わかりませんでした', '分かりませんでした', 'わかりません', '分かりません', 'わからないです',
  '知りません', '知らないです', '存じません', 'お答えできません', '答えられません', '分からない', 'わからない'];
function ngIndex(text) {
  let best = -1;
  for (const p of NG_PHRASES) {
    const i = text.indexOf(p);
    if (i >= 0 && (best < 0 || i < best)) best = i;
  }
  return best;
}

// サンプリング（top-k 閾値は Float32Array のネイティブソートで取得）
// v5.11: ペナルティの窓制限＋助詞・句読点除外、Min-p フィルタ、exclude（巻き戻し時のマスク）対応
let sortBuf = null;
function sampleNext(logits, ids, rand, exclude, endBias) {
  const V = itos.length;
  const last = Float32Array.from(logits);
  const windowIds = new Set(ids.slice(-PENALTY_WINDOW));
  for (const t of windowIds) {
    const ch = itos[t];
    if (ch && PENALTY_EXEMPT.has(ch)) continue;
    if (last[t] > 0) last[t] /= CFG.repPenalty; else last[t] *= CFG.repPenalty;
  }
  if (exclude) for (const t of exclude) last[t] = -Infinity;
  if (endBias) {
    // 文末完結バイアス：句点・終端を少しだけ押し上げ、読点を少しだけ下げる
    for (const ch of ['。', '！', '？']) { const t = stoi.get(ch); if (t !== undefined) last[t] += 0.8; }
    last[EOS] += 0.6;
    const comma = stoi.get('、'); if (comma !== undefined) last[comma] -= 0.6;
  }
  for (let i = 0; i < V; i++) last[i] /= CFG.temperature;
  if (!sortBuf || sortBuf.length !== V) sortBuf = new Float32Array(V);
  sortBuf.set(last); sortBuf.sort();
  const thresh = sortBuf[V - Math.min(CFG.topK, V)];
  // Min-p: 最大確率トークンの minP 倍未満の確率しかない候補を足切りする
  // （温度適用後の logit 空間では l_i >= l_max + ln(minP) と等価）
  let maxL = -Infinity;
  for (let i = 0; i < V; i++) if (last[i] > maxL) maxL = last[i];
  const minPThresh = (CFG.minP && CFG.minP > 0) ? maxL + Math.log(CFG.minP) : -Infinity;
  const floor = Math.max(thresh, minPThresh);
  let sum = 0; const cand = [];
  for (let i = 0; i < V; i++) if (last[i] >= floor && isFinite(last[i])) { last[i] = Math.exp(last[i] - floor); sum += last[i]; cand.push(i); }
  if (!cand.length) { // 足切りし過ぎた場合は、マスクされていない最良トークンを返す
    let bestLogit = -Infinity, bestIdx = -1;
    const raw = Float32Array.from(logits);
    for (let i = 0; i < V; i++) { if (exclude && exclude.has(i)) continue; if (raw[i] > bestLogit) { bestLogit = raw[i]; bestIdx = i; } }
    return bestIdx >= 0 ? bestIdx : 0;
  }
  let r = rand() * sum;
  for (const i of cand) { r -= last[i]; if (r <= 0) return i; }
  return cand[cand.length - 1];
}

const emptyTensor = () => new ort.Tensor('float32', new Float32Array(0), [1, NH, 0, HD]);

// past テンソルは [1, NH, L, HD] 並び（ヘッド優先）なので、接頭辞の切り出しはヘッドごとに行う
function slicePastData(data, L, prefix) {
  const out = new Float32Array(prefix * NH * HD);
  for (let h = 0; h < NH; h++) out.set(data.subarray(h * L * HD, (h * L + prefix) * HD), h * prefix * HD);
  return out;
}

async function runKV(inputIds, offset, past) {
  const feeds = {
    idx: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]),
    offset: new ort.Tensor('int64', BigInt64Array.from([BigInt(offset)]), []),
  };
  for (let i = 0; i < NL; i++) { feeds['past_k_' + i] = past[i].k; feeds['past_v_' + i] = past[i].v; }
  const out = await session.run(feeds);
  const newPast = [];
  for (let i = 0; i < NL; i++) newPast.push({ k: out['present_k_' + i], v: out['present_v_' + i] });
  return { past: newPast, logits: out.logits.data };
}

// KV版生成（Prefix Cache 再利用つき）
async function genKV(ids, rand) {
  const V = itos.length;
  let prefix = 0;
  if (cacheIds && cachePast) {
    const n = Math.min(cacheIds.length, ids.length - 1);
    while (prefix < n && cacheIds[prefix] === ids[prefix]) prefix++;
    if (prefix < 3) prefix = 0;
  }
  let past = [], pastLen = prefix;
  if (prefix > 0) {
    const L0 = cacheIds.length;
    for (let i = 0; i < NL; i++) past.push({
      k: new ort.Tensor('float32', slicePastData(cachePast[i].k, L0, prefix), [1, NH, prefix, HD]),
      v: new ort.Tensor('float32', slicePastData(cachePast[i].v, L0, prefix), [1, NH, prefix, HD]),
    });
  } else {
    for (let i = 0; i < NL; i++) past.push({ k: emptyTensor(), v: emptyTensor() });
  }
  const reused = prefix;
  // 差分をプリフィル
  const pre = ids.slice(prefix);
  let r = await runKV(pre, pastLen, past);
  past = r.past; pastLen += pre.length;
  let lastLogits = r.logits.subarray((pre.length - 1) * V, pre.length * V);

  const allIds = ids.slice();
  const genStart = allIds.length; // ここから先が生成分（文字単位なので生成トークン数＝生成文字数）
  let outText = '', genCount = 0;
  // v5.11: NG表現の巻き戻し用に、各生成位置の logits を保持する
  const logitsHist = [];
  const excludeMap = new Map(); // 生成位置 -> 巻き戻しでマスクするトークン集合（累積）
  let rollbacks = 0;
  for (let step = 0; step < CFG.maxNew * 3 && genCount < CFG.maxNew; step++) {
    if (cancelFlag) break;
    logitsHist[genCount] = Float32Array.from(lastLogits);
    const endBias = genCount >= Math.floor(CFG.maxNew * 0.8) && !looksCode(outText);
    const nxt = sampleNext(lastLogits, allIds, rand, excludeMap.get(genCount), endBias);
    if (nxt === EOS) break;
    if (!itos[nxt]) break;
    allIds.push(nxt); genCount++;
    outText += itos[nxt];
    // NG表現が出たら、開始位置まで KV を巻き戻して開始トークンをマスクし、別表現へ分岐する
    const hit = ngIndex(outText);
    if (hit >= 0) {
      if (rollbacks >= 3) { outText = outText.slice(0, hit); break; }
      rollbacks++;
      const bad = allIds[genStart + hit];
      if (!excludeMap.has(hit)) excludeMap.set(hit, new Set());
      excludeMap.get(hit).add(bad);
      allIds.length = genStart + hit;
      outText = outText.slice(0, hit);
      genCount = hit;
      const keepLen = genStart + hit;
      if (keepLen < pastLen) {
        const sliced = [];
        for (let i = 0; i < NL; i++) sliced.push({
          k: new ort.Tensor('float32', slicePastData(past[i].k.data, pastLen, keepLen), [1, NH, keepLen, HD]),
          v: new ort.Tensor('float32', slicePastData(past[i].v.data, pastLen, keepLen), [1, NH, keepLen, HD]),
        });
        past = sliced; pastLen = keepLen;
      }
      lastLogits = logitsHist[hit];
      continue;
    }
    if (outText.includes('\nユーザー：') || outText.includes('\nAI：')) {
      outText = outText.split('\nユーザー：')[0].split('\nAI：')[0];
      break;
    }
    if (step % 2 === 0 || outText.length < 12) postMessage({ type: 'partial', text: outText });
    if (outText.length >= 600) break;
    if (genCount >= CFG.maxNew || pastLen >= effSeq()) break;
    r = await runKV([nxt], pastLen, past);
    past = r.past; pastLen += 1;
    lastLogits = r.logits.subarray(0, V);
  }
  // past がカバーする範囲だけをキャッシュに残す（次ターンの接頭辞として再利用）
  if (pastLen >= 3 && pastLen <= effSeq()) {
    cacheIds = allIds.slice(0, pastLen);
    cachePast = [];
    for (let i = 0; i < NL; i++) cachePast.push({ k: past[i].k.data, v: past[i].v.data });
  }
  return { text: outText, genCount, ctxLen: Math.min(allIds.length, effSeq()), reused };
}

// 従来版生成（全系列を毎回計算）
async function genFull(ids, rand) {
  const V = itos.length;
  const genStart = ids.length;
  let outText = '', genCount = 0;
  const excludeMap = new Map();
  let rollbacks = 0;
  for (let step = 0; step < CFG.maxNew * 3 && genCount < CFG.maxNew; step++) {
    if (cancelFlag) break;
    const ctx = ids.slice(-effSeq());
    const feeds = { idx: new ort.Tensor('int64', BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]) };
    const out = await session.run(feeds);
    const logits = out.logits.data;
    const endBiasF = genCount >= Math.floor(CFG.maxNew * 0.8) && !looksCode(outText);
    const nxt = sampleNext(logits.subarray((ctx.length - 1) * V, ctx.length * V), ids, rand, excludeMap.get(genCount), endBiasF);
    if (nxt === EOS) break;
    if (!itos[nxt]) break;
    ids.push(nxt); genCount++;
    outText += itos[nxt];
    // NG表現は開始位置まで巻き戻して開始トークンをマスクし、別表現へ分岐（全文再生成はしない）
    const hit = ngIndex(outText);
    if (hit >= 0) {
      if (rollbacks >= 3) { outText = outText.slice(0, hit); break; }
      rollbacks++;
      const bad = ids[genStart + hit];
      if (!excludeMap.has(hit)) excludeMap.set(hit, new Set());
      excludeMap.get(hit).add(bad);
      ids.length = genStart + hit;
      outText = outText.slice(0, hit);
      genCount = hit;
      continue;
    }
    if (outText.includes('\nユーザー：') || outText.includes('\nAI：')) {
      outText = outText.split('\nユーザー：')[0].split('\nAI：')[0];
      break;
    }
    if (step % 3 === 0 || outText.length < 12) postMessage({ type: 'partial', text: outText });
    if (outText.length >= 600) break;
  }
  cacheIds = null; cachePast = null;
  return { text: outText, genCount, ctxLen: Math.min(ids.length, effSeq()), reused: 0 };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'config') { if (m.cfg) CFG = { ...CFG, ...m.cfg }; return; }
    if (m.type === 'reset') { cacheIds = null; cachePast = null; return; }
    if (m.type === 'stop') { cancelFlag = true; return; }
    if (m.type === 'init') {
      if (m.cfg) CFG = { ...CFG, ...m.cfg };
      postMessage({ type: 'init-progress', phase: 'ort' });
      importScripts(m.ortBase + 'ort.min.js');
      ort.env.wasm.wasmPaths = m.ortBase;
      ort.env.wasm.numThreads = CFG.threads;
      itos = m.itos;
      stoi = new Map(); itos.forEach((ch, i) => { if (ch) stoi.set(ch, i); });
      const buf = new Uint8Array(m.modelBytes);
      let usedEP = 'wasm';
      // WebGPU は明示選択時のみ。自動では使わない（一部環境で作成が終わらず固まるため）。
      // 明示時も25秒で見切って WASM にフォールバックする。
      const wantGPU = CFG.ep === 'webgpu' && typeof navigator !== 'undefined' && 'gpu' in navigator;
      session = null;
      const memOpts = CFG.lowMem ? { enableCpuMemArena: false, enableMemPattern: false } : {};
      postMessage({ type: 'init-progress', phase: 'session' });
      if (wantGPU) {
        try {
          session = await Promise.race([
            ort.InferenceSession.create(buf, { executionProviders: ['webgpu'], ...memOpts }),
            new Promise((_, rej) => setTimeout(() => rej(new Error('webgpu timeout')), 25000)),
          ]);
          usedEP = 'webgpu';
        } catch (gpuErr) { session = null; }
      }
      if (!session) session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], ...memOpts });
      postMessage({ type: 'init-progress', phase: 'session-done' });
      kvMode = session.inputNames.includes('offset');
      cacheIds = null; cachePast = null;
      postMessage({ type: 'ready', kv: kvMode, ep: usedEP, isolated: (typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false) });
      return;
    }
    if (m.type === 'generate') {
      cancelFlag = false;
      const t0 = performance.now();
      const ids = buildIds(m.history);
      const rand = mulberry32((Math.random() * 1e9) | 0);
      const r = kvMode ? await genKV(ids, rand) : await genFull(ids, rand);
      const ms = performance.now() - t0;
      postMessage({ type: 'done', text: r.text, stopped: cancelFlag, stats: {
        tps: r.genCount > 0 ? +(r.genCount / (ms / 1000)).toFixed(1) : 0,
        gen: r.genCount, ctx: r.ctxLen, ctxMax: effSeq(), reused: r.reused } });
      return;
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
