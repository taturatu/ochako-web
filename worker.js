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
// デコード設定（標準は製品標準 B1.5。画面から変更できる）
let CFG = { temperature: 0.65, topK: 40, repPenalty: 1.15, maxNew: 150, threads: 1, ep: 'auto', maxSeq: 512, lowMem: false };
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

// サンプリング（penalty は出現トークンだけに適用、top-k 閾値は Float32Array のネイティブソートで取得）
let sortBuf = null;
function sampleNext(logits, ids, rand) {
  const V = itos.length;
  const last = Float32Array.from(logits);
  for (const t of new Set(ids)) { if (last[t] > 0) last[t] /= CFG.repPenalty; else last[t] *= CFG.repPenalty; }
  for (let i = 0; i < V; i++) last[i] /= CFG.temperature;
  if (!sortBuf || sortBuf.length !== V) sortBuf = new Float32Array(V);
  sortBuf.set(last); sortBuf.sort();
  const thresh = sortBuf[V - Math.min(CFG.topK, V)];
  let sum = 0; const cand = [];
  for (let i = 0; i < V; i++) if (last[i] >= thresh) { last[i] = Math.exp(last[i] - thresh); sum += last[i]; cand.push(i); }
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
  let outText = '', genCount = 0;
  for (let step = 0; step < CFG.maxNew; step++) {
    if (cancelFlag) break;
    const nxt = sampleNext(lastLogits, allIds, rand);
    if (nxt === EOS) break;
    if (!itos[nxt]) break;
    allIds.push(nxt); genCount++;
    outText += itos[nxt];
    if (outText.includes('\nユーザー：') || outText.includes('\nAI：')) {
      outText = outText.split('\nユーザー：')[0].split('\nAI：')[0];
      break;
    }
    if (step % 2 === 0 || outText.length < 12) postMessage({ type: 'partial', text: outText });
    if (outText.length >= 600) break;
    if (step === CFG.maxNew - 1 || pastLen >= effSeq()) break;
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
  let outText = '', genCount = 0;
  for (let step = 0; step < CFG.maxNew; step++) {
    if (cancelFlag) break;
    const ctx = ids.slice(-effSeq());
    const feeds = { idx: new ort.Tensor('int64', BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]) };
    const out = await session.run(feeds);
    const logits = out.logits.data;
    const nxt = sampleNext(logits.subarray((ctx.length - 1) * V, ctx.length * V), ids, rand);
    if (nxt === EOS) break;
    if (!itos[nxt]) break;
    ids.push(nxt); genCount++;
    outText += itos[nxt];
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
      kvMode = session.inputNames.includes('offset');
      cacheIds = null; cachePast = null;
      postMessage({ type: 'ready', kv: kvMode, ep: usedEP, isolated: (typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false) });
      return;
    }
    if (m.type === 'generate') {
      cancelFlag = false;
      const t0 = performance.now();
      const ids = encode(buildPrompt(m.history));
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
