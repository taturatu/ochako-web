// おちゃこAI 生成ワーカー（UIスレッドを止めないため推論はここで実行）
// プロンプト形式は製品版アプリと同一：
//   ユーザー：{発話}\nAI：{返事}\n…\nAI：
// 停止マーカーは「\nユーザー：」「\nAI：」。
// モデルがKVキャッシュ版（入力に offset / past_* を持つ）の場合は高速モード、
// そうでなければ従来の全系列モードで動く（自動判定）。
let session = null, itos = [], kvMode = false;
let stoi = new Map();

const TEMPERATURE = 0.65, TOP_K = 40, REP_PENALTY = 1.15, MAX_NEW = 150, MAX_SEQ = 512;
const BOS = 1, EOS = 2;
const NL = 12, NH = 10, HD = 64;

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

function clean(text) {
  for (const mk of ['\nユーザー：', '\nAI：']) { const p = text.indexOf(mk); if (p >= 0) text = text.slice(0, p); }
  return text.replace(/\n+$/, '').trim();
}

function sampleNext(logitsArr, ids, rand) {
  const V = itos.length;
  let last = Array.from(logitsArr);
  const seen = new Set(ids.slice(1));
  for (const t of seen) { if (last[t] > 0) last[t] /= REP_PENALTY; else last[t] *= REP_PENALTY; }
  for (let i = 0; i < V; i++) last[i] /= TEMPERATURE;
  const sorted = [...last].sort((a, b) => b - a);
  const thresh = sorted[Math.min(TOP_K, V) - 1];
  let max = -Infinity; for (let i = 0; i < V; i++) if (last[i] >= thresh && last[i] > max) max = last[i];
  let sum = 0; const probs = new Float64Array(V);
  for (let i = 0; i < V; i++) { if (last[i] >= thresh) { probs[i] = Math.exp(last[i] - max); sum += probs[i]; } }
  let r = rand() * sum, nxt = EOS;
  for (let i = 0; i < V; i++) { if (probs[i] > 0) { r -= probs[i]; if (r <= 0) { nxt = i; break; } } }
  return nxt;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      importScripts(m.ortBase + 'ort.min.js');
      ort.env.wasm.wasmPaths = m.ortBase;
      itos = m.itos;
      itos.forEach((c, i) => { if (c !== null && c !== undefined) stoi.set(c, i); });
      // 実測では単一スレッドが最速だった（共有CPUではスレッド追加が逆効果）。
      // また SharedArrayBuffer が無い環境での不安定さも避けられる。
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(new Uint8Array(m.modelBytes), { executionProviders: ['wasm'] });
      kvMode = session.inputNames.includes('offset');
      self.postMessage({ type: 'ready', kv: kvMode });
    } else if (m.type === 'generate') {
      const out = kvMode ? await generateKV(m.history) : await generateFull(m.history);
      self.postMessage({ type: 'done', id: m.id, text: out });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id || null, message: String(err && err.message || err) });
  }
};

function promptIds(history) {
  const promptText = buildPrompt(history);
  let ids = [BOS];
  for (const ch of promptText) if (stoi.has(ch)) ids.push(stoi.get(ch));
  return ids.slice(-MAX_SEQ);
}

async function generateFull(history) {
  const rand = mulberry32((Math.random() * 2 ** 31) | 0);
  let ids = promptIds(history);
  const outIds = [];
  for (let step = 0; step < MAX_NEW; step++) {
    const ctx = ids.slice(-MAX_SEQ);
    const input = new ort.Tensor('int64', BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]);
    const res = await session.run({ idx: input });
    const logits = res.logits.data;
    const V = itos.length, off = (ctx.length - 1) * V;
    const nxt = sampleNext(logits.slice(off, off + V), ids, rand);
    if (nxt === EOS) break;
    if (itos[nxt] === null || itos[nxt] === undefined) break;
    outIds.push(nxt); ids.push(nxt);
    const text = outIds.map(i => itos[i] ?? '').join('');
    if (text.includes('\nユーザー：') || text.includes('\nAI：')) break;
    self.postMessage({ type: 'token', text: clean(text) });
  }
  return clean(outIds.map(i => itos[i] ?? '').join(''));
}

async function generateKV(history) {
  const rand = mulberry32((Math.random() * 2 ** 31) | 0);
  let ids = promptIds(history);
  const outIds = [];
  const V = itos.length;
  // prefill
  let pasts = {};
  for (let i = 0; i < NL; i++) {
    pasts[`past_k_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, NH, 0, HD]);
    pasts[`past_v_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, NH, 0, HD]);
  }
  const feed = {
    idx: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    offset: new ort.Tensor('int64', BigInt64Array.from([0n]), []),
    ...pasts,
  };
  let res = await session.run(feed);
  let pastLen = ids.length;
  let lastLogits = res.logits.data.slice((ids.length - 1) * V, ids.length * V);
  pasts = {};
  for (let i = 0; i < NL; i++) { pasts[`past_k_${i}`] = res[`present_k_${i}`]; pasts[`past_v_${i}`] = res[`present_v_${i}`]; }
  for (let step = 0; step < MAX_NEW; step++) {
    const nxt = sampleNext(lastLogits, ids, rand);
    if (nxt === EOS) break;
    if (itos[nxt] === null || itos[nxt] === undefined) break;
    outIds.push(nxt); ids.push(nxt);
    const text = outIds.map(i => itos[i] ?? '').join('');
    if (text.includes('\nユーザー：') || text.includes('\nAI：')) break;
    self.postMessage({ type: 'token', text: clean(text) });
    if (step === MAX_NEW - 1 || pastLen + 1 >= MAX_SEQ) break;
    const feed2 = {
      idx: new ort.Tensor('int64', BigInt64Array.from([BigInt(nxt)]), [1, 1]),
      offset: new ort.Tensor('int64', BigInt64Array.from([BigInt(pastLen)]), []),
      ...pasts,
    };
    res = await session.run(feed2);
    pastLen += 1;
    lastLogits = res.logits.data.slice(0, V);
    pasts = {};
    for (let i = 0; i < NL; i++) { pasts[`past_k_${i}`] = res[`present_k_${i}`]; pasts[`past_v_${i}`] = res[`present_v_${i}`]; }
  }
  return clean(outIds.map(i => itos[i] ?? '').join(''));
}
