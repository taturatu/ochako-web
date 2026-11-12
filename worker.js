// おちゃこAI 生成ワーカー（UIスレッドを止めないため推論はここで実行）
let session = null, itos = [];
let stoi = new Map();

const TEMPERATURE = 0.65, TOP_K = 40, REP_PENALTY = 1.15, MAX_NEW = 150, MAX_SEQ = 512;
const BOS = 1, EOS = 2;

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      importScripts(m.ortBase + 'ort.min.js');
      ort.env.wasm.wasmPaths = m.ortBase;
      ort.env.wasm.numThreads = Math.min(4, self.navigator.hardwareConcurrency || 1);
      itos = m.itos;
      itos.forEach((c, i) => { if (c !== null && c !== undefined) stoi.set(c, i); });
      session = await ort.InferenceSession.create(new Uint8Array(m.modelBytes), { executionProviders: ['wasm'] });
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'generate') {
      const out = await generate(m.text);
      self.postMessage({ type: 'done', id: m.id, text: out });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id || null, message: String(err && err.message || err) });
  }
};

async function generate(promptText) {
  const rand = mulberry32((Math.random() * 2 ** 31) | 0);
  let ids = [BOS];
  for (const ch of promptText) if (stoi.has(ch)) ids.push(stoi.get(ch));
  const outIds = [];
  for (let step = 0; step < MAX_NEW; step++) {
    const ctx = ids.slice(-MAX_SEQ);
    const input = new ort.Tensor('int64', BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]);
    const res = await session.run({ idx: input });
    const logits = res.logits.data; // [1, T, V]
    const V = itos.length, off = (ctx.length - 1) * V;
    let last = Array.from(logits.slice(off, off + V));
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
    if (nxt === EOS) break;
    outIds.push(nxt); ids.push(nxt);
    const text = outIds.map(i => itos[i] ?? '').join('');
    if (text.includes('\nユーザー') || text.includes('\nあなた：')) break;
    self.postMessage({ type: 'token', text });
  }
  let text = outIds.map(i => itos[i] ?? '').join('');
  for (const mk of ['\nユーザー', '\nあなた：']) { const p = text.indexOf(mk); if (p >= 0) text = text.slice(0, p); }
  return text.trim();
}
