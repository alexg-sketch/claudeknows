// Runs the claudeknows SmallCNN in plain JavaScript (no libraries).
// Batch-norm is folded into the conv weights by claudeknows/export_web.py,
// so the network here is just: 3 x (conv-relu-conv-relu-maxpool) -> linear.

function decodeWeights(base64) {
  let bytes;
  if (typeof atob === "function") {
    const bin = atob(base64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } else {
    bytes = new Uint8Array(Buffer.from(base64, "base64"));
  }
  return new Float32Array(bytes.buffer);
}

function buildModel(meta, flat) {
  let pos = 0;
  const take = (n) => {
    const out = flat.subarray(pos, pos + n);
    pos += n;
    return out;
  };
  const convs = [];
  const ch = meta.channels;
  for (let i = 0; i + 1 < ch.length; i++) {
    const cin = ch[i], cout = ch[i + 1];
    convs.push({ cin, cout, w: take(cout * cin * 9), b: take(cout) });
  }
  const fc = { w: take(meta.numClasses * meta.fcIn), b: take(meta.numClasses) };
  if (pos !== flat.length) throw new Error("weight size mismatch");
  return { meta, convs, fc };
}

// 3x3 convolution, padding 1, stride 1. Input/output are CHW.
function conv3x3(input, H, W, layer) {
  const { cin, cout, w, b } = layer;
  const HW = H * W;
  const out = new Float32Array(cout * HW);
  for (let o = 0; o < cout; o++) {
    const oBase = o * HW;
    out.fill(b[o], oBase, oBase + HW);
    for (let c = 0; c < cin; c++) {
      const iBase = c * HW;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const wv = w[((o * cin + c) * 3 + ky) * 3 + kx];
          const x0 = Math.max(0, 1 - kx), x1 = Math.min(W, W + 1 - kx);
          for (let y = 0; y < H; y++) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= H) continue;
            const oRow = oBase + y * W, iRow = iBase + iy * W + kx - 1;
            for (let x = x0; x < x1; x++) out[oRow + x] += wv * input[iRow + x];
          }
        }
      }
    }
  }
  return out;
}

function relu(t) {
  for (let i = 0; i < t.length; i++) if (t[i] < 0) t[i] = 0;
  return t;
}

function maxPool2(input, C, H, W) {
  const h = H >> 1, w = W >> 1;
  const out = new Float32Array(C * h * w);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = c * H * W + 2 * y * W + 2 * x;
        out[(c * h + y) * w + x] = Math.max(input[i], input[i + 1], input[i + W], input[i + W + 1]);
      }
    }
  }
  return out;
}

// input: Float32Array of length 3*32*32 (CHW, already normalised). Returns logits.
function forward(model, input) {
  let t = input, H = 32, W = 32;
  model.convs.forEach((layer, i) => {
    t = relu(conv3x3(t, H, W, layer));
    if (i % 2 === 1) {
      t = maxPool2(t, layer.cout, H, W);
      H >>= 1;
      W >>= 1;
    }
  });
  const { w, b } = model.fc, n = model.meta.fcIn;
  const logits = new Float32Array(b.length);
  for (let k = 0; k < b.length; k++) {
    let s = b[k];
    for (let j = 0; j < n; j++) s += w[k * n + j] * t[j];
    logits[k] = s;
  }
  return logits;
}

function softmax(logits) {
  const m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const s = e.reduce((a, v) => a + v, 0);
  return e.map((v) => v / s);
}

// RGBA pixels of a 32x32 image -> normalised CHW Float32Array.
function pixelsToInput(rgba, mean, std) {
  const out = new Float32Array(3 * 32 * 32);
  for (let p = 0; p < 1024; p++) {
    for (let c = 0; c < 3; c++) out[c * 1024 + p] = (rgba[p * 4 + c] / 255 - mean[c]) / std[c];
  }
  return out;
}

if (typeof module !== "undefined") {
  module.exports = { decodeWeights, buildModel, forward, softmax, pixelsToInput };
}
