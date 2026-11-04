const fs = require('fs');
const path = require('path');
const HTML = path.join(__dirname, '..', '模型监看.html');
const src = [...fs.readFileSync(HTML, 'utf8').matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];

/* ---------- 极简 DOM 桩 ---------- */
class CL {
  constructor() { this.s = new Set(); }
  add(...c) { c.forEach(x => x && this.s.add(x)); }
  remove(...c) { c.forEach(x => this.s.delete(x)); }
  toggle(c, f) { if (f === undefined) f = !this.s.has(c); f ? this.s.add(c) : this.s.delete(c); return f; }
  contains(c) { return this.s.has(c); }
}
class El {
  constructor(id) {
    this.id = id || ''; this._h = {}; this.children = []; this.classList = new CL(); this.style = {};
    this.dataset = {}; this._t = ''; this._html = ''; this._v = ''; this.hidden = false;
    this.disabled = false; this.checked = false; this.parentNode = null;
  }
  get value() { if (this._v === '' && this.children.length && this.children[0].value !== undefined) return this.children[0].value; return this._v; }
  set value(v) { this._v = String(v); }
  get textContent() { return this._t; } set textContent(v) { this._t = String(v); }
  get innerHTML() { return this._html; } set innerHTML(v) { this._html = String(v); this.children = []; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  addEventListener(t, f) { (this._h[t] = this._h[t] || []).push(f); }
  dispatch(t, ev) {
    const e = Object.assign({ type: t, target: this, preventDefault() {}, stopPropagation() {} }, ev || {});
    (this._h[t] || []).forEach(f => f.call(this, e));
  }
  closest(sel) { if (!sel.startsWith('[data-')) return null; const k = sel.slice(6, -1); return this.dataset && this.dataset[k] !== undefined ? this : null; }
  contains(n) { if (n === this) return true; for (const c of this.children) { if (c === n || c.contains(n)) return true; } return false; }
  querySelectorAll() { return []; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 450, right: 800, bottom: 450 }; }
  setPointerCapture() {} releasePointerCapture() {} focus() {} select() {} remove() {}
}
const els = {};
const get = id => els[id] || (els[id] = new El(id));
const docH = {};
global.document = {
  getElementById: get, createElement: () => new El(), body: new El('body'),
  addEventListener(t, f) { (docH[t] = docH[t] || []).push(f); },
  execCommand() { return true; }
};
global.window = { addEventListener() {} };
try {
  Object.defineProperty(global, 'navigator', {
    value: { clipboard: { writeText: t => { global.__clip = t; return Promise.resolve(); } } },
    configurable: true, writable: true
  });
} catch (e) { global.navigator.clipboard = { writeText: t => { global.__clip = t; return Promise.resolve(); } }; }

const store = new Map();
global.localStorage = { getItem: k => store.has(k) ? store.get(k) : null, setItem: (k, v) => store.set(k, String(v)) };

let calls = [];
const resp = obj => Promise.resolve({ ok: true, json: () => Promise.resolve(obj), blob: () => Promise.resolve(new Blob([new Uint8Array([255, 216, 255, 217])])) });
global.fetch = function (url, opt) {
  calls.push({ url: String(url), opt: opt || {} });
  const u = String(url);
  if (u.includes('/health')) return resp({ model_ready: true, device: 'NVIDIA GeForce RTX 4050 Laptop GPU' });
  if (u.includes('/videos')) return resp({ ok: true, items: [{ name: 'a.mp4', path: String.raw`D:\ra3\a.mp4`, size: 9e8 }] });
  if (u.includes('/info')) return resp({ ok: true, name: 'a.mp4', duration: 887.88, fps: 30, width: 1920, height: 1080 });
  if (u.includes('/still')) return resp({});
  if (u.includes('/jobs?') || u.endsWith('/jobs')) return resp({ id: 'job1' });
  if (u.includes('/live')) return resp(LIVE);
  if (u.includes('/frame')) return resp({});
  return resp({});
};
global.__imgs = [];
global.Image = class { constructor() { this._s = ''; } set src(v) { this._s = v; global.__imgs.push(v); setTimeout(() => this.onload && this.onload(), 0); } get src() { return this._s; } };

const LIVE = {
  id: 'job1', seg: 1,
  counts: { '其它小型客车': 9, '其它大型客车': 0, '公交车': 1, '出租车': 1, '小型货车': 0, '大型货车': 0, '摩托车': 4 },
  totals: { '其它小型客车': 11, '其它大型客车': 2, '公交车': 1, '出租车': 3, '小型货车': 0, '大型货车': 0, '摩托车': 5 },
  total: 22, progress: 0.02, t: 18.5, duration: 887.88, start: 10, clip: 30,
  log: '120 帧已处理 15.2 帧/s 视频位置 00:18 第 1 段', ver: 7, running: true, done: false, error: '', elapsed: 4.2,
  events: [{ t: 11.2, cat: '其它小型客车', n: 1 }, { t: 15.8, cat: '摩托车', n: 2 }]
};

eval(src);

let pass = 0, fail = 0;
const ok = (n, c, d) => { c ? pass++ : fail++; console.log((c ? 'PASS ' : 'FAIL ') + n + (c ? '' : '  → ' + d)); };
const wait = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  await wait(80);
  ok('健康检查通过', /已连接/.test(els.srvText.textContent), els.srvText.textContent);
  ok('扫描后自动选中视频', els.fileSel.children.length === 1, els.fileSel.children.length);
  ok('显示了片长信息', /14:47/.test(els.durInfo.textContent), els.durInfo.textContent);
  ok('取了静帧（走 fetch）', calls.some(c => c.url.includes('/still')), 'no still');

  /* 框选：点「框选」后在画面上拖一个框 */
  els.btnRoiDraw.dispatch('click', {});
  ok('进入框选模式', els.stage.classList.contains('drawing') === true, els.stage.classList);
  els.stage.dispatch('pointerdown', { clientX: 100, clientY: 100, pointerId: 1, target: els.stage });
  els.stage.dispatch('pointermove', { clientX: 500, clientY: 300, pointerId: 1, target: els.stage });
  els.stage.dispatch('pointerup', { clientX: 500, clientY: 300, pointerId: 1, target: els.stage });
  ok('框出来了（roi 可见）', els.roi.hidden === false, els.roi.hidden);
  ok('框位置 12.5% / 22.2%', els.roi.style.left === '12.5%' && /^22\.2/.test(els.roi.style.top), els.roi.style.left + ' ' + els.roi.style.top);
  ok('框大小 50% / 44.4%', els.roi.style.width === '50%' && /^44\.4/.test(els.roi.style.height), els.roi.style.width + ' ' + els.roi.style.height);
  ok('拖完自动退出框选', els.stage.classList.contains('drawing') === false, els.stage.classList);
  ok('区域勾选框自动打开', els.chkRoi.checked === true, els.chkRoi.checked);

  els.lineRange.value = '45';
  els.lineRange.dispatch('input', { target: els.lineRange });
  ok('滑条改线生效', els.lineVal.textContent === '45%', els.lineVal.textContent);

  els.chkRoi.checked = true;
  els.chkRoi.dispatch('change', { target: els.chkRoi });
  els.startInput.value = '00:10';
  els.startInput.dispatch('change', { target: els.startInput });
  els.btnStart.dispatch('click', {});
  await wait(40);

  const post = calls.find(c => c.url.includes('/jobs?'));
  ok('发起了任务', !!post, 'no post');
  ok('带上了本机路径', post && post.url.includes('src=D%3A%5Cra3%5Ca.mp4'), post && post.url.slice(0, 90));
  ok('带上了识别线 45%', post && /line=0\.450/.test(post.url), post && post.url);
  ok('带上了起点与时长', post && /start=10/.test(post.url) && /clip=30/.test(post.url), post && post.url);
  ok('带上了识别区域', post && /roi=/.test(post.url), post && post.url);

  await wait(900);
  const rows = els.statBody.innerHTML;
  ok('表格有 7 类', (rows.match(/<tr/g) || []).length === 8, (rows.match(/<tr/g) || []).length);
  ok('本段小客车 9、累计 11', rows.indexOf('其它小型客车</td><td class="n">9</td><td class="n">11') > -1, rows.slice(0, 160));
  ok('合计本段 15 / 累计 22', rows.indexOf('合计</td><td class="n">15</td><td class="n">22') > -1, rows.slice(-120));
  ok('进度条有宽度', parseFloat(els.prog.style.width) > 20, els.prog.style.width);
  ok('状态显示视频位置', /00:18/.test(els.status.textContent), els.status.textContent);
  ok('过线明细 2 条', els.evCount.textContent === '2', els.evCount.textContent);
  ok('明细里有摩托车', /摩托车/.test(els.evs.innerHTML), els.evs.innerHTML);

  ok('画面已经显示（empty 隐藏）', els.empty.style.display === 'none', els.empty.style.display);
  ok('取过实时帧', calls.some(c => c.url.includes('/frame')), 'no frame');
  ok('img 拿到了 blob 地址', global.__imgs.some(u => u.indexOf('blob:') === 0), JSON.stringify(global.__imgs.slice(0, 3)));

  els.btnCopy.dispatch('click', {});
  await wait(20);
  ok('复制累计 7 个数字', global.__clip === '11\t2\t1\t3\t0\t0\t5', JSON.stringify(global.__clip));
  els.btnCopySeg.dispatch('click', {});
  await wait(20);
  ok('复制本段 7 个数字', global.__clip === '9\t0\t1\t1\t0\t0\t4', JSON.stringify(global.__clip));

  els.btnSave.dispatch('click', {});
  const mon = JSON.parse(store.get('vseg.mon.settings') || 'null');
  ok('设置写进自己的键', !!mon && Math.abs(mon.line - 45) < 1e-6, JSON.stringify(mon && mon.line));
  ok('设置里有区域', !!mon && !!mon.roi && mon.roiOn === true, JSON.stringify(mon && mon.roi));
  ok('没碰主页面的 vseg.settings', !store.has('vseg.settings'), [...store.keys()].join(','));

  const before = els.startInput.value;
  docH.keydown.forEach(f => f({ key: 'ArrowRight', code: 'ArrowRight', target: els.body, preventDefault() {} }));
  await wait(20);
  ok('→ 键把起点推到 00:15', els.startInput.value === '00:15' && before === '00:10', before + ' → ' + els.startInput.value);

  els.btnReset.dispatch('click', {});
  ok('复位回到 65%', els.lineVal.textContent === '65%', els.lineVal.textContent);

  console.log(`\n${pass} 通过 / ${fail} 失败`);
  process.exit(fail ? 1 : 0);
})();
