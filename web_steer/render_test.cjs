// Run with: node --test web_steer/render_test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(`${__dirname}/app.js`, 'utf8');
const start = source.indexOf('const PROMPT_REFERENCE_HEIGHT');
const end = source.indexOf('function render()', start);
const scope = {};
vm.createContext(scope);
vm.runInContext(source.slice(start, end), scope);

function record(ops, width = 520, height = 520) {
  const events = [];
  const ctx = {
    save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {},
    scale(x, y) { events.push(['scale', x, y]); },
    arc(x, y, r) { events.push(['disk', x, y, r, this.fillStyle]); },
    fill() {}, stroke() { events.push(['stroke', this.strokeStyle, this.lineWidth]); },
  };
  scope.drawPromptOps(ctx, ops, width, height);
  return events;
}

test('target matches local orange radius 6 and white radius 8', () => {
  assert.deepEqual(record([{ type: 'point', point: { x: .5, y: .5 } }]), [
    ['scale', 1, 1], ['disk', 260, 260, 8, '#ffffff'], ['disk', 260, 260, 6, '#ff5000'],
  ]);
});

test('trajectory matches local segment-index gradient and thin line, without arrow', () => {
  const events = record([{ type: 'trajectory', points: [0, .25, .5, .75].map(x => ({ x, y: .5 })) }]);
  assert.deepEqual(events.filter(e => e[0] === 'stroke'), [
    ['stroke', 'rgb(255, 255, 255)', 2],
    ['stroke', 'rgb(255, 167, 127)', 2],
    ['stroke', 'rgb(255, 80, 0)', 2],
  ]);
});

test('phone preview and exported prompt use the same image-relative geometry', () => {
  const ops = [{ type: 'point', point: { x: .4, y: .6 } }];
  const preview = record(ops, 390, 260);
  const exported = record(ops, 336, 224);
  assert.deepEqual(preview.slice(1), exported.slice(1));
  assert.equal(preview[0][1], 260 / 520);
  assert.equal(exported[0][1], 224 / 520);
});

test('edge points stay within the reference image', () => {
  const events = record([{ type: 'point', point: { x: 1, y: 1 } }]);
  assert.deepEqual(events[1], ['disk', 519, 519, 8, '#ffffff']);
});

function promptSender(ops) {
  let resolveRequest;
  const button = { disabled: false };
  const scope = {
    state: { ops, sending: false },
    Set,
    $: () => button,
    apiFetch: () => new Promise(resolve => { resolveRequest = resolve; }),
    buildPayload: () => ({}),
    render() {},
    showToast() {},
    updateButtons() { button.disabled = false; },
  };
  vm.createContext(scope);
  const start = source.indexOf('async function sendPrompt()');
  const end = source.indexOf('function showToast(', start);
  vm.runInContext(source.slice(start, end), scope);
  return { scope, reply: value => resolveRequest(value) };
}

test('successful submission clears only sent marks, preserving edits made in flight', async () => {
  const sent = { type: 'point', point: { x: .2, y: .3 } };
  const added = { type: 'point', point: { x: .4, y: .5 } };
  const sender = promptSender([sent]);
  const pending = sender.scope.sendPrompt();
  sender.scope.state.ops.push(added);
  sender.reply({ ok: true, json: async () => ({ sequence: 1 }) });
  await pending;
  assert.equal(sender.scope.state.ops.length, 1);
  assert.equal(sender.scope.state.ops[0], added);
  assert.equal(sender.scope.state.sending, false);
});

test('failed submission retains sketch for retry', async () => {
  const mark = { type: 'point', point: { x: .2, y: .3 } };
  const sender = promptSender([mark]);
  const pending = sender.scope.sendPrompt();
  sender.reply({ ok: false, json: async () => ({ error: 'Rejected' }) });
  await pending;
  assert.equal(sender.scope.state.ops[0], mark);
  assert.equal(sender.scope.state.sending, false);
});
