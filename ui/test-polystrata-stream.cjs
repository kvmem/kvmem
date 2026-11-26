// Exercise the shared send/regenerate stream against real SSE fixtures, without a model.
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const assert = require('node:assert/strict');
const app = fs.readFileSync(path.join(__dirname, '../backends/polystrata/terminal/serve/web/kvmem-app.js'), 'utf8');
const stream = app.slice(app.indexOf('async function streamInto('), app.indexOf('// Regenerates the answer'));
const frame = value => `data: ${typeof value === 'string' ? value : JSON.stringify(value)}\n\n`;
function harness(chunks, overrides = {}) {
  let request;
  const context = vm.createContext({
    TextDecoder, performance, Number, Error,
    health: {model: 'test'}, chatOverrides: overrides,
    settingKeys: {temperature: 'temperature', max: 'max_tokens', thinking: 'reasoning_effort'},
    settings: {mcp: false}, mcpInfo: {tools: 0},
    apiMessages: () => [{role: 'user', content: 'hi'}], headers: () => ({}),
    projectionLoaded: () => false, updateAssistant: () => {}, scrollDown: () => {},
    requestAnimationFrame: () => 1, cancelAnimationFrame: () => {},
    fmt: (n, digits = 0) => n.toFixed(digits), onTool: () => {},
    fetch: async (_, options) => {
      request = JSON.parse(options.body);
      let i = 0;
      return {ok: true, body: {getReader: () => ({read: async () => {
        const value = chunks[i++];
        if (value instanceof Error) throw value;
        return value === undefined ? {done: true} : {done: false, value: Buffer.from(value)};
      }})}};
    },
  });
  vm.runInContext(stream, context);
  return {run: m => context.streamInto(m, {}, null), request: () => request};
}
const message = () => ({text: '', reasoning: ''});
(async () => {
  const events = frame({choices: [{delta: {reasoning_content: 'reason'}}]}) +
    frame({choices: [{delta: {content: 'answer'}, finish_reason: 'stop'}], usage: {completion_tokens: 4}, timings: {predicted_per_second: 12.5}}) + frame('[DONE]');
  const h = harness([events.slice(0, 13), events.slice(13)]), m = message();
  await h.run(m);
  assert.equal(m.text, 'answer'); assert.equal(m.reasoning, 'reason');
  assert.equal(m.streamDone, true); assert.equal(m.finishReason, 'stop');
  assert.match(m.meta, /12.5 tok\/s/); assert.match(m.meta, /正常结束/);
  assert.deepEqual(Object.keys(h.request()).sort(), ['messages', 'model', 'stream']);

  const explicit = harness([frame('[DONE]')], {temperature: 0, max: 17, thinking: 'none', preserve: false});
  await explicit.run(message());
  assert.equal(explicit.request().temperature, 0); assert.equal(explicit.request().max_tokens, 17);
  assert.equal(explicit.request().reasoning_effort, 'none');
  assert.equal(explicit.request().chat_template_kwargs.preserve_thinking, false);

  // A regenerated answer must not inherit a preceding answer's DONE/finish/timing flags.
  const stale = {...message(), streamDone: true, finishReason: 'stop', timings: {predicted_per_second: 99}};
  await assert.rejects(harness([frame({choices: [{delta: {content: 'partial'}}]})]).run(stale), /未收到完成标记/);
  assert.equal(stale.streamDone, false); assert.equal(stale.finishReason, null); assert.equal(stale.timings, null);

  const limited = message();
  await harness([frame({choices: [{delta: {reasoning_content: 'thinking'}, finish_reason: 'length'}]})]).run(limited);
  assert.match(limited.meta, /思考阶段达到输出上限/);
  const stopped = message(), abort = Object.assign(new Error('cancelled'), {name: 'AbortError'});
  await harness([abort]).run(stopped); assert.equal(stopped.stopped, true); assert.match(stopped.meta, /已停止/);
  await assert.rejects(harness([frame({error: {message: 'engine failed'}})]).run(message()), /engine failed/);
  console.log('PASS: poly.4 shared streaming preserves overrides, reasoning, native timing, completion, abort and retry boundaries');
})().catch(error => { console.error(error); process.exitCode = 1; });
