const assert = require('node:assert/strict');
globalThis.FileReader = class {
  readAsDataURL(file) {
    file.arrayBuffer().then((data) => {
      this.result = `data:${file.type};base64,${Buffer.from(data).toString('base64')}`;
      this.onloadend?.();
    });
  }
};
(async () => {
  const { readAttachment, attachmentContent, supportedAttachment } = await import('./lightweight/attachments.ts');
  const source = '你好\nconst value = "<script>";\n';
  const file = await readAttachment(new File([source], 'demo.ts'), false);
  assert.equal(file.content, source);
  assert.equal(file.type, 'text');
  assert.equal(supportedAttachment(file), true);
  assert.equal(supportedAttachment({ type: 'text', name: 'bad', content: null }), false);
  const utf16 = Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from('中文', 'utf16le')]);
  assert.equal((await readAttachment(new File([utf16], 'bom.txt'), false)).content, '中文');
  await assert.rejects(() => readAttachment(new File([new Uint8Array([255])], 'bad.txt'), false), /UTF-8/);
  await assert.rejects(() => readAttachment(new File(['abc\0'], 'binary.txt'), false), /二进制/);
  await assert.rejects(() => readAttachment(new File(['%PDF'], 'doc.pdf'), false), /仅支持/);
  await assert.rejects(() => readAttachment(new File(['x'.repeat(1024 * 1024 + 1)], 'big.txt'), false), /1 MiB/);
  await assert.rejects(() => readAttachment(new File(['x'], 'photo.png'), false), /不支持图片/);
  await assert.rejects(() => readAttachment(new File([new Uint8Array(10 * 1024 * 1024 + 1)], 'photo.png'), true), /10 MiB/);
  await assert.rejects(() => readAttachment(new File(['x'], 'photo.webp', {type:'image/webp'}), true,
    {image_types:['image/png','image/jpeg']}), /PNG|JPEG/);
  assert.equal((await readAttachment(new File(['x'], 'photo.jpg', {type:'image/jpeg'}), true,
    {image_types:['image/png','image/jpeg']})).type, 'image');
  assert.equal(typeof attachmentContent('', [file]), 'string', 'text-only model receives ordinary text');
  assert.ok(attachmentContent('Question', [file]).includes(source), 'content is not truncated or executed');
  const image = { type: 'image', name: 'a.png', size: 1, base64Url: 'data:image/png;base64,AA==' };
  assert.deepEqual(attachmentContent('Question', [file, image]), [
    { type: 'text', text: attachmentContent('Question', [file]) },
    { type: 'image_url', image_url: { url: image.base64Url } },
  ]);
  console.log('PASS attachment type/size/encoding/binary guards and text-only/mixed request content');
})().catch(error => { console.error(error); process.exitCode = 1; });
