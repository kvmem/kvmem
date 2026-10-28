# Pelican demo: extra-long / xhigh reasoning on limited VRAM

## 中文

这是 2026-10-28 KVMem 发布视频中的鹈鹕骑行 HTML 动画示例。
实测设备为 **NVIDIA GeForce RTX 5060 Ti 16GB**，模型为 **27B IQ3**，
两份输出分别使用 **medium** 和 **xhigh** 推理档位。设备、模型和档位信息由本次实测的操作者提供。

| medium | xhigh |
|:---:|:---:|
| ![medium 鹈鹕骑行动画](../assets/pelican-reasoning/medium.gif) | ![xhigh 鹈鹕骑行动画](../assets/pelican-reasoning/xhigh.gif) |

图片与动画直接复用发布视频素材，由模型生成的原始 HTML 在浏览器中渲染、录制；没有重绘模型输出。
首页使用静态预览，点击后打开对应动画。medium 的录制裁去了网页外边距并保留完整 SVG 场景，
xhigh 保留完整页面；GIF 沿用录制时间，统一输出为 15 fps、960 像素宽。

KVMem 的生成预留空间 R（Gen reserve）可以循环使用：追加空间不足时，
从 GPU 驻留 KV 中重新选择工作集，释放追加空间后继续生成。因此一次回答可以超过 R，
也能为 `xhigh` 档位的长推理提供空间；输出仍受请求限制、逻辑上下文和模型停止条件约束。
机制与验证记录见 [后端差异](../multi-backend-differences.md) 和 [图召回 / Pressure](../graph-recall.md)。

这两份输出是单次实测示例，不是受控质量评测；没有据此计算推理 token 数、耗时、显存峰值或平均质量提升。
`xhigh` 是推理档位，不是固定输出长度。16GB 指显卡容量；运行模型和保存长历史仍需系统内存。

## English

These pelican cycling HTML animations appeared in the KVMem release video on
2026-10-28. The local runs used an **NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM**
and a **27B IQ3 model**, with **medium** and **xhigh** reasoning settings.
Hardware, model and setting information was supplied by the operator of these runs.

The previews and animations reuse the video's assets, captured by rendering the
model's original HTML in a browser. The artwork was not redrawn. The medium capture
removes page margins while retaining the complete SVG scene; xhigh retains the
whole page. GIFs preserve capture timing at 15 fps and 960 pixels wide. The README
uses still previews that link to the animations above.

KVMem reuses its generation reserve R: when append space runs short, it selects a
working set from resident GPU KV and frees space to continue generating. A response
can therefore exceed R, making room for long reasoning at `xhigh`. Request limits,
logical context and model stop conditions still apply. See [backend differences](../multi-backend-differences.md)
and [graph recall / Pressure](../graph-recall.md) for the mechanism and validation.

These are individual examples, not a controlled quality benchmark. They do not
establish reasoning-token counts, timing, peak VRAM use or an average quality gain.
`xhigh` is a reasoning setting, not a fixed output length. The 16 GB figure is GPU
capacity; model execution and long-history storage also need system RAM.

## Original output provenance

The original HTML files used in video production have the following SHA-256 values:

| File | SHA-256 |
|---|---|
| `0.19.0-pelican-medium.html` | `774c07c93aca16fc8476e95d2d7eb03be7b4dabcc7050eedc3f8d9b72257013f` |
| `0.19.0-pelican-xhigh.html` | `4a42cfca207c0c3157ef848a189232504c334f0b502cbf7039e7d531ea5e4eca` |
