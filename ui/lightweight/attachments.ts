export type ChatAttachment =
  | { type: "image"; name: string; size: number; base64Url: string }
  | { type: "text"; name: string; size: number; content: string };

const imageTypes: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
  webp: "image/webp", gif: "image/gif",
};
const textExtensions = "txt md markdown log csv tsv json jsonl ndjson ipynb yaml yml toml ini cfg conf xml svg html htm css scss less js jsx mjs cjs ts tsx py pyi c h cc cpp cxx hpp cs java kt kts go rs swift sh bash zsh ps1 bat cmd sql vue svelte rb php lua r tex diff patch env gitignore dockerignore".split(" ");
export const fileAccept = [
  ...Object.values(imageTypes), "text/*", ...textExtensions.map((ext) => "." + ext),
].join(",");

export async function readAttachment(
  file: File,
  vision: boolean,
  capabilities?: { image_types?: string[] },
): Promise<ChatAttachment> {
  const extension = file.name.split(".").pop()!.toLowerCase();
  const imageType = imageTypes[extension] || (Object.values(imageTypes).includes(file.type) ? file.type : "");
  if (imageType) {
    if (!vision) throw Error("当前模型不支持图片");
    const allowed = capabilities?.image_types;
    if (allowed?.length && !allowed.includes(imageType)) {
      const labels = allowed.map((type) => type.split("/")[1]?.toUpperCase()).join(" / ");
      throw Error(`当前后端仅支持 ${labels} 图片`);
    }
    if (file.size > 10 * 1024 * 1024) throw Error("图片每张不超过 10 MiB");
    const base64Url = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(Error("读取文件失败"));
      reader.onabort = () => reject(Error("读取文件失败"));
      reader.readAsDataURL(file.slice(0, file.size, imageType));
    });
    return { type: "image", name: file.name, size: file.size, base64Url };
  }
  if (!textExtensions.includes(extension) && !file.type.startsWith("text/") &&
      !/^(readme|license|makefile|dockerfile)$/i.test(file.name))
    throw Error("仅支持图片、文本和代码文件");
  if (file.size > 1024 * 1024) throw Error("文本或代码文件每个不超过 1 MiB");
  const data = new Uint8Array(await file.arrayBuffer());
  const encoding = data[0] === 0xff && data[1] === 0xfe ? "utf-16le"
    : data[0] === 0xfe && data[1] === 0xff ? "utf-16be" : "utf-8";
  let content: string;
  try { content = new TextDecoder(encoding, { fatal: true }).decode(data); }
  catch { throw Error("无法读取文本编码，请另存为 UTF-8 后重试"); }
  if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(content))
    throw Error("文件包含二进制内容，无法作为文本发送");
  return { type: "text", name: file.name, size: file.size, content };
}

export function supportedAttachment(value: any) {
  return value?.type === "image" ? typeof value.base64Url === "string" && !!value.base64Url
    : value?.type === "text" && typeof value.content === "string" && typeof value.name === "string";
}

export function attachmentContent(content: string, attachments: ChatAttachment[]) {
  const text = [content, ...attachments.filter((file) => file.type === "text").map((file) =>
    `--- File: ${JSON.stringify(file.name)} ---\n${file.content}\n--- End file ---`,
  )].filter(Boolean).join("\n\n");
  const images = attachments.filter((file) => file.type === "image");
  return images.length ? [
    { type: "text", text },
    ...images.map((file) => ({ type: "image_url", image_url: { url: file.base64Url } })),
  ] : text;
}
