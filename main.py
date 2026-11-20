"""astrbot_plugin_vision_memory

Companion plugin for astrbot_plugin_livingmemory.

1. Image -> text: caption images through an AstrBot provider (vision-capable)
   so a text-only main model (e.g. DeepSeek) can "see" them. The caption is
   added to the user message BEFORE LivingMemory recalls, so memories can match
   character names/features in the caption.
2. Guidance prompt: appended to the system prompt, telling the model to
   recall -> web-search -> state names -> memorize (LivingMemory's tools).

Long-term memory itself is handled entirely by LivingMemory. All model calls go
through AstrBot's own providers; this plugin holds no API keys.
"""

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star


class VisionMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    # priority > default so we run before LivingMemory's recall hook (best effort)
    @filter.on_llm_request(priority=10)
    async def on_req(self, event: AstrMessageEvent, req: ProviderRequest):
        guidance = (self.config.get("guidance_prompt") or "").strip()
        if guidance and guidance not in (req.system_prompt or ""):
            req.system_prompt = (req.system_prompt or "") + "\n\n" + guidance

        if self.config.get("enable_caption", True) and req.image_urls:
            await self._caption(req)

    async def _caption(self, req: ProviderRequest) -> None:
        pid = (self.config.get("caption_provider") or "").strip()
        if not pid:
            return  # not configured: leave images untouched
        prompt = self.config.get("caption_prompt") or "请用中文详细描述这张图片。"
        parts = []
        for i, url in enumerate(list(req.image_urls), 1):
            try:
                resp = await self.context.llm_generate(
                    chat_provider_id=pid, prompt=prompt, image_urls=[url]
                )
                text = (resp.completion_text or "").strip()
            except Exception as e:  # never break the chat because captioning failed
                logger.warning(f"[vision_memory] caption failed: {type(e).__name__}: {e}")
                text = ""
            parts.append(f"[图片{i}描述] {text or '(识别失败)'}")
        logger.info(f"[vision_memory] captioned {len(parts)} image(s)")
        if self.config.get("replace_images", True):
            req.image_urls = []
        req.prompt = (req.prompt or "") + "\n\n" + "\n".join(parts)

    async def terminate(self):
        pass
