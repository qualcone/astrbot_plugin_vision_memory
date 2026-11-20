"""astrbot_plugin_vision_memory

1. Image -> text: caption images with a vision-capable provider before the
   (text-only) main model sees the request.
2. Guidance prompt: tell the main model when to use AstrBot's built-in web
   search for unfamiliar characters/works.
3. Long-term facts: a small KV-backed memory the model can write
   (remember_character) and query (recall_character); matching facts are
   injected into later requests so the model does not need to search again.

All model calls go through AstrBot's own providers; this plugin holds no keys.
"""

import asyncio
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

try:  # available in recent AstrBot versions
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover
    TextPart = None

KV_KEY = "facts_v1"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


class VisionMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ storage
    async def _load(self) -> dict:
        data = await self.get_kv_data(KV_KEY, {})
        return data if isinstance(data, dict) else {}

    async def _save(self, facts: dict) -> None:
        limit = int(self.config.get("max_facts", 500))
        if len(facts) > limit:
            # evict least recently used
            keep = sorted(
                facts.items(), key=lambda kv: kv[1].get("last_used", 0), reverse=True
            )[:limit]
            facts = dict(keep)
        await self.put_kv_data(KV_KEY, facts)

    @staticmethod
    def _terms(fact: dict, key: str) -> list[str]:
        return [key] + [_norm(a) for a in fact.get("aliases", [])]

    # ------------------------------------------------------------------ hooks
    @filter.on_llm_request()
    async def on_req(self, event: AstrMessageEvent, req: ProviderRequest):
        # 1) guidance (stable text -> system prompt)
        guidance = (self.config.get("guidance_prompt") or "").strip()
        if guidance and guidance not in (req.system_prompt or ""):
            req.system_prompt = (req.system_prompt or "") + "\n\n" + guidance

        # 2) image -> text
        caption_text = ""
        if self.config.get("enable_caption", True) and req.image_urls:
            caption_text = await self._caption(event, req)

        # 3) inject matching long-term facts
        if self.config.get("enable_memory", True):
            await self._inject_facts(req, (req.prompt or "") + "\n" + caption_text)

    async def _caption(self, event: AstrMessageEvent, req: ProviderRequest) -> str:
        pid = (self.config.get("caption_provider") or "").strip()
        if not pid:
            return ""  # not configured: leave images untouched
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
        joined = "\n".join(parts)
        logger.info(f"[vision_memory] captioned {len(parts)} image(s)")
        if self.config.get("replace_images", True):
            req.image_urls = []
        req.prompt = (req.prompt or "") + "\n\n" + joined
        return joined

    async def _inject_facts(self, req: ProviderRequest, text: str) -> None:
        facts = await self._load()
        if not facts:
            return
        low = _norm(text)
        scored = []
        for key, f in facts.items():
            best = 0
            for t in self._terms(f, key):
                if len(t) >= 2 and t in low:
                    best = max(best, len(t))
            if best:
                scored.append((best, key))
        if not scored:
            return
        scored.sort(reverse=True)
        top = scored[: int(self.config.get("recall_top_k", 3))]
        lines = []
        async with self._lock:
            facts = await self._load()
            for _, key in top:
                f = facts.get(key)
                if not f:
                    continue
                f["last_used"] = time.time()
                lines.append(f"- {f['name']}（作品：{f.get('work') or '未知'}）：{f.get('description', '')}")
            await self._save(facts)
        if not lines:
            return
        note = "[长期记忆：以下角色/作品信息已确认过，无需再联网搜索]\n" + "\n".join(lines)
        if TextPart is not None:
            req.extra_user_content_parts.append(TextPart(text=note))
        else:
            req.prompt = (req.prompt or "") + "\n\n" + note

    # ------------------------------------------------------------------ tools
    @filter.llm_tool(name="remember_character")
    async def remember_character(
        self, event: AstrMessageEvent, name: str, work: str, description: str, aliases: str = ""
    ):
        """把已确认的动漫/游戏角色或作品信息存入长期记忆，下次遇到不用再搜索。只在通过联网搜索或用户明确告知后确认无误时调用。

        Args:
            name(string): 角色名或表情包名称
            work(string): 所属作品名
            description(string): 一句话外貌特征或出处说明
            aliases(string): 其他叫法，用英文逗号分隔，可为空
        """
        name = (name or "").strip()
        if not name:
            return "name 不能为空，未保存。"
        alias_list = [a.strip() for a in (aliases or "").replace("，", ",").split(",") if a.strip()]
        async with self._lock:
            facts = await self._load()
            facts[_norm(name)] = {
                "name": name,
                "work": (work or "").strip(),
                "description": (description or "").strip()[:300],
                "aliases": alias_list,
                "created": time.time(),
                "last_used": time.time(),
            }
            await self._save(facts)
        return f"已记住：{name}（{work}）。"

    @filter.llm_tool(name="recall_character")
    async def recall_character(self, event: AstrMessageEvent, keyword: str):
        """在长期记忆中按关键词（角色名、作品名或外貌特征）查找已确认过的角色信息。遇到不熟悉的角色时，联网搜索前先调用它。

        Args:
            keyword(string): 角色名、作品名或外貌特征关键词
        """
        kw = _norm(keyword)
        if not kw:
            return "关键词为空。"
        facts = await self._load()
        hits = []
        for key, f in facts.items():
            blob = _norm(" ".join([f["name"], f.get("work", ""), f.get("description", ""), *f.get("aliases", [])]))
            if kw in blob:
                hits.append(f"- {f['name']}（作品：{f.get('work') or '未知'}）：{f.get('description', '')}")
        k = int(self.config.get("recall_top_k", 3))
        return "\n".join(hits[:k]) if hits else "记忆中没有相关记录，可以联网搜索。"

    # ------------------------------------------------------------------ commands
    @filter.command_group("vmem")
    def vmem(self):
        """长期记忆管理"""

    @vmem.command("list")
    async def vmem_list(self, event: AstrMessageEvent):
        """列出已记住的条目"""
        facts = await self._load()
        if not facts:
            yield event.plain_result("长期记忆为空。")
            return
        lines = [f"{f['name']}（{f.get('work') or '?'}）" for f in facts.values()]
        yield event.plain_result(f"共 {len(lines)} 条：\n" + "\n".join(lines[:50]))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @vmem.command("del")
    async def vmem_del(self, event: AstrMessageEvent, name: str):
        """删除一条记忆（管理员）"""
        async with self._lock:
            facts = await self._load()
            if facts.pop(_norm(name), None) is None:
                yield event.plain_result("没找到这一条。")
                return
            await self._save(facts)
        yield event.plain_result(f"已删除：{name}")

    async def terminate(self):
        pass
