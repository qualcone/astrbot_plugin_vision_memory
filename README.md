# astrbot_plugin_vision_memory

让文字模型（如 DeepSeek）的机器人能"看图"，遇到不熟悉的角色时联网确认，并把结论存进 **LivingMemory**。本插件不保存 API Key，所有模型调用都走 AstrBot 已配置的提供商。

## 本插件做的事
1. **图转文**：收到图片时，用你指定的识图提供商生成描述，替换原图后交给主模型。
2. **行为指引**：在系统提示词后追加一段（不覆盖人设），让模型按"回忆 → 联网搜索 → 说出角色名和作品 → 写入记忆"的流程处理。

长期记忆本身（召回、自动总结、写入工具）完全由 [astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory) 负责。

## 配置步骤
1. 安装 LivingMemory 和本插件（AstrBot ≥ 4.24.2）。
2. 添加一个能识图的提供商（如 Gemini），在本插件配置里选它。**同时清空 AstrBot 内置的"默认图片转述模型"**，避免重复转述。
3. LivingMemory 配置：
   - `provider_settings.embedding_provider_id`：选一个已启用的嵌入提供商（必需）。
   - `agent_tools.enable_memorize_tool`：**开启**（默认关闭；开启后需重载插件）。这样模型才能主动把确认过的角色写入记忆。
   - `agent_tools.enable_recall_tool`：保持开启（默认开启）。
   - `recall_engine.top_k`：建议 3（默认 5），控制每次注入的条数。
   - `reflection_engine.summary_trigger_rounds`：默认 10 轮对话总结一次。
4. 在机器人实际使用的配置文件里开启 AstrBot 内置 Web Search（配置 → AI → 能力）。主模型必须支持 Function Calling，否则搜索和记忆工具都不会被调用。

## 注意
- 图片会发送到你选的识图提供商；免费层数据可能被服务商用于改进产品。
- 记忆由模型写入，可能出错；可用 `/lmem forget <id>`（管理员）删除。
- 描述文字会在 LivingMemory 召回之前加入用户消息（best effort，依赖 hook 优先级）。
