from openai import OpenAI
import json
import time

client = OpenAI()


class LayeredMemoryAgent:
    """分层记忆 Agent（MemGPT 思想的工程实现）
    
    三层记忆架构：
    1. Core Memory（核心记忆）— 始终在上下文中，放最关键的信息
    2. Working Memory（工作记忆）— 当前任务相关的短期信息
    3. Archive Memory（归档记忆）— 外部存储，按需检索
    """
    
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        
        # 核心记忆：始终在 Prompt 中，放用户画像和关键偏好
        self.core_memory = {
            "user_name": "",
            "preferences": [],
            "key_facts": [],
            "active_goals": [],
        }
        
        # 工作记忆：当前对话的相关上下文
        self.working_memory = []
        self.max_working_items = 10
        
        # 归档记忆：持久化存储，模拟向量数据库
        self.archive_memory = []
        
        # 对话历史
        self.conversation = []
    
    def chat(self, user_input: str) -> str:
        """主对话入口"""
        
        # 1. 自动记忆管理：检查是否需要更新记忆
        self._auto_manage_memory(user_input)
        
        # 2. 构建包含记忆的 Prompt
        messages = self._build_messages(user_input)
        
        # 3. 调用 LLM
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2000,
            tools=self._get_memory_tools()
        )
        
        # 4. 处理工具调用（记忆自我编辑）
        reply = self._process_response(response)
        
        # 5. 保存到对话历史
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": reply})
        
        return reply
    
    def _auto_manage_memory(self, user_input: str):
        """自动从用户输入中提取关键信息更新记忆"""
        
        # 使用小模型快速判断是否需要更新记忆
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"""分析以下用户输入，提取需要记住的关键信息。

当前核心记忆：
{json.dumps(self.core_memory, ensure_ascii=False, indent=2)}

用户输入：{user_input}

如果包含需要记忆的信息（如姓名、偏好、重要事实），返回 JSON：
{{"updates": {{"field": "值", ...}}, "archive": "需要归档的内容或null"}}

如果没有需要记忆的信息，返回：
{{"updates": {{}}, "archive": null}}"""
            }],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 更新核心记忆
        for key, value in result.get("updates", {}).items():
            if key in self.core_memory and value:
                if isinstance(self.core_memory[key], list):
                    if value not in self.core_memory[key]:
                        self.core_memory[key].append(value)
                else:
                    self.core_memory[key] = value
        
        # 归档长内容
        archive_content = result.get("archive")
        if archive_content:
            self.archive_memory.append({
                "content": archive_content,
                "timestamp": time.time(),
                "source": "auto_extract"
            })
    
    def _build_messages(self, user_input: str) -> list[dict]:
        """构建包含记忆的完整 Prompt"""
        
        # System Prompt：包含核心记忆
        system_prompt = f"""你是一个具备分层记忆能力的 Agent。

## 核心记忆（始终记住）
{json.dumps(self.core_memory, ensure_ascii=False, indent=2)}

## 工作记忆（当前任务相关）
{json.dumps(self.working_memory[-5:], ensure_ascii=False, indent=2)}

## 记忆管理指令
- 核心记忆中的信息是你的"常识"，始终作为回答的依据
- 如果用户问到归档记忆中的内容，使用 search_archive 工具搜索
- 如果需要记住新的重要信息，使用 update_core_memory 工具
- 如果当前对话产生了需要长期保存的内容，使用 archive_content 工具"""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # 添加近期对话（保留最近 10 轮）
        recent = self.conversation[-20:]  # 10轮 = 20条消息
        messages.extend(recent)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _get_memory_tools(self) -> list[dict]:
        """定义记忆管理工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_archive",
                    "description": "在归档记忆中搜索相关内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_core_memory",
                    "description": "更新核心记忆中的字段",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "字段名（user_name/preferences/key_facts/active_goals）"
                            },
                            "value": {
                                "type": "string",
                                "description": "新的值"
                            }
                        },
                        "required": ["field", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "archive_content",
                    "description": "将内容保存到归档记忆",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "要归档的内容"
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
        ]
    
    def _process_response(self, response) -> str:
        """处理 LLM 响应，执行记忆工具调用"""
        message = response.choices[0].message
        
        # 如果没有工具调用，直接返回文本
        if not message.tool_calls:
            return message.content or ""
        
        # 执行工具调用
        tool_results = []
        text_parts = []
        
        if message.content:
            text_parts.append(message.content)
        
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if func_name == "search_archive":
                results = self._search_archive(args["query"])
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "result": json.dumps(results, ensure_ascii=False)
                })
                text_parts.append(f"[检索到归档内容: {len(results)} 条]")
            
            elif func_name == "update_core_memory":
                field = args["field"]
                value = args["value"]
                if field in self.core_memory:
                    if isinstance(self.core_memory[field], list):
                        self.core_memory[field].append(value)
                    else:
                        self.core_memory[field] = value
                    text_parts.append(f"[已更新核心记忆: {field}]")
            
            elif func_name == "archive_content":
                self.archive_memory.append({
                    "content": args["content"],
                    "timestamp": time.time(),
                    "source": "agent_archived"
                })
                text_parts.append("[已归档内容]")
        
        return "\n".join(text_parts)
    
    def _search_archive(self, query: str) -> list[dict]:
        """在归档记忆中搜索（简化版：关键词匹配）"""
        results = []
        query_lower = query.lower()
        
        for item in self.archive_memory:
            if query_lower in item["content"].lower():
                results.append(item)
        
        return results[:5]  # 最多返回5条


# 使用示例
agent = LayeredMemoryAgent()

# 第一轮：用户介绍自己
print(agent.chat("你好！我叫小明，我是一名数据科学家，平时喜欢用 Python"))

# 第二轮：用户提出偏好
print(agent.chat("我比较喜欢简洁的回答，不要太啰嗦"))

# 第三轮：用户讨论工作
print(agent.chat("我正在做一个客户流失预测项目，使用的是 XGBoost"))

# 第四轮：验证记忆是否保持
print(agent.chat("我之前说我在做什么项目来着？"))
# Agent 应该能从核心记忆中回忆起"客户流失预测项目"

# 查看核心记忆状态
print("\n当前核心记忆：")
print(json.dumps(agent.core_memory, ensure_ascii=False, indent=2))
