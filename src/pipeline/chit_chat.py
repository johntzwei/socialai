"""Chit-chat filter judge: classifies whether a conversation is casual chit-chat."""

from pipeline.judge import Judge

SYSTEM_PROMPT = """You are a dataset curator. Analyze the first exchange (user message and assistant response) in a conversation with an AI assistant.

Determine if this conversation is CHIT-CHAT or TASK-ORIENTED.

CHIT-CHAT (keep = true):
- Simple greetings or small talk ("Hi", "How are you?", "What's up?")
- Casual social conversation, sharing feelings, venting
- Phatic communication with no concrete task
- Social or romantic role-play scenarios

NOT CHIT-CHAT (keep = false):
- The user asks the AI to act as a specific professional persona
- Creative writing tasks ("Write a poem", "Write a letter")
- Programming or technical tasks
- Information seeking, factual questions, advice requests
- Content generation, translation, summarization
- Online threads/posts, story completion
- Large corpus of text with no conversational prompt

Return ONLY valid JSON:
{
  "reasoning": "...",
  "keep": true/false
}"""


class ChitChatJudge(Judge):

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def format_conversation(self, row: dict) -> str | None:
        conversation = row.get("conversation", [])
        if not conversation:
            return None
        user_msg = conversation[0].get("content", "").strip()
        if len(user_msg) < 5:
            return None
        if len(user_msg) > 4000:
            user_msg = user_msg[:4000]

        # Include first assistant response if available
        assistant_msg = ""
        if len(conversation) > 1:
            assistant_msg = conversation[1].get("content", "").strip()
            if len(assistant_msg) > 4000:
                assistant_msg = assistant_msg[:4000]

        text = f"[USER]: {user_msg}"
        if assistant_msg:
            text += f"\n\n[ASSISTANT]: {assistant_msg}"
        return text

    def judge_type(self) -> str:
        return "chit_chat"


if __name__ == "__main__":
    ChitChatJudge.cli()
