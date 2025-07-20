from langchain_core.messages import HumanMessage, AIMessage

class ChatMemoryAgent:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def get_conversation(self) -> list:
        return self.messages

    def clear(self):
        self.messages = []

# ✅ Test
if __name__ == "__main__":
    chat = ChatMemoryAgent()
    chat.add_user_message("What does my policy cover in Pune?")
    chat.add_ai_message("Your policy covers hospitalization in Tier-1 cities including Pune.")
    chat.add_user_message("Is knee surgery included?")

    print("\n📚 Chat History:")
    for msg in chat.get_conversation():
        print(f"{msg.type}: {msg.content}")
