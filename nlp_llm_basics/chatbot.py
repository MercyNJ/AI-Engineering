import anthropic


# --------------------------------------------------
# Mock chatbot — demonstrates the architecture without an API
# --------------------------------------------------

class MockChatbot:
    """Simulated chatbot for learning conversation structure."""

    RESPONSES = {
        "variable": "A variable is a named container for a value. Example: name = 'Amara'.",
        "function": "A function is reusable code. Define it using def and call it when needed.",
        "loop": "A loop repeats code. For example, a for loop runs once for each item.",
        "list": "A list stores multiple items in order. Example: fruits = ['apple', 'mango'].",
        "default": "Great question! Could you be more specific so I can help?",
    }

    def __init__(self, name="Iv"):
        self.name = name
        self.history = []

    def chat(self, message):
        # Keep the chatbot focused on its intended subject
        if "competitor" in message.lower():
            reply = "I'm here to help with Iv Code Academy content only!"
        else:
            msg_lower = message.lower()
            reply = self.RESPONSES["default"]

            for keyword, response in self.RESPONSES.items():
                if keyword != "default" and keyword in msg_lower:
                    reply = response
                    break

        # Save both sides of the conversation
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": reply})

        return reply

    def reset(self):
        self.history = []
        return "Conversation reset!"

    def summary(self):
        turns = len(self.history) // 2
        words = sum(
            len(message["content"].split())
            for message in self.history
        )

        return f"Turns: {turns} | Total words: {words}"


# Try the mock chatbot
mock_bot = MockChatbot("Iv")

print(f"=== Chat with {mock_bot.name} ===\n")

conversation = [
    "What is a variable?",
    "How do I use a loop?",
    "Can you show me a list example?",
    "What about a function?",
    "Thank you!",
]

for message in conversation:
    print(f"You: {message}")
    reply = mock_bot.chat(message)
    print(f"Iv: {reply}\n")

print("Session summary:", mock_bot.summary())
print(mock_bot.reset())
print("After reset:", mock_bot.summary())


# --------------------------------------------------
# Real chatbot using Anthropic
# --------------------------------------------------

class Chatbot:
    def __init__(self, persona, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = persona
        self.history = []
        self.max_turns = 20

    def chat(self, user_message):
        """Send a message and get a response."""

        self.history.append({
            "role": "user",
            "content": user_message
        })

        # Send only the most recent messages to manage context size
        trimmed_history = self.history[-self.max_turns:]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system,
                messages=trimmed_history,
            )

            reply = response.content[0].text

            self.history.append({
                "role": "assistant",
                "content": reply
            })

            return reply

        except Exception as error:
            # Remove the unanswered user message so history stays consistent
            self.history.pop()
            return f"Error: {error}"

    def chat_stream(self, user_message):
        """Display the response as it arrives."""

        self.history.append({
            "role": "user",
            "content": user_message
        })

        full_reply = ""

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=self.system,
                messages=self.history[-self.max_turns:],
            ) as stream:

                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_reply += text

            print()

            self.history.append({
                "role": "assistant",
                "content": full_reply
            })

            return full_reply

        except Exception as error:
            self.history.pop()
            print(f"\nError: {error}")
            return None

    def reset(self):
        """Clear conversation history."""
        self.history = []
        return "Conversation reset!"

    def summary(self):
        turns = len(self.history) // 2

        words = sum(
            len(message["content"].split())
            for message in self.history
        )

        return f"Turns: {turns} | Total words: {words}"


# --------------------------------------------------
# Create a chatbot with a defined persona
# --------------------------------------------------

tutor = Chatbot(
    persona="""You are Iv, an AI learning assistant for Iv Code Academy.

    You help students learning Python and AI in East Africa.
    Use simple English and give short, clear answers.
    Use relevant examples from Kenya, Nigeria, or Ghana when helpful.
    Stay focused on Python and AI learning.
    Always encourage the student."""
)


# Test conversation memory
print("\n=== REAL CHATBOT ===\n")

print("You: What is a variable?")
print("Iv:", tutor.chat("What is a variable?"))

print("\nYou: Can you give me a real-world example?")
print("Iv:", tutor.chat("Can you give me a real-world example?"))

print("\nSession summary:", tutor.summary())


# --------------------------------------------------
# Simple command-line chat loop
# --------------------------------------------------

def run_chatbot():
    bot = Chatbot(
        persona="You are Iv, a helpful Python and AI tutor. Use simple English."
    )

    print("\nAI Tutor ready!")
    print("Type 'quit' to exit or 'reset' to start over.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            print(bot.reset())
            continue

        response = bot.chat(user_input)
        print(f"\nIv: {response}\n")