import anthropic


class PropertyChatbot:
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


# Create a property assistant with a defined persona
property_assistant = PropertyChatbot(
    persona="""You are a helpful AI Property Assistant.

    Help users understand property options and describe their housing needs.
    Use simple, clear English.
    Ask relevant follow-up questions when important details are missing.
    Help users compare property features such as location, bedrooms,
    bathrooms, furnishing, parking, water supply and amenities.

    Do not invent property listings, prices or availability.
    If you do not have the required information, say so clearly.
    Stay focused on property-related questions."""
)


# Example conversation
conversation = [
    "I'm looking for a two-bedroom home in Ruiru.",
    "I also need reliable water and secure parking.",
    "What other features should I consider?",
]

print("=== PROPERTY ASSISTANT ===\n")

for message in conversation:
    print(f"You: {message}")
    reply = property_assistant.chat(message)
    print(f"Assistant: {reply}\n")

print("Session summary:", property_assistant.summary())


# Simple command-line chat loop
def run_property_chatbot():
    bot = PropertyChatbot(
        persona="""You are a helpful AI Property Assistant.
        Help users with property-related questions.
        Use simple English and never invent property listings or prices."""
    )

    print("\nProperty Assistant ready!")
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
        print(f"\nAssistant: {response}\n")