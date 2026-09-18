# ============================================================
# CLASSES & OBJECTS BASICS
# ============================================================
#
# Classes help us organise related data and behaviour together.
#
# A class is a blueprint.
# An object is a specific thing created from that blueprint.
# ============================================================


# ============================================================
# 1. THE PROBLEM WITHOUT CLASSES
# ============================================================

# If we have only a few AI models, we could store their
# information in separate variables.

model1_name = "GPT-4o"
model1_accuracy = 0.94
model1_cost = 0.030

model2_name = "Claude-3"
model2_accuracy = 0.938
model2_cost = 0.024

# But what happens if we have 20 models?
#
# We would end up with many separate variables:
#
# model3_name
# model3_accuracy
# model3_cost
# model4_name
# model4_accuracy
# model4_cost
#
# Classes give us a cleaner way to organise this information.


# ============================================================
# 2. CREATING A CLASS
# ============================================================

# A class is a blueprint for creating objects.
#
# Here, AIModel is our blueprint.
class AIModel:

    # __init__() runs automatically when we create
    # an object from this class.
    #
    # name, accuracy and cost_per_1k are values
    # that we provide when creating the object.
    def __init__(self, name, accuracy, cost_per_1k):

        # self refers to the specific object being created.
        #
        # These are called attributes.
        self.name = name
        self.accuracy = accuracy
        self.cost_per_1k = cost_per_1k

    # A function inside a class is called a method.
    #
    # This method describes the AI model.
    def describe(self):

        return f"{self.name}: {self.accuracy:.0%} accuracy, ${self.cost_per_1k}/1k tokens"

    # This method checks whether the model's cost
    # is within a budget provided by the user.
    def is_affordable(self, budget):

        return self.cost_per_1k <= budget


# ============================================================
# 3. CREATING OBJECTS
# ============================================================

# AIModel is the class (blueprint).
#
# gpt4 is an object/instance created from that blueprint.
gpt4 = AIModel("GPT-4o", 0.94, 0.030)

# We can create another object from the same class.
claude = AIModel("Claude-3", 0.938, 0.024)

# And another one.
llama = AIModel("Llama-3", 0.897, 0.000)


# ============================================================
# 4. ACCESSING ATTRIBUTES
# ============================================================

# Each object has its own data.
#
# gpt4.name gives us the name stored inside the gpt4 object.
print(gpt4.name)

# gpt4.accuracy gives us the accuracy stored inside gpt4.
print(gpt4.accuracy)

# claude.name gives us Claude's name.
print(claude.name)

# Each object keeps its own values.
print(llama.cost_per_1k)


# ============================================================
# 5. CALLING METHODS
# ============================================================

# describe() is a method belonging to the AIModel class.
#
# We call it using the object followed by a dot.
print(gpt4.describe())
print(claude.describe())
print(llama.describe())


# is_affordable() is another method.
#
# We provide a budget as an argument.
print(llama.is_affordable(0.01))

# Llama's cost is 0.000, which is less than 0.01,
# so the result is True.


# ============================================================
# 6. UNDERSTANDING THE KEY TERMS
# ============================================================

# CLASS
# AIModel is the class.
#
# It is the blueprint used to create AI model objects.


# OBJECT / INSTANCE
# gpt4, claude and llama are objects.
#
# They are specific instances created from the AIModel class.


# __init__
# __init__() sets up a new object when it is created.


# self
# self refers to the specific object being worked on.
#
# For example:
#
# gpt4.name
#
# refers to the name belonging to the gpt4 object.


# ATTRIBUTE
# name, accuracy and cost_per_1k are attributes.
#
# They store data about an object.


# METHOD
# describe() and is_affordable() are methods.
#
# They define behaviour/actions that an object can perform.


# ============================================================
# 7. A MORE REALISTIC EXAMPLE
# ============================================================

# We can create a class that keeps track of
# AI model evaluation results.

class ModelEvaluator:

    # When we create an evaluator, we provide
    # the model name.
    def __init__(self, model_name):

        # Store the model name inside the object.
        self.model_name = model_name

        # Start with an empty list.
        #
        # We will add evaluation results to this list later.
        self.results = []

    # This method adds one evaluation result.
    def add_result(self, question, predicted, actual):

        # Compare the prediction with the actual answer.
        #
        # == checks whether two values are equal.
        correct = predicted == actual

        # Store the result as a dictionary inside the list.
        self.results.append({
            "question": question,
            "predicted": predicted,
            "actual": actual,
            "correct": correct
        })

    # This method calculates the model's accuracy.
    def accuracy(self):

        # If there are no results yet,
        # return 0 instead of trying to divide by zero.
        if not self.results:
            return 0

        # Count how many results were correct.
        #
        # r represents each result in self.results.
        correct = sum(1 for r in self.results if r["correct"])

        # Accuracy = correct answers / total answers.
        return correct / len(self.results)

    # This method prints a summary of the evaluation.
    def summary(self):

        print(f"Model: {self.model_name}")
        print(f"Tests: {len(self.results)}")
        print(f"Accuracy: {self.accuracy():.0%}")


# ============================================================
# 8. USING THE MODEL EVALUATOR
# ============================================================

# Create an evaluator object for GPT-4o.
evaluator = ModelEvaluator("GPT-4o")


# Add the first test.
evaluator.add_result(
    "Capital of Kenya?",
    "Nairobi",
    "Nairobi"
)


# Add the second test.
#
# The prediction is Lagos, but the actual answer is Abuja.
# Therefore, this result is incorrect.
evaluator.add_result(
    "Capital of Nigeria?",
    "Lagos",
    "Abuja"
)


# Add the third test.
#
# This prediction is correct.
evaluator.add_result(
    "Capital of Ghana?",
    "Accra",
    "Accra"
)


# Print the evaluation summary.
evaluator.summary()

