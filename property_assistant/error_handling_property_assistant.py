import time


# ============================================================
# AI PROPERTY ASSISTANT — ERROR HANDLING
# ============================================================

# Sample property data
properties = [
    {
        "id": 1,
        "name": "Kilimani Apartment",
        "location": "Kilimani",
        "bedrooms": 2,
        "price_kes": 75000
    },
    {
        "id": 2,
        "name": "Ruiru Townhouse",
        "location": "Ruiru",
        "bedrooms": 3,
        "price_kes": 55000
    },
    {
        "id": 3,
        "name": "Westlands Apartment",
        "location": "Westlands",
        "bedrooms": 2,
        "price_kes": 95000
    }
]


# ============================================================
# 1. VALIDATE USER INPUT
#    Using raise to create our own errors
# ============================================================

def validate_search(location, bedrooms, budget):
    """Validate a property search before processing it."""

    if not location:
        raise ValueError("Location is required")

    if bedrooms <= 0:
        raise ValueError("Bedrooms must be greater than 0")

    if budget <= 0:
        raise ValueError("Budget must be greater than 0")

    return True


# ============================================================
# 2. SEARCH PROPERTIES
#    try / except handles invalid input
# ============================================================

def search_properties(location, bedrooms, budget):
    """Find properties matching the user's requirements."""

    try:
        validate_search(location, bedrooms, budget)

        matches = []

        for property in properties:
            if (
                property["location"].lower() == location.lower()
                and property["bedrooms"] == bedrooms
                and property["price_kes"] <= budget
            ):
                matches.append(property)

        return matches

    except ValueError as e:
        print(f"Invalid search: {e}")
        return []


# ============================================================
# 3. HANDLE MISSING DATA
#    KeyError can happen when expected property data is missing
# ============================================================

def display_property(property):
    """Display a property's information safely."""

    try:
        print(
            f"{property['name']} | "
            f"{property['location']} | "
            f"{property['bedrooms']} bedrooms | "
            f"KES {property['price_kes']:,}"
        )

    except KeyError as e:
        print(f"Property data is missing the field: {e}")


# ============================================================
# 4. CALCULATE A PROPERTY METRIC
#    Demonstrates multiple exception types
# ============================================================

def calculate_price_per_bedroom(property):
    """Calculate the property's price per bedroom."""

    try:
        price = property["price_kes"]
        bedrooms = property["bedrooms"]

        return round(price / bedrooms)

    except KeyError as e:
        print(f"Missing property field: {e}")
        return None

    except ZeroDivisionError:
        print("Cannot calculate price per bedroom: bedrooms is zero")
        return None

    except TypeError:
        print("Price and bedrooms must be numbers")
        return None


# ============================================================
# 5. LOAD PROPERTY DATA
#    finally runs whether loading succeeds or fails
# ============================================================

def load_property_data(filepath):
    """Load property data from a text file safely."""

    file = None

    try:
        file = open(filepath, "r")

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

        return lines

    except FileNotFoundError:
        print(f"Property data file not found: {filepath}")
        return []

    finally:
        if file:
            file.close()

        print(f"Finished loading attempt: {filepath}")


# ============================================================
# 6. SIMULATE AN AI/API CALL
#    External services can fail, so we retry
# ============================================================

def ask_ai(question, retries=3):
    """Simulate an AI API request with retry handling."""

    for attempt in range(retries):

        try:
            # Simulate an API call
            if attempt < 2:
                raise ConnectionError("AI service temporarily unavailable")

            return f"AI response for: {question}"

        except ConnectionError:
            print(
                f"AI connection failed "
                f"(attempt {attempt + 1}/{retries})"
            )

            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected AI error: {e}")
            return None

    print("AI service unavailable after all retries")
    return None


# ============================================================
# 7. RUN THE PROPERTY ASSISTANT
# ============================================================

def property_assistant(location, bedrooms, budget):

    print("\n=== AI PROPERTY ASSISTANT ===")

    # Search properties
    matches = search_properties(
        location,
        bedrooms,
        budget
    )

    if not matches:
        print("No matching properties found.")
        return

    # Display matching properties
    print(f"\nFound {len(matches)} matching property/properties:\n")

    for property in matches:
        display_property(property)

        price_per_bedroom = calculate_price_per_bedroom(property)

        if price_per_bedroom is not None:
            print(
                f"Price per bedroom: "
                f"KES {price_per_bedroom:,}"
            )

        print()


# ============================================================
# 8. TEST THE ASSISTANT
# ============================================================

print("=== TEST 1: Valid search ===")

property_assistant(
    location="Kilimani",
    bedrooms=2,
    budget=80000
)


print("\n=== TEST 2: Invalid budget ===")

property_assistant(
    location="Kilimani",
    bedrooms=2,
    budget=-50000
)


print("\n=== TEST 3: Missing data file ===")

data = load_property_data("properties_backup.txt")
print(f"Loaded {len(data)} lines")


print("\n=== TEST 4: AI API ===")

response = ask_ai(
    "Summarise these property results for the user."
)

if response:
    print(response)