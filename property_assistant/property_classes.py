# Property Assistant — Classes & Objects

import os

import pandas as pd

# Property class


class Property:
    def __init__(
        self,
        property_id,
        location,
        property_type,
        bedrooms,
        bathrooms,
        price_kes,
        furnished
    ):
        self.property_id = property_id
        self.location = location
        self.property_type = property_type
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.price_kes = price_kes
        self.furnished = furnished

    def describe(self):
        furnished_status = "Furnished" if self.furnished else "Unfurnished"

        return (
            f"{self.property_type} in {self.location}: "
            f"{self.bedrooms} bedrooms, "
            f"{self.bathrooms} bathrooms, "
            f"KES {self.price_kes:,.0f} | "
            f"{furnished_status}"
        )

    def matches_budget(self, budget):
        return self.price_kes <= budget

    def has_bedrooms(self, bedrooms):
        return self.bedrooms >= bedrooms

    def is_furnished(self):
        return self.furnished


# Property manager


class PropertyManager:
    def __init__(self):
        self.properties = []

    def add_property(self, property):
        self.properties.append(property)

    def list_all(self, properties=None):
        properties = properties if properties is not None else self.properties

        for property in properties:
            print(property.describe())

    def search_by_location(self, location):
        return [
            property
            for property in self.properties
            if property.location.lower() == location.lower()
        ]

    def search_by_type(self, property_type):
        return [
            property
            for property in self.properties
            if property.property_type.lower() == property_type.lower()
        ]

    def within_budget(self, budget):
        return [
            property
            for property in self.properties
            if property.matches_budget(budget)
        ]

    def with_minimum_bedrooms(self, bedrooms):
        return [
            property
            for property in self.properties
            if property.has_bedrooms(bedrooms)
        ]

    def furnished_properties(self):
        return [
            property
            for property in self.properties
            if property.is_furnished()
        ]

    def sort_by_price(self, properties=None, ascending=True):
        properties = properties if properties is not None else self.properties

        return sorted(
            properties,
            key=lambda property: property.price_kes,
            reverse=not ascending
        )

    def most_expensive(self):
        return max(
            self.properties,
            key=lambda property: property.price_kes
        )

    def cheapest(self):
        return min(
            self.properties,
            key=lambda property: property.price_kes
        )

    def summary(self):
        if not self.properties:
            print("No properties available.")
            return

        locations = set(property.location for property in self.properties)
        property_types = set(
            property.property_type for property in self.properties
        )

        print("\n=== Property Summary ===")
        print(f"Total properties: {len(self.properties)}")
        print(f"Locations: {len(locations)}")
        print(f"Property types: {len(property_types)}")
        print(f"Cheapest: KES {self.cheapest().price_kes:,.0f}")
        print(f"Most expensive: KES {self.most_expensive().price_kes:,.0f}")


# Load property data

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties_data = pd.read_csv(CSV_PATH)

# Build property manager

manager = PropertyManager()

for _, row in properties_data.iterrows():
    property = Property(
        property_id=row["property_id"],
        location=row["property_location"],
        property_type=row["property_type"],
        bedrooms=row["bedrooms"],
        bathrooms=row["bathrooms"],
        price_kes=row["price_kes"],
        furnished=row["furnished"]
    )

    manager.add_property(property)

# Property summary

manager.summary()

# Search by location

location = "Kilimani"

location_results = manager.search_by_location(location)

print(f"\n=== Properties in {location} ===")
manager.list_all(location_results)

# Search by property type

property_type = "Apartment"

type_results = manager.search_by_type(property_type)

print(f"\n=== {property_type}s ===")
manager.list_all(type_results)

# Find properties within a budget

budget = 100000

budget_results = manager.within_budget(budget)

print(f"\n=== Properties Within KES {budget:,.0f} ===")
manager.list_all(budget_results)

# Find properties with a minimum number of bedrooms

bedrooms = 3

bedroom_results = manager.with_minimum_bedrooms(bedrooms)

print(f"\n=== Properties With {bedrooms}+ Bedrooms ===")
manager.list_all(bedroom_results)

# Find furnished properties

furnished_results = manager.furnished_properties()

print("\n=== Furnished Properties ===")
manager.list_all(furnished_results)

# Sort properties by price

sorted_properties = manager.sort_by_price()

print("\n=== Properties Sorted by Price ===")
manager.list_all(sorted_properties)

# Find cheapest and most expensive properties

cheapest = manager.cheapest()
most_expensive = manager.most_expensive()

print(f"\nCheapest: {cheapest.describe()}")
print(f"Most expensive: {most_expensive.describe()}")
