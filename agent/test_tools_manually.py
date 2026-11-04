
from tools import (
    get_race_results,
    get_qualifying_results,
    get_fastest_lap,
    get_driver_standings
)


print("\nTesting race results...")
print(get_race_results(2025, "Miami"))

print("\nTesting qualifying...")
print(get_qualifying_results(2025, "Miami"))

print("\nTesting fastest lap...")
print(get_fastest_lap(2025, "Miami"))

print("\nTesting driver standings...")
print(get_driver_standings(2025))
