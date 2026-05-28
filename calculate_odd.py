from kprojection import KProjectionCoverage

odd_description = {
    "Weather": ["Dry", "Rain", "Fog"],
    "Lighting": ["Daytime", "Nighttime", "Glare"],
    "Camera": ["Valid", "Obscured"],
    "Scene": ["Urban", "Non-Urban"],
    "Speed": ["Under 50", "Over 50"]
}

scenario_standard = {
    "Weather": "Dry", 
    "Lighting": "Daytime", 
    "Camera": "Valid", 
    "Scene": "Urban", 
    "Speed": "Under 50"
}

scenario_fog = {
    "Weather": "Fog", 
    "Lighting": "Daytime", 
    "Camera": "Valid", 
    "Scene": "Urban", 
    "Speed": "Under 50"
}

scenario_night = {
    "Weather": "Dry", 
    "Lighting": "Nighttime", 
    "Camera": "Valid", 
    "Scene": "Urban", 
    "Speed": "Under 50"
}

scenarios_tested = [scenario_standard, scenario_fog, scenario_night]

print("--- ODD Coverage Results ---")
for k in [1, 2, 3]:
    cov = KProjectionCoverage(k=k, desc=odd_description)
    cov.add_scenarios(scenarios_tested)
    result = cov.compute()
    
    percentage = result.coverage * 100
    print(f"k={k} Coverage: {percentage:.2f}% (Covered {result.covered} out of {result.total} projected points)")