
# Ontology Mock
# In the real system, this queries an OWL reasoner.
# Here we use a dictionary to simulate the "Semantic Layer".

ONTOLOGY_MAP = {
    # Pump 1 (Example tags)
    'FIT101': {'device': 'Pump P101', 'type': 'Flow Sensor', 'related_cmd': 'Start/Stop P101 (FC05)'},
    'LIT101': {'device': 'Tank T101', 'type': 'Level Sensor', 'related_cmd': 'Valve Control / Pump Logic'},
    'P101':   {'device': 'Pump P101', 'type': 'Pump Status', 'related_cmd': 'Write Coil (FC05)'},
    'MV101':  {'device': 'Valve MV101', 'type': 'Motorized Valve', 'related_cmd': 'Open/Close (FC05)'},
    
    # Pump 2 / Stage 2 (from paper)
    'F_PU2':  {'device': 'Pump P201', 'type': 'Flow Sensor', 'related_cmd': 'Start/Stop P201 (FC05)'}, # Simulated tag name from paper
    'P_J289': {'device': 'Pipe J289', 'type': 'Pressure Sensor', 'related_cmd': 'Pump P201 Logic'},
    'L_T1':   {'device': 'Tank T101', 'type': 'Level Sensor', 'related_cmd': 'Fill T101'},
    
    # SWaT Actual Tags (Subset)
    'FIT201': {'device': 'Pump P201', 'type': 'Flow', 'related_cmd': 'FC05 P201'},
    'AIT201': {'device': 'Analyzer 201', 'type': 'Conductivity', 'related_cmd': 'Check Chemical Dosing'},
    'P201':   {'device': 'Pump P201', 'type': 'Status', 'related_cmd': 'Write Coil P201'},
}

def semantic_lookup(feature_name):
    """
    Returns semantic info for a given feature.
    """
    # Clean name
    clean = feature_name.strip()
    return ONTOLOGY_MAP.get(clean, {'device': 'Unknown', 'type': 'Sensor', 'related_cmd': 'None'})

def generate_alert(shap_results):
    """
    Generates a protocol-aware alert based on top SHAP features.
    Args:
        shap_results: List of (feature_name, importance)
    """
    print("\n--- OntoXAI-LSTM Alert ---")
    detected = False
    
    # Group by related command
    cmd_scores = {}
    
    for feat, score in shap_results:
        info = semantic_lookup(feat)
        cmd = info['related_cmd']
        device = info['device']
        
        if cmd not in cmd_scores:
            cmd_scores[cmd] = 0.0
        cmd_scores[cmd] += score
        
        # print(f"  > Feature: {feat} ({info['type']}) | SHAP: {score:.4f} | -> {cmd}")

    # Find dominating command
    if cmd_scores:
        top_cmd = max(cmd_scores, key=cmd_scores.get)
        total_score = cmd_scores[top_cmd]
        
        if total_score > 0.1: # Threshold
             print(f"CRITICAL: Anomalous behavior linked to: {top_cmd}")
             print(f"Accumulated Attribution: {total_score:.4f}")
             detected = True
        else:
             print("Warning: Low confidence anomaly.")
    
    if not detected:
        print("No definitive protocol mapping found.")
    print("--------------------------\n")
