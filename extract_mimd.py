import zmq
import json
import time

def query_cognitive_mesh():
    context = zmq.Context()
    
    # 1. Connect to the Request-Reply socket (Port 5555 from your logs)
    print("🔌 Connecting to Cognitive Mesh on port 5555...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    # List of likely commands based on your method names
    commands = [
        {"action": "get_status"},       # Standard status check
        {"action": "introspect"},       # From your cognitive_intelligent_system.py
        {"action": "get_rules"},        # Specific target
        {"command": "report"}           # Generic fallback
    ]
    
    for cmd in commands:
        try:
            print(f"?? Sending command: {cmd}")
            socket.send_json(cmd)
            
            # Wait for reply with timeout
            if socket.poll(2000): # 2 second timeout
                message = socket.recv_json()
                print("?? Response received!")
                analyze_response(message)
                break # Stop if we got a valid response
            else:
                print("? No response (timed out). Trying next command...")
                # Reset socket on timeout
                socket.close()
                socket = context.socket(zmq.REQ)
                socket.connect("tcp://localhost:5555")
                
        except Exception as e:
            print(f"? Error: {e}")

def analyze_response(data):
    """Parses the JSON response to find the learned rules."""
    
    # 1. Look for Rules
    rules = data.get('reasoning', {}).get('rules', []) or data.get('rules', [])
    if rules:
        print(f"\n?? FOUND {len(rules)} LEARNED RULES:")
        for i, rule in enumerate(rules):
            # Handle both dictionary and string representations
            if isinstance(rule, dict):
                antecedents = ", ".join(rule.get('antecedents', []))
                consequent = rule.get('consequent', 'UNKNOWN')
                conf = rule.get('confidence', 0.0)
                print(f"   {i+1}. IF [{antecedents}] THEN {consequent} (Confidence: {conf:.2f})")
            else:
                print(f"   {i+1}. {rule}")
    else:
        print("\n? No explicit rules found in response structure.")

    # 2. Look for Concepts
    concepts = data.get('abstraction', {}).get('concepts_by_level', {}) or data.get('concepts', {})
    if concepts:
        print(f"\n?? ACTIVE CONCEPTS:")
        print(json.dumps(concepts, indent=2))
        
    # 3. Look for Cross-Domain Transfers
    transfers = data.get('cognitive_metrics', {}).get('knowledge_transfers', 0)
    print(f"\n?? KNOWLEDGE TRANSFERS: {transfers}")
    
    # 4. Dump full raw data to file just in case
    with open('mind_dump.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\n?? Full state saved to 'mind_dump.json'")

if __name__ == "__main__":
    query_cognitive_mesh()
  
