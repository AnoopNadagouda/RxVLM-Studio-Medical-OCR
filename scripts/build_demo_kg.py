import os
import json
from pathlib import Path
import networkx as nx
from pyvis.network import Network

def build_clinical_knowledge_graph():
    print("=" * 70)
    print("Building Stage III Clinical Knowledge Graph Visualizer")
    print("=" * 70)

    # 1. Sample Structured Extractions from OCR VLM Stage 2/3 Outputs
    prescriptions = [
        {
            "doctor": "Dr. A. K. Sharma (MD, Cardiology)",
            "patient": "Patient #1042 (Male, 48y)",
            "medications": [
                {
                    "name": "Paracetamol",
                    "strength": "500 mg",
                    "schedule": "1-0-1 (After Meals)",
                    "duration": "5 Days"
                },
                {
                    "name": "Amoxicillin",
                    "strength": "250 mg",
                    "schedule": "1-1-1 (Every 8h)",
                    "duration": "7 Days"
                },
                {
                    "name": "Pantoprazole",
                    "strength": "40 mg",
                    "schedule": "1-0-0 (Before Breakfast)",
                    "duration": "14 Days"
                }
            ]
        },
        {
            "doctor": "Dr. S. Roy (MBBS, General Medicine)",
            "patient": "Patient #1089 (Female, 34y)",
            "medications": [
                {
                    "name": "Metformin",
                    "strength": "500 mg",
                    "schedule": "0-0-1 (Night)",
                    "duration": "30 Days"
                },
                {
                    "name": "Atorvastatin",
                    "strength": "10 mg",
                    "schedule": "0-0-1 (Night)",
                    "duration": "30 Days"
                }
            ]
        }
    ]

    # 2. Initialize NetworkX Graph
    G = nx.DiGraph()

    # Define Node Aesthetics (Color & Icon Shapes)
    NODE_STYLES = {
        "Doctor": {"color": "#FF6B6B", "shape": "star", "size": 25},
        "Patient": {"color": "#4D96FF", "shape": "diamond", "size": 22},
        "Medication": {"color": "#6BCB77", "shape": "box", "size": 20},
        "Dosage_Strength": {"color": "#FFD93D", "shape": "ellipse", "size": 16},
        "Frequency_Schedule": {"color": "#9B51E0", "shape": "dot", "size": 16}
    }

    # 3. Populate Nodes and Edges
    for rx in prescriptions:
        doc_name = rx["doctor"]
        pat_name = rx["patient"]

        # Add Doctor Node
        G.add_node(doc_name, label=doc_name, type="Doctor", title=f"Prescribing Doctor: {doc_name}", **NODE_STYLES["Doctor"])
        
        # Add Patient Node
        G.add_node(pat_name, label=pat_name, type="Patient", title=f"Patient Record: {pat_name}", **NODE_STYLES["Patient"])
        
        # Edge: PRESCRIBED_BY (Patient -> Doctor)
        G.add_edge(pat_name, doc_name, label="PRESCRIBED_BY", color="#FF6B6B", title="Patient treated by Doctor")

        for med in rx["medications"]:
            med_name = med["name"]
            strength = f"{med_name} {med['strength']}"
            schedule = f"{med['schedule']} ({med['duration']})"

            # Add Medication Node
            G.add_node(med_name, label=med_name, type="Medication", title=f"Drug Name: {med_name}", **NODE_STYLES["Medication"])
            
            # Edge: PRESCRIBED_TO (Doctor -> Medication) & (Patient -> Medication)
            G.add_edge(doc_name, med_name, label="PRESCRIBES", color="#4D96FF", title="Doctor prescribed Drug")
            G.add_edge(pat_name, med_name, label="TAKES", color="#4D96FF", title="Patient taking Drug")

            # Add Dosage Strength Node
            G.add_node(strength, label=med['strength'], type="Dosage_Strength", title=f"Strength: {med['strength']}", **NODE_STYLES["Dosage_Strength"])
            G.add_edge(med_name, strength, label="HAS_STRENGTH", color="#FFD93D", title="Drug Concentration")

            # Add Frequency Schedule Node
            G.add_node(schedule, label=med['schedule'], type="Frequency_Schedule", title=f"Schedule: {med['schedule']} for {med['duration']}", **NODE_STYLES["Frequency_Schedule"])
            G.add_edge(med_name, schedule, label="TAKEN_SCHEDULE", color="#9B51E0", title="Dosing Frequency")

    print(f"Graph Construction Complete: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")

    # 4. Export to Interactive Pyvis HTML
    net = Network(height="750px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
    net.from_nx(G)

    # Enable smooth physics & dark mode styling
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "color": "#ffffff",
          "size": 14,
          "face": "Inter, sans-serif"
        }
      },
      "edges": {
        "color": { "inherit": false },
        "smooth": { "type": "continuous" },
        "font": { "color": "#cbd5e1", "size": 11, "align": "middle" }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    output_dir = Path("./reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html = output_dir / "demo_knowledge_graph.html"
    
    net.write_html(str(output_html))
    print(f"\n[Success] Interactive Knowledge Graph successfully exported to: {output_html.resolve()}")

if __name__ == "__main__":
    build_clinical_knowledge_graph()
