from burr_with_mcp import application

if __name__ == "__main__":
    # Create an instance of the application
    app = application()
    
    # Generate the visualization
    app.visualize(
        output_file_path="statemachine", 
        include_conditions=False, 
        include_state=False, 
        format="png"
    )
    
    print("FSM visualization generated: statemachine.png")