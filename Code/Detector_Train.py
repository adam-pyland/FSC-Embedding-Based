from ultralytics import YOLO
import os

def main():
    # Define your paths cleanly
    dataset_yaml = r'C:\Adams\Vehicle_Dataset\data.yaml'
    output_dir = r'C:\Adams\FSOD\Codes\FSC-Embedding-Based\Outputs\YOLO26_Train_FAIR1M'
    run_name = 'fair1m_vehicle_training'
    
    # Target the 'last.pt' file for resuming (it contains the epoch & optimizer state)
    last_weights_path = os.path.join(output_dir, run_name, 'weights', 'last.pt')
    
    # 1. Check if an interrupted run exists
    if os.path.exists(last_weights_path):
        print("Found previous training state. Resuming YOLO Training...")
        model = YOLO(last_weights_path)
        
        # When resuming, you ONLY pass resume=True. 
        # YOLO reads args.yaml in the output directory to remember your epochs, imgsz, etc.
        model.train(resume=True)
        
    else:
        print("No previous run found. Starting YOLO26 Training from scratch...")
        base_model_path = r'C:\Adams\FSOD\Codes\FSC-Embedding-Based\yolo26n-obb.pt'
        model = YOLO(base_model_path) 
        
        # 2. Start fresh training with all your parameters
        model.train(
            data=dataset_yaml,
            epochs=100,             
            imgsz=1024,             
            batch=16,               
            device=0,               
            project=output_dir,     
            name=run_name,
            save=True,              
            val=True,               
            plots=True              
        )

    print("Training Complete. Starting isolated Validation...")
    
    # 3. Explicitly load the BEST model for the final validation
    # This ensures we don't validate using the "last" epoch if an earlier epoch was better
    best_weights_path = os.path.join(output_dir, run_name, 'weights', 'best.pt')
    best_model = YOLO(best_weights_path)

    # Outputs are saved to: C:\Adams\FSOD\Codes\FSC-Embedding-Based\Outputs\fair1m_vehicle_validation\
    metrics = best_model.val(
        data=dataset_yaml,      
        split='val',
        project=output_dir,     
        name='fair1m_vehicle_validation'
    )

    # Print out core validation metrics
    print("\n--- Validation Results ---")
    print(f"Mean Average Precision (mAP@50): {metrics.box.map50:.4f}")
    print(f"Mean Average Precision (mAP@50-95): {metrics.box.map:.4f}")
    print(f"Results saved to: {os.path.join(output_dir, 'fair1m_vehicle_validation')}")

if __name__ == '__main__':
    main()