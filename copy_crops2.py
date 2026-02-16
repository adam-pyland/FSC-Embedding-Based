import os
import shutil
import cv2
import matplotlib
# Use Qt5Agg or TkAgg for interaction - standard for Linux/Docker environments

    
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') 
from pathlib import Path

# --- CONFIGURATION ---
src_dir = "/work/Sultan/ultralytics/runs/detect/runs/rotem_pred_for_resnet_12_2_26/crops/vehicle/"
dst_dir = "/work/Sultan/data/for_resnet/0/"
os.makedirs(dst_dir, exist_ok=True)

# Filter for images
images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
images.sort()

class CropSelector:
    def __init__(self, image_list):
        self.image_list = image_list
        self.batch_size = 50 
        self.rows = 5
        self.cols = 10
        self.current_idx = 41000
        
    def on_click(self, event):
        if event.inaxes is None:
            return
        
        img_name = event.inaxes.get_title()
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"COPIED: {img_name}")
            
            # Visual feedback
            event.inaxes.patch.set_facecolor('red')
            event.inaxes.set_alpha(0.3)
            event.canvas.draw()

    def show_next_batch(self):
        if self.current_idx >= len(self.image_list):
            print("Finished all images!")
            plt.close('all')
            return

        plt.clf()
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        batch = self.image_list[self.current_idx : self.current_idx + self.batch_size]
        
        for i, img_name in enumerate(batch):
            ax = fig.add_subplot(self.rows, self.cols, i + 1)
            
            # Checkerboard background logic
            row, col = divmod(i, self.cols)
            color = '#f0f0f0' if (row + col) % 2 == 0 else '#ffffff'
            ax.set_facecolor(color)
            
            img_path = os.path.join(src_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            
            ax.set_title(img_name, fontsize=5, pad=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.suptitle(f"Batch {self.current_idx//50 + 1}. CLICK to copy. CLOSE window for next batch.", fontsize=12)
        plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.92, wspace=0.05, hspace=0.15)
        
        print(f"Showing batch starting at index {self.current_idx}. Close the window to see more.")
        plt.show() # This blocks until you close the window
        
        self.current_idx += self.batch_size
        self.show_next_batch()

if __name__ == '__main__':
    if not images:
        print(f"No images found in {src_dir}")
    else:
        # Create figure and run
        plt.figure(figsize=(20, 10))
        selector = CropSelector(images)
        selector.show_next_batch()