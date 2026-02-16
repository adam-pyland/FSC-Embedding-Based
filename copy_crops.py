import os
import shutil
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
src_dir = "/work/Sultan/ultralytics/runs/detect/runs/rotem_pred_for_resnet_12_2_26/crops/vehicle/"
dst_dir = "/work/Sultan/data/for_resnet/0/"
os.makedirs(dst_dir, exist_ok=True)

images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
images.sort()

class CropSelector:
    def __init__(self, image_list):
        self.image_list = image_list
        self.batch_size = 50  # Increased to 50
        self.rows = 5
        self.cols = 10
        self.current_idx = 0
        
    def on_click(self, event):
        if event.inaxes is None:
            return
        
        img_name = event.inaxes.get_title()
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"COPIED: {img_name}")
            
            # Feedback: Turn the background red and dim the image
            event.inaxes.patch.set_facecolor('red')
            event.inaxes.set_alpha(0.3)
            event.canvas.draw()

    def show_next_batch(self):
        if self.current_idx >= len(self.image_list):
            print("Done!")
            return

        plt.clf()
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        batch = self.image_list[self.current_idx : self.current_idx + self.batch_size]
        
        for i, img_name in enumerate(batch):
            row = i // self.cols
            col = i % self.cols
            
            ax = fig.add_subplot(self.rows, self.cols, i + 1)
            
            # Create Checkerboard Pattern
            # If row+col is even, light gray; if odd, white
            color = '#f0f0f0' if (row + col) % 2 == 0 else '#ffffff'
            ax.set_facecolor(color)
            
            img = cv2.imread(os.path.join(src_dir, img_name))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            
            # Using very small font to keep title from eating space
            ax.set_title(img_name, fontsize=5, pad=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Show the background color at the edges
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.suptitle(f"50-Image Batch: {self.current_idx//50 + 1}. Click to Keep. Close to Continue.", fontsize=12)
        
        # Maximize area: left, bottom, right, top, wspace, hspace
        plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.92, wspace=0.05, hspace=0.15)
        plt.show()
        
        self.current_idx += self.batch_size
        self.show_next_batch()

# Run
plt.figure(figsize=(20, 10)) # Wide aspect ratio for 10 columns
selector = CropSelector(images)
selector.show_next_batch()