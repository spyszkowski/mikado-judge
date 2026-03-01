PS C:\Users\spyszkowski\Workspaces\experiments\mikado-judge> 


#convert cvat to obb 
python .\scripts\lines_to_obb.py --cvat-xml "C:\Users\spyszkowski\Downloads\project_3_dataset_2026_03_01_17_40_50_cvat for images 1.1\annotations.xml" --output data\labels --classes configs\classes.yaml
INFO: Loaded 5 classes, thickness config: {'mikado': 10, 'blue': 7, 'red': 7, 'yellow': 7, 'green': 7, 'default': 7}

Converted 68 images, 681 sticks total
  blue: 127
  red: 107
  yellow: 182
  green: 265

Labels written → data\labels


#preview
python scripts\visualize_obb.py --images frames/  --labels data/labels/  --classes configs/classes.yaml  --interactive

#prepare training set
python scripts/prepare_dataset.py  --frames frames/  --labels data/labels/  --output data/dataset/  --val-ratio 0.2
INFO: Found 68 image/label pairs

Dataset statistics:
  Train: 54 images
  Val:   14 images
  Total: 68 images
  Train sessions: 54
  Val sessions:   14