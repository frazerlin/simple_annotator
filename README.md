# SimpleAnnotator
A simple annotator for labeling images or repairing labels.

## Requirement

- python 3.x
- cv2 (opencv-python) 
- matplotlib  

## Usage
#### Annotate Mode:
- "a": change to mode [annotate] (a polygon labeling mode);  
- left button: draw a polygon, set key points (it will end when a point near the initial one);  
- backspace:  retract the last stroke of the polygon;
#### Bbox Mode:
- "b": change to mode [bbox];  
- left button: draw a bounding box as label;  

#### Change Mode:
- "c": change to mode [change];  
- left button: change the label of target instance to set label;  
- right button: change the target label of the whole image to set label;  

#### Set  Label:
- "x": (x in [0,9]) : set a label (x), 0 can be used as eraser;  
- "`": set a label (255);  
- "-": auto set a unused label;  

#### Others:
- "ctrl+z": revocation;  
- "esc": quit;  
- "enter": save output in [--out] and quit;  


#### Label a new image:

```
python simple_annotator.py --img img.jpg --out out.png 
```

#### Repair a label:

```
python simple_annotator.py --img img.jpg --gt gt.png --out out.png
```


## Contact

If you have any questions, feel free to contact me via: `frazer.linzheng(at)gmail.com`.

## More

For more about interactive segmentation or useful annotation tools, feel free to visit my [homepage](https://www.lin-zheng.com).

