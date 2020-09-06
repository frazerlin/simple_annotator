
import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Annotator(object):
    def __init__(self,img,gt=None,save_file=None):
        self.PAL=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32,0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128,224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32,128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]
        self.cmap= mpl.colors.LinearSegmentedColormap.from_list('cmap', self.__get_list_cmap(), 256)
        self.mode_dict={'n':'none', 'a':'annotate','b':'bbox','c':'change'}
        self.label_dict={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0,'`':255}
        self.save_file=save_file

        self.img_src = np.array(Image.open(img)) if isinstance(img,str) else img
        self.gt_src= np.zeros(self.img_src.shape[:2],dtype=np.uint8) if gt is None else (np.array(Image.open(gt)) if isinstance(gt,str) else gt)

        self.gt_cur=self.gt_src.copy()
        self.merge_cur=cv2.addWeighted(self.img_src,0.5,self.__label_to_rgb(self.gt_cur),0.5,0)  
        self.gt_backup=[self.gt_cur.copy()]

        self.mode= 'annotate'  
        self.label= 1
        self.polygon=[]
        self.press_state=False
    
    def __get_list_cmap(self):
        cmap=[]
        for i in range(0,len(self.PAL),3):
            cmap.append('#{}{}{}'.format( hex(self.PAL[i])[2:].zfill(2),hex(self.PAL[i+1])[2:].zfill(2),hex(self.PAL[i+2])[2:].zfill(2)))
        return cmap

    def __label_to_rgb(self,label):
        rgb=np.zeros((label.shape[0],label.shape[1],3),dtype=np.uint8)
        index_set=set(label.flat)
        for index in index_set:
            rgb[label==index]=self.PAL[index*3:index*3+3]
        return rgb

    def __update_title(self):
        self.fig.suptitle('[{}][{}]'.format(self.mode,self.label),fontsize=16)
        self.ax1.set_title('Image')
        self.ax1.axis('off')
        self.ax2.set_title('Label')
        self.ax2.axis('off')
        self.fig.canvas.draw()

    def __update_merge(self):
        self.merge_cur=cv2.addWeighted(self.img_src,0.5,self.__label_to_rgb(self.gt_cur),0.5,0)

    def __update_show(self):  
        self.ax1.imshow(self.merge_cur)
        self.ax2.imshow(self.gt_cur,cmap=self.cmap,norm=mpl.colors.Normalize(vmin=0,vmax=255))
        self.fig.canvas.draw()
        
    def __update(self):
        self.__update_title()
        self.__update_merge()
        self.__update_show()
        
    def __on_key_press(self,event):
        if event.key=='-':
            ins_set= set(self.gt_cur.flat)
            for label in range(1,256):
                if label not in ins_set:
                    break
            self.label=label
            self.__update_title()
            
        elif event.key in self.label_dict:
            self.label=self.label_dict[event.key]
            self.__update_title()
            
        elif event.key in self.mode_dict:
            self.mode=self.mode_dict[event.key]
            self.__update_title()
            
        elif event.key=='ctrl+z':
            self.gt_cur=self.gt_backup[-1].copy()
            self.gt_backup.pop()
            self.__update()

        elif event.key=='backspace':
            if len(self.polygon)!=0:
                self.polygon.pop()
                self.ax1.lines.pop()
                if len(self.polygon)!=0:
                    self.ax1.lines.pop()
                self.__update_show()

        elif event.key=='escape':
            self.fig.canvas.mpl_disconnect(self.cid_bre)
            plt.close()

        elif event.key=='enter':
            if self.save_file is not None:
                gt=Image.fromarray(self.gt_cur)
                gt.putpalette(self.PAL)
                gt.save(self.save_file)
            self.fig.canvas.mpl_disconnect(self.cid_bre)
            plt.close()

    def __on_button_press(self,event):
        self.press_state=True
        if not event.inaxes:return
        if self.mode=='annotate':
            if event.button==1:
                x,y= int(event.xdata+0.5), int(event.ydata+0.5)
                if len(self.polygon)>=3 and  np.sum((np.array([x,y])-self.polygon[0])**2)<100:
                    self.gt_backup.append(self.gt_cur.copy())
                    cv2.fillPoly(self.gt_cur,np.array(self.polygon)[None,:,:],self.label)
                    self.polygon=[]
                    self.ax1.cla()
                    self.__update()
                else:
                    self.ax1.plot(x,y,'ro')
                    if len(self.polygon)!=0:self.ax1.plot((self.polygon[-1][0],x),(self.polygon[-1][1],y),'g')
                    self.polygon.append([x,y])
                    self.fig.canvas.draw()

        elif self.mode=='bbox':
            if event.button==1:
                self.bbox_spt=[int(event.xdata+0.5), int(event.ydata+0.5)]
                self.press_state=True
                self.ax1.cla()
                self.__update()

        elif self.mode=='change':
            if event.button==1:
                x,y= int(event.xdata+0.5), int(event.ydata+0.5)
                label_target=self.gt_cur[y,x]
                h, w = self.gt_cur.shape
                mask = np.zeros((h+2, w+2), np.uint8)
                flooded=self.gt_cur.copy()
                cv2.floodFill(flooded,mask,(x,y),254)
                self.gt_backup.append(self.gt_cur.copy())
                self.gt_cur[flooded==254]=self.label
                self.ax1.cla()
                self.__update()
            
            elif event.button==3:
                x,y= int(event.xdata+0.5), int(event.ydata+0.5)
                label_target=self.gt_cur[y,x]
                self.gt_backup.append(self.gt_cur.copy())
                self.gt_cur[self.gt_cur==label_target]=self.label
                self.ax1.cla()
                self.__update()

    def __on_motion_notify(self,event):
        if not event.inaxes:return
        if self.mode=='bbox' and self.press_state:
            spt,ept=self.bbox_spt,[int(event.xdata+0.5), int(event.ydata+0.5)]
            self.ax1.patches=[]
            self.ax1.add_patch(patches.Rectangle(spt,ept[0]-spt[0],ept[1]-spt[1],color='orange',fill=False,linewidth=2))
            self.fig.canvas.draw()

    def __on_button_release(self,event):
        self.press_state=False
        if not event.inaxes:return
        if self.mode=='bbox':
            spt,ept=self.bbox_spt,[int(event.xdata+0.5), int(event.ydata+0.5)]
            cv2.rectangle(self.gt_cur,tuple(spt),tuple(ept),self.label,-1)
            self.ax1.clear()
            self.ax2.clear()
            self.__update_title()
            self.__update_merge()
            self.__update_show()
        
    def main(self):
        self.fig = plt.figure('Annotator',figsize=(20,12))
        self.cid_kpe=self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.cid_bpe=self.fig.canvas.mpl_connect("button_press_event",  self.__on_button_press)
        self.cid_mne=self.fig.canvas.mpl_connect('motion_notify_event', self.__on_motion_notify)
        self.cid_bre=self.fig.canvas.mpl_connect("button_release_event",  self.__on_button_release)
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.ax1.axis('off')
        self.ax1.set_title('Image')
        self.ax2 = self.fig.add_subplot(1,2,2)
        self.ax2.axis('off')
        self.ax2.set_title('Label')
        self.__update()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Annotator")
    parser.add_argument('--img', type=str, default='img.jpg',help='source image')
    parser.add_argument('--gt', type=str, default=None, help='repair target')
    parser.add_argument('--out', type=str, default='out.png', help='output label')
    args = parser.parse_args()
    anno=Annotator(img=args.img,gt=args.gt,save_file=args.out)
    anno.main()
