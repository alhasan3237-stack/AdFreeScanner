import cv2, numpy as np, glob, os, time, v3
v3.NET_CAND=False; v3.NET_W=0.3
net = cv2.dnn.readNetFromONNX(os.path.expanduser("~/net/pagenet.onnx"))
def prob_for(img):
    net.setInput(cv2.dnn.blobFromImage(img, 1.0, (96, 128), (0,0,0), False, False)); return net.forward()[0,0]
tiles=[]; T=[]
for f in sorted(glob.glob(os.path.expanduser("~/real/set/r*"))):
    nm=os.path.basename(f)[:3]
    if nm in {"r08","r23","r24"}: continue
    im=cv2.imread(f); h,w=im.shape[:2]; s=400/max(h,w); sm=cv2.resize(im,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)
    t0=time.time(); p=prob_for(sm); q=v3.detect_color(sm,prob=p); T.append(time.time()-t0)
    t=sm.copy(); H,W=t.shape[:2]
    if q is not None: cv2.polylines(t,[np.int32(np.array(q)*[W,H])],True,(0,200,0),2)
    c=v3.LAST_CONF or (0,1,0); cv2.putText(t,f"{nm} n{(c[2] or 0):.2f}",(3,16),0,0.45,(0,0,255),1)
    X=np.zeros((400,400,3),np.uint8); X[:H,:W]=t; tiles.append(X)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(os.path.expanduser("~/real/live.jpg"),np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]))
print("avg ms",np.mean(T)*1000)
