# By: Justin Wei
# Date: April 25, 2017
# Description: computes Principal Component Analysis (PCA) on a given
# video by converting each frame into a different data dimension

from sklearn.decomposition import PCA
import numpy as np
import cv2

number_components = 8
scale = 0.35
def main():
    array, width, height = buildArray('./short.mov',scale)
    pca_components = perform_pca(array)
    scores = np.matmul(pca_components,array)

    nframe = array.shape[1]
    i = 1
    for iframe in range(0, nframe):

        row1,row2,row3 = multiply_pca_by_scores(pca_components, scores[:,iframe], (height,width))
        original = array[:,iframe].reshape((height,width))
        row1 = np.hstack((original,row1))
        final = np.vstack((row1,row2))
        final = np.vstack((final,row3))

        cv2.imshow('Components', final)
        
        cv2.waitKey(50)

    cv2.destroyAllWindows()

def multiply_pca_by_scores(pca_components, scores, frameShape):
    numberComponents = scores.shape[0]
    
    for val in range(0,numberComponents):
        img = scores[val]*pca_components[val,:] + 50
        img = np.clip(img,0,255)
        img = img.astype('uint8')
        img = img.reshape(frameShape)
        
        if val == 0:
            row1 = img
        elif val == 2:
            row2 = img
        elif val == 5:
            row3 = img
        else:
            if val < 2:
                row1 = np.hstack((row1,img))
            elif val < 5:
                row2 = np.hstack((row2,img))
            else:
                row3 = np.hstack((row3,img))
        
    return row1,row2,row3

    
def buildArray(videoPath,scale):
    video = cv2.VideoCapture(videoPath)

    firstFrame = True
    while(video.isOpened()):
        ret, frame = video.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height,width = frame.shape
            frame = cv2.resize(frame, (int(width*scale),int(height*scale)))
            frame = frame.reshape(frame.size,1)
                
            if firstFrame:
                array = frame
                firstFrame = False
            else:
                array = np.hstack((array,frame))
        else:
            video.release()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()
    cv2.destroyAllWindows()

    return array, int(width*scale), int(height*scale)

def perform_pca(data):
    pca = PCA(n_components=number_components)
    pca.fit(np.matrix.transpose(data))
    return pca.components_
    
    
main()
