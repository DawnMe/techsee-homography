import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import copy

def minRectForBool(boolMask):        
        indices = np.where(boolMask)
        coordinates = [[y,x] for x,y in zip(indices[0], indices[1])]
        rect = cv2.boxPoints( cv2.minAreaRect(np.array(coordinates)) )  
        intrect = np.int0( rect ).tolist()
        return intrect

def processSegmentation( pred, dictColorValues, ports_order ):
    # find the left-most and right-most port in the image
    l = next((x for x in ports_order if x in np.ravel(pred)), None)
    r = next((x for x in ports_order[::-1] if x in np.ravel(pred)), None)
    ml = [0]*len(ports_order)  
    lindex = ports_order.index(l); rindex = ports_order.index(r)
    ml[lindex] = 1; ml[rindex] = 1

    # find the min area rect for the prediction and ground truth
    ports_expected = ports_order[lindex:rindex+1]
    boolMask = np.in1d(pred, ports_expected).reshape(pred.shape)  
    bpred = minRectForBool(boolMask)    

    #     # plot contours
    #     cntImg = img.copy()
    #     cv2.drawContours(cntImg, [np.array(bpred)], -1, (255, 0,0), 50)   

    # get the ports names from the left and right values
    lPort = [k for k,v in dictColorValues.items() if v==l][0]
    rPort = [k for k,v in dictColorValues.items() if v==r][0]
    
    return boolMask, bpred, lPort, rPort
    
def jsonLoad(jsonFilename):
    ''' load the model image and ports of the modem from json file '''
    data = json.load(open(jsonFilename))    
    firstKey = list(data.keys())[0]
    jsonDict = data[firstKey]
    
    # save the base folder and update model image filename
    basedFolderParts = re.split('\\\\|/',jsonFilename)
    baseFolder = '' if len(basedFolderParts)<=1 else basedFolderParts[0]+'/'
    jsonDict['baseFolder'] = baseFolder    
    jsonDict['filename'] = jsonDict['baseFolder'] + jsonDict['filename']
    
    return jsonDict

def plotPorts(jsonDict, img, cntrColor = (0,0,255)):
    ''' plot the ports on the model image of the modem'''
    contourImg = img.copy()    
    cntrW = int(0.01*np.max(contourImg.shape[:2]))     
    
    for k,vals in jsonDict['regions'].items():
        portType = vals['region_attributes']['name']
        portShape = vals['shape_attributes']
        shapeName = portShape['name']   
        if shapeName == 'rect':
            y,w,x,h = [portShape[x] for x in ['y', 'width', 'x', 'height']]
            cv2.rectangle(contourImg, (x,y), (x+w,y+h), cntrColor, cntrW)                
        elif shapeName == 'circle':
            cy,cx,r = [portShape[x] for x in ['cy', 'cx', 'r']]
            cv2.circle(contourImg, (cx, cy), r, cntrColor, cntrW)
        elif shapeName == 'list':
            listVals = portShape['values']
            cv2.drawContours(contourImg, [np.int0(listVals)], -1, cntrColor, cntrW)
            
    return contourImg

def transformPorts(jsonDict, M):
    ''' transform the josn shapes with the transformation M '''
   
    doTransform = lambda x: [np.int0( cv2.perspectiveTransform( np.array(x, dtype='float32').reshape(-1,1,2) , M )  )]
 
    newDict = copy.deepcopy(jsonDict)
    for k,vals in newDict['regions'].items():
        
        portType = vals['region_attributes']['name']
        portShape = vals['shape_attributes']
        shapeName = portShape['name']   
        
        if shapeName == 'rect':
            y,w,x,h = [portShape[x] for x in ['y', 'width', 'x', 'height']]                
            listPrev = [[y,x], [y, x+w], [y+h, x+w], [y+h, x]]
            listt = doTransform(listPrev)[0].squeeze().tolist()   
            newDict['regions'][k]['shape_attributes'] = {'name': 'list', 'values': listt }
            #             print(newDict['regions'][k]['shape_attributes'], '\n')
        elif shapeName == 'circle':
            cy,cx,r = [portShape[x] for x in ['cy', 'cx', 'r']]
            cxt, cyt = doTransform([cy,cx])[0].squeeze().tolist()
            cxp, cyp = doTransform([cy,cx+r])[0].squeeze().tolist()
            newr = int(np.linalg.norm( np.array([cxt, cyt])-np.array([cxp, cyp]) ) )
            newDict['regions'][k]['shape_attributes'] = { 'name': 'circle', 'cy':cyp, 'cx':cxp, 'r': newr }
            
    return newDict

def extremePorts(portShape):    
    ''' 
    return the top left, top right, bottom right, bottom left points of a shape.
    Only works for straight images.
    '''
    shapeName = portShape['name']       
    if shapeName == 'rect':
        y,w,x,h = [portShape[x] for x in ['y', 'width', 'x', 'height']]
        tl, tr, br, bl = [[y,x], [y, x+w], [y+h, x+w], [y+h, x]]
    elif shapeName == 'circle':
        cy,cx,r = [portShape[x] for x in ['cy', 'cx', 'r']]
        # left, up, right, bottom = [ [cy, cx-r], [cy-r, cx], [cy, cx+r], [cy+r, cx] ]
        tl, tr, br, bl = [ [cy-r, cx-r], [cy-r, cx+r], [cy+r, cx+r], [cy+r, cx-r] ]
    return tl, tr, br, bl

def getBoundingPoints(jsonDict, leftPort, rightPort):
    ''' 
    get the ports contrours and the left and right ports, 
    and return the top left, top right, bottom right, bottom left points.
    '''       
    portShapeLeft  = [vals['shape_attributes'] for k,vals in jsonDict['regions'].items() \
                      if vals['region_attributes']['name']==leftPort][0]
    portShapeRight = [vals['shape_attributes'] for k,vals in jsonDict['regions'].items() \
                      if vals['region_attributes']['name']==rightPort][0]
    tl, _, _, bl = extremePorts( portShapeLeft )    
    _, tr, br, _ = extremePorts( portShapeRight )
    
    ymin = min(tl[0], tr[0])
    ymax = max(br[0], bl[0])
    
    tlf, trf, brf, blf = tl, tr, br, bl
    tlf[0] = ymin; trf[0] = ymin;
    brf[0] = ymax; blf[0] = ymax;
    
    return tlf, trf, brf, blf 
    
def getHomography( predicted ):
    # class definitions
    dictColorValues = dict(background=0, modem=1, dsl=2, usb=3, power=4, switch=5, lan=6)
    ports_order = [2,3,6,4,5] # left to right:  dsl, usb, lan, power, switch
    jsonFileName = './VtechNB403.json'

    # json and model image
    jsonDict = jsonLoad(jsonFileName)
    imgModel = cv2.cvtColor(cv2.imread(jsonDict['filename']), cv2.COLOR_BGR2RGB)
    portsImg = plotPorts(jsonDict, imgModel)

    # process segmentation for homography
    boolMask, cntpred, leftPort, rightPort = processSegmentation( predicted, dictColorValues, ports_order)

    # homography and ports contours
    tl, tr, br, bl = getBoundingPoints(jsonDict, leftPort, rightPort)
    goodM, _ = cv2.findHomography(np.array([tl, tr, br, bl]), np.array(cntpred), cv2.RANSAC, 5.0)

    # transform the dictionary for plotting
    #updatedDict2 = transformPorts(jsonDict, goodM)
    #transImg2 = plotPorts(updatedDict2, imgRGB)

    return cntpred, goodM