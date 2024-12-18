import cv2
import numpy as np
import math



def rafayset(rectangles):
    
    pairs = []
    used_indices = set()

    for i in range(len(rectangles)):
        if i in used_indices:
            continue

        x1, y1, w1, h1 = rectangles[i]
        centroid1 = (x1 + w1 / 2, y1 + h1 / 2)

        min_distance = float('inf')
        closest_index = None

        for j in range(len(rectangles)):
            if i == j or j in used_indices:
                continue

            x2, y2, w2, h2 = rectangles[j]
            centroid2 = (x2 + w2 / 2, y2 + h2 / 2)

            distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_index = j

        if closest_index is not None:
            pairs.append((rectangles[i], rectangles[closest_index]))
            used_indices.add(i)
            used_indices.add(closest_index)

    return pairs


def crops(frame,bb):
    
    cn=[]
    
    boxes = rafayset(bb)
    print(len(boxes))
    
    for i in range(0,len(boxes)):
        
        print(boxes[i][0],boxes[i][1])
        
        x,y,w,h=boxes[i][1]
        x2,y2,w2,h2=boxes[i][0]
        
        temp = frame[y:(y2+h2),x:(x2+w2)]
        
        #print(frame.shape,bb[i],sp1,sp2)
        
        rgb= frame.copy()
        #rgb=cv2.rectangle(rgb, (x, y), (x+w2,y+h2), (255, 0, 0), 2)
        
        
        #rgb=cv2.circle(rgb, (x,y), 10, (100,200,0), 10)
        #rgb=cv2.circle(rgb, (x2,(y2)), 10, (100,200,0), 10)
        dx=x-x2
        dy=y-y2
        rgb=cv2.rectangle(rgb, (x,h), (w,y2), (0, 255, 244), 2)
        
        np=0
        maxn=-1
        for i in range(h,y2):
            #print(rgb[h+np,x+4])
            if(rgb[h+np,x+4][0]>rgb[h+np,x+4][1] and rgb[h+np,x+4][0]>rgb[h+np,x+4][2]):
                rgb=cv2.circle(rgb, (x+4,h+np), 2, (200,0,200), 2)
                if(maxn<np):
                    maxn=np
            np+=1
            
        
        print(y2-(maxn+h),(y2-h),((y2-(maxn+h))/(y2-h))*100)
        cv2.imshow("temp",rgb)
        cv2.waitKey(0)
        
    
    
def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Broaden the lower and upper thresholds for blue color in HSV
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([135, 255, 255])
    
    # Create a binary mask for blue regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Black out everything except the blue regions
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)
    
    # Find contours for the blue regions
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest blue contour
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    if largest_contour is not None:
        # Get the bounding box of the largest blue contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding box
        
        # Define the lower and upper thresholds for red color in HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Create a binary mask for red regions within the entire frame
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Apply Gaussian blur to reduce noise and improve contour detection
        red_mask = cv2.GaussianBlur(red_mask, (7, 7), 0)
        
        # Apply dilation to merge small regions and avoid tiny detections
        red_mask = cv2.dilate(red_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        # Find contours for the red regions
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum contour area to filter out small, unnecessary regions
        min_contour_area = 500  # Adjust this value as needed
        
        # List to store bounding boxes
        bounding_boxes = []
        
        # Find and store black bounding boxes around valid red regions within the entire frame
        for contour in red_contours:
            if cv2.contourArea(contour) > min_contour_area:
                bx, by, bw, bh = cv2.boundingRect(contour)
                bounding_boxes.append([bx, by, bx+bw, by+bh])
                #cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 0), 2)  # Draw black bounding box
        
        #try:
        
        

    # Check if the 'a' key is pressed (ASCII value of 'a' is 97)
        
        crops(frame,bounding_boxes)
        #except Exception as e:
        #    print(e)
            
    return frame


pic = cv2.imread('captured_image.jpg')

process_frame(pic)