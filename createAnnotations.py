import time
import json
import argparse
import os
from operator import itemgetter
import csv
##Format annotations to x,y,width,height
CLASSES=['ignored regions','pedestrian', 'people', 'bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--gtpath", dest = 'groundtruth', help = 
                        "Path to groundtruth",
                        default = "./annotations", type = str)
    parser.add_argument("--imgpath", dest = 'images',help = 
                        "Path to images")
    return parser.parse_args()
if __name__ == '__main__':
    args = arg_parse()
    gt_path = args.groundtruth
    #images_path = args.images
    ground_truth = []
    csvFile = open('./'+os.path.basename(os.path.dirname(gt_path))+'.csv','w')
    writer = csv.writer(csvFile)
    for filename in os.listdir(gt_path):
        ground_truth = []
        frameID = 0
        filePath = os.path.join(gt_path,filename)
        if (os.path.isfile(filePath)):
            with open(os.path.join(gt_path,filename), 'r') as f:
                for line in f:
                    (frame_id,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion ) = map(int,(line.split(',')))
                    if(object_category!=0 and object_category !=11):
                        ground_truth.append(list((frame_id,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion)))
            sortedList = sorted(ground_truth, key=itemgetter(0))
            for obj in sortedList:
                current_FrameID = obj[0]
                bbox_left = obj[2]
                bbox_top = obj[3]
                bbox_width = obj[4]
                bbox_height = obj[5]
                object_category = obj[7]
                video = os.path.splitext(filename)[0]
                row = [str(os.path.abspath(os.path.dirname(os.path.dirname(gt_path))+'/sequences/{}/{}.jpg'.format(video,str(current_FrameID).zfill(7)))),str(bbox_left),str(bbox_top),str(bbox_left+bbox_width),str(bbox_top+bbox_height),str(CLASSES[int(object_category)])]
                writer.writerow(row)
                #txtFile.write(str(os.path.abspath('../sequences/{}/{}.jpg'.format(video,str(current_FrameID).zfill(7))))+' '+str(bbox_left)+ ' ' +str(bbox_top)+ ' ' +str(bbox_left+bbox_width)+ ' ' +str(bbox_top+bbox_height)+' '+str(CLASSES[int(object_category)]))
                #txtFile.write('\n')
            #     pathtoDir = os.path.join(gt_path,os.path.splitext(filename)[0])
            #     if(current_FrameID>frameID):
            #         frameID = current_FrameID
            #         if not os.path.exists(pathtoDir):
            #             os.makedirs(pathtoDir)
            #         txtFile = open(os.path.join(pathtoDir,str(current_FrameID)+'.txt'),'w')
            #     txtFile.write("0"+ ' ' +str(bbox_left)+ ' ' +str(bbox_top)+ ' ' +str(bbox_width)+ ' ' +str(bbox_height))
            #     txtFile.write('\n')
            # print("Total frames in video {} is {}".format(os.path.splitext(filename)[0],frameID))