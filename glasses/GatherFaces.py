"""
This module reads a video input and saves a rectangular
face patch as a jpg file.

How to start it:

How to use it:
saving -> press 'a'
fix rectangle position (centered) -> press 'f'
exit -> press 'q'
"""

import cv2
import argparse
import pafy
import time
from CommonGlasses import CvFaceCapture

# parse arg input
parser = argparse.ArgumentParser(prog='GatherFaces',
                    description='The program detects faces and user an safe them (key a)')
parser.add_argument('-f', '--videofile', required=False, default=None)
parser.add_argument('-y', '--youtube', required=False, default=None)
args = parser.parse_args()

# open video source
cv_video_input = None
if(args.videofile != None):
    print("Read from file:", args.videofile)
    cv_video_input = cv2.VideoCapture(args.videofile)
elif(args.youtube != None):
    print("Stream from youtube:", args.youtube)
    #yvid = pafy.new(args.youtube)
    #print("Title:", yvid.title)
    #cv_video_input = cv2.VideoCapture('https://rr5---sn-o097znz7.googlevideo.com/videoplayback?expire=1678942265&ei=2UsSZOfWF_qDsfIPqbOkwAM&ip=193.233.231.169&id=o-AOBSKGDoygmCVl2M1CRsSYtijc9RQ40xfIrn_4tZA-WI&itag=22&source=youtube&requiressl=yes&mh=xu&mm=31%2C29&mn=sn-o097znz7%2Csn-n4v7snly&ms=au%2Crdu&mv=m&mvi=5&pl=22&initcwndbps=1162500&vprv=1&mime=video%2Fmp4&ns=e6xuHtFBd2_P2JFa-s1412ML&cnr=14&ratebypass=yes&dur=10393.878&lmt=1572319500258251&mt=1678920306&fvip=3&fexp=24007246&c=WEB&txp=1306222&n=8TAL3RMEc8yMyw&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cvprv%2Cmime%2Cns%2Ccnr%2Cratebypass%2Cdur%2Clmt&sig=AOq0QJ8wRAIgE_wlVbgLER3AQVT7zfAdi9HTScOXD1eLZKmEwt6-kh8CIC-Wij6qL_vAnDmHA4xyRQTmT7VdBgowZuE2x-4nzAWL&lsparams=mh%2Cmm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpl%2Cinitcwndbps&lsig=AG3C_xAwRQIgLi6XBSHg3DvZVI3uG-hgZYdFX10pXxvuvD-3cw2tOfICIQC8Urk2qzz-e_MO8-XSoKXzbaFM16d6K_0vWelWZ5EsZg%3D%3D&title=Richard+Jones%3A+Introduction+to+game+programming+-+PyCon+2014')
else:
    print("Stream from camera:")
    cv_video_input = cv2.VideoCapture(0)

capture = CvFaceCapture(cv_video_input)
output_path = 'data/current/'

addFaceToFile = False
fixRect = False
count = 0
x, y, w, h = 0, 0, 5, 5

while(True):
    frame, faces = capture.read()
    if len(faces) > 0 or fixRect:

        # update the rect pos with the actual location of the current face
        if not fixRect:
            [x, y, w, h] = faces[0]

        rect_color = (255, 0, 0)
        if addFaceToFile:
            # save the current face image
            face = frame[y:y+h, x:x+w]
            file_name = output_path + str(time.time()*1000) + ".jpg"
            cv2.imwrite(file_name, face)
            count += 1
            addFaceToFile = False
            rect_color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Number of samples ' + str(count), (40, 40), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    keyCode = cv2.waitKey(100)
    if keyCode == ord('q'):
        break
    if keyCode == ord('a'):
        addFaceToFile = not addFaceToFile
    if keyCode == ord('f'):
        fixRect = not fixRect

cv_video_input.release()
cv2.destroyAllWindows()
