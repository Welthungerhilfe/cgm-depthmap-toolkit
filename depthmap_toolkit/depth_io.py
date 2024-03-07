
import numpy as np
import struct
from zipfile import ZipFile

def load_depth_file(filename: str, useScale: bool = False, debug: bool = False) -> tuple[np.array, np.array, np.array, np.array]:
    """function that loads a zipped depth image file that can be generated in CGM app
        when storing data on the phone memory. The format is from Lubos with image
        size and position info followed by data and separated by a line break
        
        Parameters:
        - filename: the file path to be loaded
        - useScale: will return the depth image in meters, otherwise in millimeters
        - debug: will print out some byte info for the loaded file
        
         Returns:
        - image with depth data
        - confidence map for the depth image
        - rotation array as 4 floats 
        - position position 3 floats
    """
    try:
        with ZipFile(filename) as archive:
            for item in archive.namelist():
                
                # look for file 'data' which is composed of 
                # 'info' + '\n' + 'data'
                if item == 'data':
                    bytes = archive.read(item)

                    # split data at first line feed '\n' - there may be more in image
                    info, data = bytes.split(b'\n', 1)
                    info = info.decode('ASCII') 

                    # split info which consists of 
                    # size = 'width x height'
                    # scale = 0.001
                    # magic=7 - 7 values will follow that represent pose
                    # pose = rotation (4 floats) and position (3 floats)
                    size, scale, magic, pose = info.split('_',3)

                    positionData = np.array(pose.split('_'), dtype=float)

                    if len(positionData) == magic:
                        print(f"File error: magic number should be 7 but read: {magic}") 
                        break;
                    
                    rotation = np.array(positionData[0:4])
                    position = np.array(positionData[4:8])
                    
                    width, height = (int(x) for x in size.split('x'))

                    if debug:    
                        print(info)

                        print(f'read of bytes: {len(bytes)}')
                        print(f'length of info: {len(info)}')
                        print(f'length of data: {len(data)}')

                        print(f'height: {height} - width: {width} - pose: {pose}')

                    #---------------------------------------------------------------
                    # # manual way
                    # format = 'BBB'
                    # dataList = list(struct.iter_unpack(format, data)) # data[0:(width*height*3)]))
                    # array = np.array(dataList)

                    # imageData = []
                    # for t in dataList:
                    #     # compose high and low byte of depth data (type: short)
                    #     t0shift = t[0] << 8 
                    #     value = t0shift + t[1] 
                    #     confidence = t[2]
                    #     imageData.append(value)

                    # image = np.array(imageData).reshape(height,width)
                    # confidence = array[:,2].reshape(height,width)
                    #---------------------------------------------------------------

                    # unpack way
                    # format of data: depth (H) as unsigned short followed by confidence (B) as unsigned char
                    format = '>HB'
                    dataList = list(struct.iter_unpack(format, data))
                    array = np.array(dataList)

                    image = array[:,0].reshape(height,width)
                    confidence = array[:,1].reshape(height,width)
                    
                    if useScale:
                        image = image * float(scale) # apply original image scale in meters

                    return image, confidence, rotation, position
    except:
        print(f"Error reading file: {filename}") 